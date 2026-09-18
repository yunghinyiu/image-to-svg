//! Eval harness for 1:1 tech-pack parity (Phase 1: measurement only).
//!
//! Compares the flat pipeline's SVG output against a reference technical
//! flat (black line art on white) and reports:
//!
//! - **silhouette IoU**: garment mask (backdrop keying) vs. the silhouette
//!   derived from the reference line art (gap-closed + flood-filled).
//! - **Chamfer distance**: symmetric mean nearest-neighbour distance between
//!   sampled ink points of our rasterized SVG and the reference ink.
//! - **button metrics**: round-blob detection on both sides, greedy matching,
//!   count + mean position error.
//! - **ink ratio**: how much linework we emit vs. the reference.
//!
//! The reference is a *redrawing*, not a photo trace, so global proportions
//! differ. All metrics first align our output to the reference with a
//! similarity transform (uniform scale + translation) fitted to the two
//! garment bounding boxes, then compare shape/detail — never raw canvas
//! position. Everything is computed on canvases with max side
//! [`EVAL_MAX_SIDE`] px.

use anyhow::{Context, Result};
use im2vec_flat::{convert_flat_bytes, flat_garment_mask, FlatOptions};
use image::{GrayImage, Luma, RgbImage};
use serde::Serialize;
use std::collections::HashMap;

/// Max side (px) of every canvas metrics are computed on.
pub const EVAL_MAX_SIDE: u32 = 1024;
/// Max ink points sampled per side for Chamfer.
pub const CHAMFER_MAX_POINTS: usize = 20_000;

// ---------------------------------------------------------------------------
// basic image ops
// ---------------------------------------------------------------------------

/// Binary ink mask: pixels darker than `thresh` -> 255.
pub fn ink_mask(gray: &GrayImage, thresh: u8) -> GrayImage {
    GrayImage::from_fn(gray.width(), gray.height(), |x, y| {
        Luma([if gray.get_pixel(x, y)[0] < thresh {
            255
        } else {
            0
        }])
    })
}

fn dilate_bool(src: &[bool], w: usize, h: usize, r: usize) -> Vec<bool> {
    let mut out = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            if !src[y * w + x] {
                continue;
            }
            let y0 = y.saturating_sub(r);
            let y1 = (y + r).min(h - 1);
            let x0 = x.saturating_sub(r);
            let x1 = (x + r).min(w - 1);
            for yy in y0..=y1 {
                for xx in x0..=x1 {
                    out[yy * w + xx] = true;
                }
            }
        }
    }
    out
}

/// Derive a silhouette mask from black-on-white line art: close small gaps
/// in the linework, flood-fill the background from the image borders through
/// non-ink pixels, invert. The final dilation compensates the ink growth from
/// gap closing so the silhouette keeps its true size.
pub fn silhouette_from_lineart(lineart: &GrayImage) -> GrayImage {
    let (w, h) = (lineart.width() as usize, lineart.height() as usize);
    let ink: Vec<bool> = lineart.pixels().map(|p| p[0] < 128).collect();
    let closed = dilate_bool(&ink, w, h, 2);

    // background = non-ink reachable from the borders
    let mut bg = vec![false; w * h];
    let mut stack: Vec<(usize, usize)> = Vec::new();
    for x in 0..w {
        stack.push((x, 0));
        stack.push((x, h - 1));
    }
    for y in 0..h {
        stack.push((0, y));
        stack.push((w - 1, y));
    }
    while let Some((x, y)) = stack.pop() {
        let i = y * w + x;
        if bg[i] || closed[i] {
            continue;
        }
        bg[i] = true;
        if x > 0 {
            stack.push((x - 1, y));
        }
        if x + 1 < w {
            stack.push((x + 1, y));
        }
        if y > 0 {
            stack.push((x, y - 1));
        }
        if y + 1 < h {
            stack.push((x, y + 1));
        }
    }

    let sil: Vec<bool> = bg.iter().map(|&b| !b).collect();
    let sil = dilate_bool(&sil, w, h, 2);
    GrayImage::from_raw(
        w as u32,
        h as u32,
        sil.iter().map(|&b| if b { 255u8 } else { 0 }).collect(),
    )
    .expect("silhouette image")
}

/// Tight bounding box of non-zero pixels, (x0, y0, x1, y1), x1/y1 exclusive.
pub fn bbox(mask: &GrayImage) -> Option<(u32, u32, u32, u32)> {
    let (w, h) = (mask.width(), mask.height());
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
    for (x, y, p) in mask.enumerate_pixels() {
        if p[0] > 0 {
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x + 1);
            y1 = y1.max(y + 1);
        }
    }
    if x1 > x0 && y1 > y0 {
        Some((x0, y0, x1, y1))
    } else {
        None
    }
}

/// Intersection-over-union of two same-size binary masks.
pub fn iou(a: &GrayImage, b: &GrayImage) -> f64 {
    assert_eq!((a.width(), a.height()), (b.width(), b.height()));
    let (mut inter, mut union) = (0u64, 0u64);
    for (pa, pb) in a.pixels().zip(b.pixels()) {
        let (x, y) = (pa[0] > 0, pb[0] > 0);
        if x && y {
            inter += 1;
        }
        if x || y {
            union += 1;
        }
    }
    if union == 0 {
        1.0
    } else {
        inter as f64 / union as f64
    }
}

// ---------------------------------------------------------------------------
// alignment
// ---------------------------------------------------------------------------

/// Uniform-scale + translation map, ours -> reference canvas coords.
#[derive(Debug, Clone, Copy)]
pub struct Similarity {
    pub scale: f32,
    pub tx: f32,
    pub ty: f32,
}

impl Similarity {
    pub fn apply(&self, p: (f32, f32)) -> (f32, f32) {
        (self.scale * p.0 + self.tx, self.scale * p.1 + self.ty)
    }
}

/// Fit our garment bbox onto the reference garment bbox: uniform scale by
/// the larger side, then align bbox centers.
pub fn align_bboxes(our_bb: (u32, u32, u32, u32), tgt_bb: (u32, u32, u32, u32)) -> Similarity {
    let (ox0, oy0, ox1, oy1) = (
        our_bb.0 as f32,
        our_bb.1 as f32,
        our_bb.2 as f32,
        our_bb.3 as f32,
    );
    let (tx0, ty0, tx1, ty1) = (
        tgt_bb.0 as f32,
        tgt_bb.1 as f32,
        tgt_bb.2 as f32,
        tgt_bb.3 as f32,
    );
    let ow = (ox1 - ox0).max(1.0);
    let oh = (oy1 - oy0).max(1.0);
    let tw = (tx1 - tx0).max(1.0);
    let th = (ty1 - ty0).max(1.0);
    let s = tw.max(th) / ow.max(oh);
    let (ocx, ocy) = ((ox0 + ox1) / 2.0, (oy0 + oy1) / 2.0);
    let (tcx, tcy) = ((tx0 + tx1) / 2.0, (ty0 + ty1) / 2.0);
    Similarity {
        scale: s,
        tx: tcx - s * ocx,
        ty: tcy - s * ocy,
    }
}

/// Warp a binary mask into a destination canvas with nearest-neighbour
/// sampling through the inverse similarity transform.
pub fn warp_mask_nearest(src: &GrayImage, sim: &Similarity, dw: u32, dh: u32) -> GrayImage {
    GrayImage::from_fn(dw, dh, |x, y| {
        let sx = (x as f32 - sim.tx) / sim.scale;
        let sy = (y as f32 - sim.ty) / sim.scale;
        let v = if sx >= 0.0 && sy >= 0.0 {
            let (ix, iy) = (sx as u32, sy as u32);
            if ix < src.width() && iy < src.height() {
                src.get_pixel(ix, iy)[0]
            } else {
                0
            }
        } else {
            0
        };
        Luma([v])
    })
}

// ---------------------------------------------------------------------------
// chamfer distance
// ---------------------------------------------------------------------------

/// Stride-sampled ink pixel centers, capped at `max_n` points.
pub fn sample_points(ink: &GrayImage, max_n: usize) -> Vec<(f32, f32)> {
    let (w, h) = (ink.width(), ink.height());
    let total = ink.pixels().filter(|p| p[0] > 0).count();
    if total == 0 {
        return Vec::new();
    }
    let stride = ((total as f64 / max_n as f64).sqrt().ceil() as u32).max(1);
    let mut pts = Vec::new();
    for y in (0..h).step_by(stride as usize) {
        for x in (0..w).step_by(stride as usize) {
            if ink.get_pixel(x, y)[0] > 0 {
                pts.push((x as f32 + 0.5, y as f32 + 0.5));
            }
        }
    }
    pts
}

/// Symmetric Chamfer: (mean ours->target, mean target->ours), in px.
pub fn chamfer(a: &[(f32, f32)], b: &[(f32, f32)]) -> (f64, f64) {
    (directed_chamfer(a, b), directed_chamfer(b, a))
}

fn directed_chamfer(a: &[(f32, f32)], b: &[(f32, f32)]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return f64::INFINITY;
    }
    const CELL: f32 = 16.0;
    let mut grid: HashMap<(i32, i32), Vec<(f32, f32)>> = HashMap::new();
    for &p in b {
        grid.entry(((p.0 / CELL).floor() as i32, (p.1 / CELL).floor() as i32))
            .or_default()
            .push(p);
    }
    let mut sum = 0f64;
    for &p in a {
        let (cx, cy) = ((p.0 / CELL).floor() as i32, (p.1 / CELL).floor() as i32);
        // Expand rings while a closer point could still hide in a farther
        // ring; guarantees the exact nearest neighbour.
        let mut best = f32::INFINITY;
        let mut ring = 0i32;
        while (ring as f32) * CELL < best && ring < 10_000 {
            for dx in -ring..=ring {
                for dy in -ring..=ring {
                    if ring > 0 && dx.abs() < ring && dy.abs() < ring {
                        continue;
                    }
                    if let Some(pts) = grid.get(&(cx + dx, cy + dy)) {
                        for &q in pts {
                            let d = (p.0 - q.0).hypot(p.1 - q.1);
                            if d < best {
                                best = d;
                            }
                        }
                    }
                }
            }
            ring += 1;
        }
        sum += best as f64;
    }
    sum / a.len() as f64
}

// ---------------------------------------------------------------------------
// button / blob detection
// ---------------------------------------------------------------------------

/// A detected round ink blob (button candidate).
#[derive(Debug, Clone, Serialize)]
pub struct Blob {
    pub cx: f32,
    pub cy: f32,
    pub radius: f32,
    pub area: usize,
    pub circularity: f32,
    pub radial_cv: f32,
}

/// Connected ink components filtered by area, bbox aspect and roundness.
/// Accepts both filled disks (perimeter circularity 4π·area/perimeter²) and
/// stroked rings (line-art buttons: pixels at a consistent radial distance
/// from the centroid, i.e. low coefficient of variation of the radial
/// distances). Returns centroid, equivalent radius, circularity, radial CV.
pub fn detect_blobs(
    ink: &GrayImage,
    min_area: usize,
    max_area: usize,
    min_circularity: f32,
) -> Vec<Blob> {
    let (w, h) = (ink.width() as usize, ink.height() as usize);
    let mut seen = vec![false; w * h];
    let mut blobs = Vec::new();

    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if seen[i] || ink.get_pixel(x as u32, y as u32)[0] == 0 {
                continue;
            }
            let mut stack = vec![(x, y)];
            seen[i] = true;
            let mut area = 0usize;
            let (mut sx, mut sy) = (0u64, 0u64);
            let (mut x0, mut y0, mut x1, mut y1) = (x, y, x, y);
            let mut perimeter = 0usize;
            // pixel coords, kept only while the component can still pass the
            // area filter (bounds memory for huge components)
            let mut pts: Vec<(u32, u32)> = Vec::new();
            while let Some((cx, cy)) = stack.pop() {
                area += 1;
                sx += cx as u64;
                sy += cy as u64;
                x0 = x0.min(cx);
                y0 = y0.min(cy);
                x1 = x1.max(cx);
                y1 = y1.max(cy);
                if area <= max_area {
                    pts.push((cx as u32, cy as u32));
                }
                let mut edge = false;
                let mut neighbours = [(0usize, 0usize); 4];
                let mut nn = 0;
                if cx > 0 {
                    neighbours[nn] = (cx - 1, cy);
                    nn += 1;
                } else {
                    edge = true;
                }
                if cx + 1 < w {
                    neighbours[nn] = (cx + 1, cy);
                    nn += 1;
                } else {
                    edge = true;
                }
                if cy > 0 {
                    neighbours[nn] = (cx, cy - 1);
                    nn += 1;
                } else {
                    edge = true;
                }
                if cy + 1 < h {
                    neighbours[nn] = (cx, cy + 1);
                    nn += 1;
                } else {
                    edge = true;
                }
                for &(nx, ny) in &neighbours[..nn] {
                    let j = ny * w + nx;
                    if ink.get_pixel(nx as u32, ny as u32)[0] == 0 {
                        edge = true;
                    } else if !seen[j] {
                        seen[j] = true;
                        stack.push((nx, ny));
                    }
                }
                if edge {
                    perimeter += 1;
                }
            }
            if area < min_area || area > max_area {
                continue;
            }
            let bw = (x1 - x0 + 1) as f32;
            let bh = (y1 - y0 + 1) as f32;
            let aspect = bw / bh.max(1.0);
            if !(0.65..=1.55).contains(&aspect) {
                continue;
            }
            let circularity = if perimeter > 0 {
                4.0 * std::f32::consts::PI * area as f32 / (perimeter as f32).powi(2)
            } else {
                0.0
            };
            // radial consistency around the centroid: filled disk ≈ 0.35,
            // thin ring ≈ 0, line/squiggle ≳ 0.55. Catches stroked rings
            // (line-art buttons) that perimeter circularity rejects.
            let (mcx, mcy) = (sx as f32 / area as f32, sy as f32 / area as f32);
            let (mut s1, mut s2) = (0.0f32, 0.0f32);
            for &(px, py) in &pts {
                let d = ((px as f32 - mcx).powi(2) + (py as f32 - mcy).powi(2)).sqrt();
                s1 += d;
                s2 += d * d;
            }
            let n = pts.len() as f32;
            let mu = s1 / n.max(1.0);
            let radial_cv = (s2 / n.max(1.0) - mu * mu).max(0.0).sqrt() / mu.max(1e-6);
            let radius = (area as f32 / std::f32::consts::PI).sqrt();
            let fill = area as f32 / (bw * bh).max(1.0);
            // Disks: perimeter circularity. Rings (line-art buttons, often
            // with 4-hole dots): pixels at a consistent radial distance from
            // the centroid. Thresholds assume near-native resolution, where
            // button rings are clean (cv ~0.05-0.1, fill ~0.2-0.3). The radius
            // cap rejects large roundish noise loops (cuffs, seam curls).
            let disk_like = circularity >= min_circularity && radius <= 20.0;
            let ring_like = radial_cv <= 0.35 && fill >= 0.12 && radius <= 20.0;
            if !(disk_like || ring_like) {
                continue;
            }
            blobs.push(Blob {
                cx: mcx,
                cy: mcy,
                radius,
                area,
                circularity,
                radial_cv,
            });
        }
    }
    blobs.sort_by(|a, b| {
        a.cy.partial_cmp(&b.cy)
            .unwrap()
            .then(a.cx.partial_cmp(&b.cx).unwrap())
    });
    blobs
}

/// Greedy nearest matching of our blobs to target blobs within `tol` px.
/// Returns (matched pairs, mean position error px or None when unmatched).
pub fn match_blobs(ours: &[Blob], target: &[Blob], tol: f32) -> (usize, Option<f64>) {
    let mut used = vec![false; target.len()];
    let mut matched = 0usize;
    let mut err_sum = 0f64;
    for o in ours {
        let mut best: Option<(usize, f32)> = None;
        for (j, t) in target.iter().enumerate() {
            if used[j] {
                continue;
            }
            let d = (o.cx - t.cx).hypot(o.cy - t.cy);
            if d <= tol && best.is_none_or(|(_, bd)| d < bd) {
                best = Some((j, d));
            }
        }
        if let Some((j, d)) = best {
            used[j] = true;
            matched += 1;
            err_sum += d as f64;
        }
    }
    let mean_err = if matched > 0 {
        Some(err_sum / matched as f64)
    } else {
        None
    };
    (matched, mean_err)
}

// ---------------------------------------------------------------------------
// svg rasterization
// ---------------------------------------------------------------------------

/// Render an SVG string to RGBA at max side `max_side` px (white background).
pub fn rasterize_svg(svg: &str, max_side: u32) -> Result<image::RgbaImage> {
    let tree = resvg::usvg::Tree::from_data(svg.as_bytes(), &resvg::usvg::Options::default())
        .map_err(|e| anyhow::anyhow!("parse svg: {e}"))?;
    let size = tree.size();
    anyhow::ensure!(
        size.width() > 0.0 && size.height() > 0.0,
        "empty svg viewport"
    );
    let scale = max_side as f32 / size.width().max(size.height());
    let (w, h) = (
        (size.width() * scale).round().max(1.0) as u32,
        (size.height() * scale).round().max(1.0) as u32,
    );
    let mut pixmap = resvg::tiny_skia::Pixmap::new(w, h)
        .ok_or_else(|| anyhow::anyhow!("pixmap alloc {w}x{h}"))?;
    pixmap.fill(resvg::tiny_skia::Color::WHITE);
    resvg::render(
        &tree,
        resvg::tiny_skia::Transform::from_scale(scale, scale),
        &mut pixmap.as_mut(),
    );
    image::RgbaImage::from_raw(w, h, pixmap.take())
        .ok_or_else(|| anyhow::anyhow!("pixmap -> RgbaImage"))
}

// ---------------------------------------------------------------------------
// full eval
// ---------------------------------------------------------------------------

/// All quantitative metrics for one input/target pair.
#[derive(Debug, Clone, Serialize)]
pub struct EvalReport {
    /// IoU of our garment mask vs. reference silhouette (bbox-aligned).
    pub silhouette_iou: f64,
    /// Symmetric Chamfer distance of ink point sets, px at eval scale.
    pub chamfer_px: f64,
    pub chamfer_ours_to_target_px: f64,
    pub chamfer_target_to_ours_px: f64,
    /// chamfer_px / eval canvas max side.
    pub chamfer_normalized: f64,
    /// ours ink px / reference ink px.
    pub ink_ratio_ours_to_target: f64,
    pub ink_pixels_ours: usize,
    pub ink_pixels_target: usize,
    pub path_count: usize,
    pub buttons_ours: usize,
    pub buttons_target: usize,
    pub buttons_matched: usize,
    pub button_position_error_px: Option<f64>,
    pub button_match_tolerance_px: f32,
    pub eval_canvas_target: [u32; 2],
    pub eval_canvas_ours: [u32; 2],
    pub svg_size: [u32; 2],
    /// garment bbox, normalized 0..1 in each canvas's own coords (web overlay).
    pub ours_bbox_norm: [f32; 4],
    pub target_bbox_norm: [f32; 4],
    /// detected button blobs (target canvas coords for ours after alignment).
    pub blobs_ours: Vec<Blob>,
    pub blobs_target: Vec<Blob>,
}

/// Everything `run_eval` produces: metrics + the SVG + visual diffs.
pub struct EvalArtifacts {
    pub report: EvalReport,
    pub svg: String,
    /// 3-panel PNG: input photo | our SVG | reference flat.
    pub diff_png: Vec<u8>,
    /// Aligned onion-skin PNG: reference ink black, our ink red.
    pub onion_png: Vec<u8>,
}

fn resize_max_side(
    gray: &GrayImage,
    max_side: u32,
    filter: image::imageops::FilterType,
) -> GrayImage {
    let (w, h) = (gray.width(), gray.height());
    if w.max(h) <= max_side {
        return gray.clone();
    }
    let s = max_side as f32 / w.max(h) as f32;
    image::imageops::resize(
        gray,
        (w as f32 * s).round() as u32,
        (h as f32 * s).round() as u32,
        filter,
    )
}

fn norm_bbox(bb: (u32, u32, u32, u32), w: u32, h: u32) -> [f32; 4] {
    [
        bb.0 as f32 / w as f32,
        bb.1 as f32 / h as f32,
        bb.2 as f32 / w as f32,
        bb.3 as f32 / h as f32,
    ]
}

/// Run the flat pipeline on `input_png`, compare against the reference flat
/// `target_png`, and return metrics + visual diffs. Measurement only: the
/// pipeline itself is untouched.
pub fn run_eval(input_png: &[u8], target_png: &[u8], opts: &FlatOptions) -> Result<EvalArtifacts> {
    use image::imageops::FilterType;

    // 1. our pipeline output + garment mask
    let flat = convert_flat_bytes(input_png, opts)?;
    let mask_full = flat_garment_mask(input_png, opts)?;
    let (sw, sh) = (flat.width, flat.height);

    // 2. reference prep (eval canvas)
    let target_full = image::load_from_memory(target_png)
        .context("decode target")?
        .to_luma8();
    let target = resize_max_side(&target_full, EVAL_MAX_SIDE, FilterType::Triangle);
    let (tw, th) = (target.width(), target.height());
    let target_ink = ink_mask(&target, 128);
    let target_sil = silhouette_from_lineart(&target);
    let target_pts = sample_points(&target_ink, CHAMFER_MAX_POINTS);
    let target_ink_n = target_ink.pixels().filter(|p| p[0] > 0).count();

    // 3. our output on the eval canvas
    let ours_rgba = rasterize_svg(&flat.svg, EVAL_MAX_SIDE)?;
    let (ow, oh) = (ours_rgba.width(), ours_rgba.height());
    let ours_gray: GrayImage = image::imageops::grayscale(&ours_rgba);
    let ours_ink = ink_mask(&ours_gray, 200);
    let ours_pts = sample_points(&ours_ink, CHAMFER_MAX_POINTS);
    let ours_ink_n = ours_ink.pixels().filter(|p| p[0] > 0).count();
    let ours_mask = image::imageops::resize(&mask_full, ow, oh, FilterType::Nearest);
    anyhow::ensure!(
        mask_full.width() == sw && mask_full.height() == sh,
        "mask/svg size mismatch"
    );

    // 4. bbox alignment (uniform scale + translation, ours -> target)
    let obb = bbox(&ours_mask).context("empty garment mask")?;
    let tbb = bbox(&target_sil).context("empty reference silhouette")?;
    let sim = align_bboxes(obb, tbb);

    // 5. silhouette IoU on the target canvas
    let warped = warp_mask_nearest(&ours_mask, &sim, tw, th);
    let silhouette_iou = iou(&warped, &target_sil);

    // 6. chamfer on aligned ink point sets
    let ours_aligned: Vec<(f32, f32)> = ours_pts.iter().map(|&p| sim.apply(p)).collect();
    let (c_ot, c_to) = chamfer(&ours_aligned, &target_pts);
    let chamfer_px = (c_ot + c_to) / 2.0;

    // 7. buttons — detect at native resolution, where small stroked rings
    // survive (the 1024px eval canvas merges rings with their 4-hole dots).
    // Positions are scaled into the target eval-canvas frame for matching.
    let tol = 0.03 * tw.max(th) as f32;
    let target_ink_native = ink_mask(&target_full, 128);
    let ours_native_rgba = rasterize_svg(&flat.svg, sw.max(sh))?;
    let (nw, _nh) = (ours_native_rgba.width(), ours_native_rgba.height());
    let ours_gray_native: GrayImage = image::imageops::grayscale(&ours_native_rgba);
    let ours_ink_native = ink_mask(&ours_gray_native, 200);
    let scale_t = tw as f32 / target_full.width() as f32;
    let scale_o = ow as f32 / nw as f32;
    let target_blobs: Vec<Blob> = detect_blobs(&target_ink_native, 40, 9000, 0.55)
        .into_iter()
        .map(|b| Blob {
            cx: b.cx * scale_t,
            cy: b.cy * scale_t,
            radius: b.radius * scale_t,
            ..b
        })
        .collect();
    let ours_blobs: Vec<Blob> = detect_blobs(&ours_ink_native, 40, 9000, 0.55)
        .into_iter()
        .map(|b| {
            let (cx, cy) = sim.apply((b.cx * scale_o, b.cy * scale_o));
            Blob {
                cx,
                cy,
                radius: b.radius * scale_o * sim.scale,
                ..b
            }
        })
        .collect();
    let (matched, pos_err) = match_blobs(&ours_blobs, &target_blobs, tol);

    let report = EvalReport {
        silhouette_iou,
        chamfer_px,
        chamfer_ours_to_target_px: c_ot,
        chamfer_target_to_ours_px: c_to,
        chamfer_normalized: chamfer_px / tw.max(th) as f64,
        ink_ratio_ours_to_target: ours_ink_n as f64 / target_ink_n.max(1) as f64,
        ink_pixels_ours: ours_ink_n,
        ink_pixels_target: target_ink_n,
        path_count: flat.path_count,
        buttons_ours: ours_blobs.len(),
        buttons_target: target_blobs.len(),
        buttons_matched: matched,
        button_position_error_px: pos_err,
        button_match_tolerance_px: tol,
        eval_canvas_target: [tw, th],
        eval_canvas_ours: [ow, oh],
        svg_size: [sw, sh],
        ours_bbox_norm: norm_bbox(obb, ow, oh),
        target_bbox_norm: norm_bbox(tbb, tw, th),
        blobs_ours: ours_blobs,
        blobs_target: target_blobs,
    };

    // 8. visual diffs
    let input_img = image::load_from_memory(input_png)
        .context("decode input")?
        .to_rgb8();
    let diff_png = diff_panels(&input_img, &ours_rgba, &target_full)?;
    let onion_png = onion_overlay(&target_ink, &ours_ink, &sim)?;

    Ok(EvalArtifacts {
        report,
        svg: flat.svg,
        diff_png,
        onion_png,
    })
}

fn panel_height(img: &RgbImage, h: u32) -> RgbImage {
    let (w0, h0) = (img.width(), img.height());
    let w = ((w0 as f32 * h as f32) / h0 as f32).round() as u32;
    image::imageops::resize(img, w.max(1), h, image::imageops::FilterType::Triangle)
}

/// 3-panel PNG: input photo | our SVG | reference flat (fixed panel order).
fn diff_panels(input: &RgbImage, ours: &image::RgbaImage, target: &GrayImage) -> Result<Vec<u8>> {
    const PH: u32 = 480;
    const GUTTER: u32 = 8;
    let ours_rgb = image::DynamicImage::ImageRgba8(ours.clone()).to_rgb8();
    let target_rgb = image::DynamicImage::ImageLuma8(target.clone()).to_rgb8();
    let panels = [
        panel_height(input, PH),
        panel_height(&ours_rgb, PH),
        panel_height(&target_rgb, PH),
    ];
    let total_w: u32 = panels.iter().map(|p| p.width()).sum::<u32>() + GUTTER * 2;
    let mut canvas = RgbImage::from_pixel(total_w, PH, image::Rgb([255, 255, 255]));
    let mut x = 0;
    for p in &panels {
        image::imageops::replace(&mut canvas, p, x as i64, 0);
        x += p.width() + GUTTER;
    }
    let mut buf = Vec::new();
    canvas.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)?;
    Ok(buf)
}

/// Onion-skin: reference ink black, our (aligned) ink red, on white.
fn onion_overlay(
    target_ink: &GrayImage,
    ours_ink: &GrayImage,
    sim: &Similarity,
) -> Result<Vec<u8>> {
    let (tw, th) = (target_ink.width(), target_ink.height());
    let ours_warped = warp_mask_nearest(ours_ink, sim, tw, th);
    let canvas = RgbImage::from_fn(tw, th, |x, y| {
        let t = target_ink.get_pixel(x, y)[0] > 0;
        let o = ours_warped.get_pixel(x, y)[0] > 0;
        if t {
            image::Rgb([20, 20, 20])
        } else if o {
            image::Rgb([220, 40, 40])
        } else {
            image::Rgb([255, 255, 255])
        }
    });
    let mut buf = Vec::new();
    canvas.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blank(w: u32, h: u32) -> GrayImage {
        GrayImage::new(w, h)
    }

    fn rect_img(w: u32, h: u32, x0: u32, y0: u32, x1: u32, y1: u32) -> GrayImage {
        let mut img = blank(w, h);
        for y in y0..y1 {
            for x in x0..x1 {
                img.put_pixel(x, y, Luma([255]));
            }
        }
        img
    }

    #[test]
    fn iou_identical_is_one() {
        let a = rect_img(20, 20, 5, 5, 15, 15);
        assert!((iou(&a, &a) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn iou_half_overlap() {
        let a = rect_img(20, 20, 0, 0, 10, 20);
        let b = rect_img(20, 20, 5, 0, 15, 20);
        // inter 100, union 300
        assert!((iou(&a, &b) - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn chamfer_identical_is_zero() {
        let pts = vec![(0.0, 0.0), (10.0, 5.0), (3.0, 7.0)];
        let (x, y) = chamfer(&pts, &pts);
        assert!(x < 1e-6 && y < 1e-6);
    }

    #[test]
    fn chamfer_shifted_is_shift() {
        let a = vec![(0.0, 0.0)];
        let b = vec![(3.0, 4.0)];
        let (x, y) = chamfer(&a, &b);
        assert!((x - 5.0).abs() < 1e-6 && (y - 5.0).abs() < 1e-6);
    }

    #[test]
    fn silhouette_fills_closed_lineart() {
        // 40x40, 2px black square outline -> silhouette should be ~ the square interior+outline
        let mut art = GrayImage::from_pixel(40, 40, Luma([255u8]));
        for x in 10..30 {
            art.put_pixel(x, 10, Luma([0]));
            art.put_pixel(x, 29, Luma([0]));
        }
        for y in 10..30 {
            art.put_pixel(10, y, Luma([0]));
            art.put_pixel(29, y, Luma([0]));
        }
        let sil = silhouette_from_lineart(&art);
        let n: u32 = sil.pixels().map(|p| (p[0] > 0) as u32).sum();
        // 20x20 square + 2px dilation compensation on each side -> ~28x28
        assert!(n > 600 && n < 900, "filled square, got {n}");
        assert_eq!(sil.get_pixel(0, 0)[0], 0, "outside stays background");
        assert_eq!(sil.get_pixel(20, 20)[0], 255, "inside filled");
    }

    #[test]
    fn align_bboxes_maps_centers() {
        let sim = align_bboxes((0, 0, 100, 200), (0, 0, 50, 100));
        assert!((sim.scale - 0.5).abs() < 1e-6);
        let (cx, cy) = sim.apply((50.0, 100.0));
        assert!((cx - 25.0).abs() < 1e-6 && (cy - 50.0).abs() < 1e-6);
    }

    #[test]
    fn detect_blobs_finds_circle() {
        // filled circle r=12 at (30,30) in 64x64
        let mut img = blank(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                let dx = x as i32 - 30;
                let dy = y as i32 - 30;
                if dx * dx + dy * dy <= 144 {
                    img.put_pixel(x, y, Luma([255]));
                }
            }
        }
        let blobs = detect_blobs(&img, 50, 2000, 0.5);
        assert_eq!(blobs.len(), 1, "one circle, got {}", blobs.len());
        let b = &blobs[0];
        assert!((b.cx - 30.0).abs() < 1.5 && (b.cy - 30.0).abs() < 1.5);
        assert!((b.radius - 12.0).abs() < 1.5, "r={}", b.radius);
    }

    #[test]
    fn detect_blobs_finds_ring() {
        // stroked circle r=10, ~2px stroke: a ring, like line-art buttons
        let mut img = blank(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                let d = (((x as i32 - 30).pow(2) + (y as i32 - 30).pow(2)) as f32).sqrt();
                if (d - 10.0).abs() <= 1.0 {
                    img.put_pixel(x, y, Luma([255]));
                }
            }
        }
        let blobs = detect_blobs(&img, 40, 2000, 0.5);
        assert_eq!(blobs.len(), 1, "one ring, got {}", blobs.len());
        let b = &blobs[0];
        assert!(b.radial_cv < 0.3, "ring cv={}", b.radial_cv);
        assert!((b.cx - 30.0).abs() < 1.5 && (b.cy - 30.0).abs() < 1.5);
    }

    #[test]
    fn detect_blobs_ignores_lines() {
        // long thin bar: not a blob
        let bar = rect_img(64, 64, 5, 30, 60, 33);
        let blobs = detect_blobs(&bar, 50, 5000, 0.5);
        assert!(blobs.is_empty(), "bar is not round");
    }
}
