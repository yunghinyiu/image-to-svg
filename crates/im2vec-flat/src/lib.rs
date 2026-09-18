//! Clothing photo -> tech-pack flat sketch (draft quality).
//!
//! Phase 1 is fully deterministic and targets flat-lay / ghost-mannequin
//! photos on plain backdrops:
//!
//! 1. background keying (backdrop color estimated from border pixels)
//! 2. XDoG stylized line extraction for seams, folds, trims
//! 3. per-view mirror symmetrization around each garment view's own
//!    center axis (multi-view front/side/back inputs keep every view;
//!    asymmetric side views pass through untouched)
//! 4. two vtracer passes (silhouette + detail linework) composed into one
//!    flat-style SVG: white garment, dark outline stroke, dark inner lines.
//!
//! On-model photos need an ML segmenter (Phase 2: segformer-b2-clothes via
//! ONNX) and are rejected with a clear error until then.

use anyhow::{bail, Context, Result};
use im2vec_core::{convert_image, ColorImage, ConvertOptions, ImPreset};
use image::{GrayImage, ImageFormat, RgbImage};
use serde::Serialize;
use std::io::Cursor;
use std::time::Instant;

/// Max image side in px; larger inputs are downscaled for speed.
const MAX_SIDE: u32 = 1600;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlatInput {
    FlatLay,
    OnModel,
}

impl FlatInput {
    pub fn parse(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "on-model" | "onmodel" | "model" => FlatInput::OnModel,
            _ => FlatInput::FlatLay,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FlatOptions {
    pub input: FlatInput,
    /// Mirror-average around the vertical center axis. Garments are
    /// symmetric; this removes asymmetric wrinkles/shadows.
    pub symmetrize: bool,
    /// Mirror the detail linework too. Off by default: asymmetric details
    /// (chest logos, pockets) must stay where they are, not be duplicated.
    pub symmetrize_lines: bool,
    /// Render inner details as dashed stitch strokes (factory convention:
    /// solid = seam, dashed = stitching). Off = solid uniform strokes.
    pub stitch_dashed: bool,
    /// Outline stroke width in px on the silhouette path.
    pub outline_width: f32,
    /// 0..=1. Higher keeps weaker lines (fabric folds); lower keeps only
    /// strong edges (seams, hems, trims).
    pub detail_strength: f32,
    /// vtracer speckle filter side length for both passes.
    pub speckle: usize,
}

impl Default for FlatOptions {
    fn default() -> Self {
        Self {
            input: FlatInput::FlatLay,
            symmetrize: true,
            symmetrize_lines: false,
            stitch_dashed: false,
            outline_width: 2.0,
            detail_strength: 0.6,
            speckle: 4,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct FlatStage {
    pub name: String,
    pub elapsed_ms: u128,
}

#[derive(Debug, Clone)]
pub struct FlatOutput {
    pub svg: String,
    pub width: u32,
    pub height: u32,
    pub path_count: usize,
    pub svg_bytes: usize,
    /// Per-stage timings, in execution order (drives the web UI sidebar).
    pub stages: Vec<FlatStage>,
    /// Downscaled PNGs of the garment mask and XDoG linework (None on encode failure).
    pub mask_preview_png: Option<Vec<u8>>,
    pub lines_preview_png: Option<Vec<u8>>,
}

fn stage(stages: &mut Vec<FlatStage>, name: &str, t: Instant) {
    stages.push(FlatStage {
        name: name.into(),
        elapsed_ms: t.elapsed().as_millis(),
    });
}

pub fn convert_flat_bytes(bytes: &[u8], opts: &FlatOptions) -> Result<FlatOutput> {
    if opts.input == FlatInput::OnModel {
        bail!("on-model photos need the Phase-2 ML segmenter (segformer clothes, ONNX) which is not bundled yet — use flat-lay / ghost-mannequin photos for now");
    }
    let t0 = Instant::now();
    let rgb = decode_downscaled(bytes)?;
    if std::env::var("IM2VEC_FLAT_DEBUG").is_ok() {
        eprintln!("decode+resize: {} ms", t0.elapsed().as_millis());
    }
    convert_flat_rgb(&rgb, opts)
}

/// Decode + downscale to [`MAX_SIDE`] (shared by the pipeline and the eval harness).
fn decode_downscaled(bytes: &[u8]) -> Result<RgbImage> {
    let img = image::load_from_memory(bytes).context("decode image (png/jpg/webp/...)")?;
    let mut rgb = img.to_rgb8();
    if rgb.width().max(rgb.height()) > MAX_SIDE {
        let scale = MAX_SIDE as f32 / rgb.width().max(rgb.height()) as f32;
        let (nw, nh) = (
            (rgb.width() as f32 * scale).round() as u32,
            (rgb.height() as f32 * scale).round() as u32,
        );
        // Area-average downscale: the generic Triangle resize costs ~2s
        // at 5MP; this is O(n) with better reduction quality than Nearest.
        rgb = downscale_area(&rgb, nw.max(1), nh.max(1));
    }
    Ok(rgb)
}

/// Backdrop-keyed garment mask + labeled components, the shared first stage
/// of the flat pipeline. Pure function of the input; extracting it changes
/// nothing about pipeline output.
fn mask_and_components(
    rgb: &RgbImage,
    symmetrize: bool,
) -> Result<(Vec<bool>, Vec<u32>, Vec<Component>)> {
    let (w, h) = (rgb.width(), rgb.height());
    let lum = luminance(rgb);
    let mut mask = smooth_mask(&foreground_mask(&lum, w, h), w, h, 2);
    let (labels, mut comps) = label_components(&mask, w as usize, h as usize);
    if symmetrize {
        // Per-view: no cross-view contamination, no re-label needed (the
        // mirror pass never steals pixels from a neighbouring view).
        symmetrize_components(&mut mask, &labels, &mut comps, w as usize, h as usize);
    }
    if !mask.iter().any(|&b| b) {
        bail!("no garment found — flat mode needs a plain, bright backdrop behind the garment");
    }
    Ok((mask, labels, comps))
}

/// Full-resolution garment mask (white = garment) for the eval harness.
/// Runs the same backdrop-keying + symmetrization as [`convert_flat_bytes`];
/// additive measurement API, does not change pipeline output.
pub fn flat_garment_mask(png_bytes: &[u8], opts: &FlatOptions) -> Result<GrayImage> {
    if opts.input == FlatInput::OnModel {
        bail!("on-model photos need the Phase-2 ML segmenter (segformer clothes, ONNX) which is not bundled yet — use flat-lay / ghost-mannequin photos for now");
    }
    let rgb = decode_downscaled(png_bytes)?;
    let (w, h) = (rgb.width(), rgb.height());
    let (mask, _, _) = mask_and_components(&rgb, opts.symmetrize)?;
    GrayImage::from_raw(
        w,
        h,
        mask.iter().map(|&b| if b { 255u8 } else { 0 }).collect(),
    )
    .context("build mask image")
}

fn convert_flat_rgb(rgb: &RgbImage, opts: &FlatOptions) -> Result<FlatOutput> {
    let (w, h) = (rgb.width(), rgb.height());
    let mut stages: Vec<FlatStage> = Vec::new();

    // 1. foreground mask via backdrop keying.
    let t = Instant::now();
    let (mask, _labels, comps) = mask_and_components(rgb, opts.symmetrize)?;
    stage(&mut stages, "background keying", t);

    // 2. silhouette pass: black garment on white.
    let t = Instant::now();
    let sil_img = mask_to_color(&mask, w, h);
    let sil = convert_image(&sil_img, w, h, &trace_opts(opts.speckle, 2.5))?;
    stage(&mut stages, "trace silhouette", t);
    let sil_paths = extract_paths(&sil.svg);
    if sil_paths.is_empty() {
        bail!("silhouette trace produced no paths");
    }
    let ow = opts.outline_width.max(0.5);
    let styled_sil: Vec<String> = sil_paths
        .iter()
        .map(|p| {
            p.replace("fill=\"#000000\"", &format!("fill=\"#ffffff\" stroke=\"#141414\" stroke-width=\"{ow:.1}\" stroke-linejoin=\"round\""))
                .replace("fill=\"rgb(0,0,0)\"", &format!("fill=\"#ffffff\" stroke=\"#141414\" stroke-width=\"{ow:.1}\" stroke-linejoin=\"round\""))
        })
        .collect();

    // 3. detail pass: XDoG response -> skeleton centerlines -> polylines.
    // Chains deep inside the garment survive (drops boundary-hugging echo
    // chains that would double the silhouette); survivors draw as uniform
    // stroked paths.
    let t = Instant::now();
    let region = erode(&mask, w, h, 2);
    let deep = erode(&mask, w, h, 4);
    let mut xd = xdog_lines(rgb, opts.detail_strength);
    let mut lines = xd.full.clone();
    for (i, v) in lines.iter_mut().enumerate() {
        if !region[i] {
            *v = 255;
        }
    }
    if opts.symmetrize_lines {
        // Mirror the linework only inside views judged symmetric above, each
        // around its own axis (asymmetric details stay where they are).
        symmetrize_gray_max_components(&mut lines, &comps, w, h);
        let (fx, fy) = (xd.sw as f32 / w as f32, xd.sh as f32 / h as f32);
        for c in xd.chains.iter_mut() {
            if c.is_empty() {
                continue;
            }
            let (mx, my) = c[c.len() / 2];
            for comp in comps.iter().skip(1).filter(|c| c.symmetrized) {
                let (bx0, bx1) = (comp.x0 as f32 * fx, comp.x1 as f32 * fx);
                let (by0, by1) = (comp.y0 as f32 * fy, comp.y1 as f32 * fy);
                if mx >= bx0 && mx < bx1 && my >= by0 && my < by1 {
                    let cx = (bx0 + bx1 - 1.0) / 2.0;
                    for p in c.iter_mut() {
                        p.0 = 2.0 * cx - p.0;
                    }
                    break;
                }
            }
        }
    }
    let (wu, hu) = (w as usize, h as usize);
    xd.chains.retain(|c| {
        if c.is_empty() {
            return false;
        }
        let inside = c
            .iter()
            .filter(|&&(px, py)| {
                let fx = ((px + 0.5) * w as f32 / xd.sw as f32) as usize;
                let fy = ((py + 0.5) * h as f32 / xd.sh as f32) as usize;
                fx < wu && fy < hu && deep[fy * wu + fx]
            })
            .count();
        inside * 4 >= c.len() * 3
    });
    stage(&mut stages, "XDoG linework", t);

    // 4. compose one flat-style SVG.
    if std::env::var("IM2VEC_FLAT_DEBUG").is_ok() {
        let mask_png = GrayImage::from_raw(
            w,
            h,
            mask.iter().map(|&b| if b { 255u8 } else { 0 }).collect(),
        )
        .expect("mask png");
        let _ = mask_png.save("/tmp/im2vec-flat-mask.png");
        let lines_png = GrayImage::from_raw(w, h, lines.clone()).expect("lines png");
        let _ = lines_png.save("/tmp/im2vec-flat-lines.png");
    }
    let mut svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\" viewBox=\"0 0 {w} {h}\">"
    );
    svg.push_str(&format!(
        "<rect width=\"{w}\" height=\"{h}\" fill=\"#ffffff\"/>"
    ));
    svg.push_str("<g id=\"silhouette\">");
    for p in &styled_sil {
        svg.push_str(p);
    }
    svg.push_str("</g>");
    svg.push_str(&format!(
        "<g id=\"details\" fill=\"none\" stroke=\"#1a1a1a\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\"{}>",
        if opts.stitch_dashed {
            " stroke-dasharray=\"7 4\""
        } else {
            ""
        }
    ));
    let (sxx, syy) = (w as f32 / xd.sw as f32, h as f32 / xd.sh as f32);
    let mut n_det = 0;
    for c in &xd.chains {
        if c.len() < 2 {
            continue;
        }
        n_det += 1;
        svg.push_str(&format!(
            "<path d=\"M{:.1},{:.1}",
            c[0].0 * sxx,
            c[0].1 * syy
        ));
        for &(px, py) in &c[1..] {
            svg.push_str(&format!("L{:.1},{:.1}", px * sxx, py * syy));
        }
        svg.push_str("\"/>");
    }
    svg.push_str("</g></svg>");

    Ok(FlatOutput {
        path_count: styled_sil.len() + n_det,
        svg_bytes: svg.len(),
        svg,
        width: w,
        height: h,
        stages,
        mask_preview_png: png_thumb_mask(&mask, w, h),
        lines_preview_png: png_thumb_gray(&lines, w, h),
    })
}

fn trace_opts(speckle: usize, simplify: f64) -> ConvertOptions {
    ConvertOptions {
        preset: ImPreset::Mono, // forces binary clustering
        mode: "spline".into(),
        hierarchical: "stacked".into(),
        filter_speckle: speckle,
        simplify: Some(simplify),
        path_precision: 2,
        max_colors: None,
        ..Default::default()
    }
}

/// `<path .../>` element slices of a vtracer SVG.
fn extract_paths(svg: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut rest = svg;
    while let Some(s) = rest.find("<path") {
        rest = &rest[s..];
        if let Some(e) = rest.find("/>") {
            out.push(&rest[..e + 2]);
            rest = &rest[e + 2..];
        } else {
            break;
        }
    }
    out
}

fn luminance(rgb: &RgbImage) -> Vec<f32> {
    rgb.pixels()
        .map(|p| (0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32) / 255.0)
        .collect()
}

fn median(mut v: Vec<f32>) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Backdrop color from border pixels; foreground = clearly different.
/// Handles light backdrops (dark garment) and dark backdrops (light garment).
/// Border-connected background is flood-filled so enclosed holes (neck hole,
/// gaps between limbs) stay part of the silhouette; stray exterior specks are
/// dropped by size while every garment view is kept (multi-view
/// front/side/back inputs survive intact).
fn foreground_mask(lum: &[f32], w: u32, h: u32) -> Vec<bool> {
    let (w, h) = (w as usize, h as usize);
    let mut border = Vec::with_capacity(2 * (w + h));
    for x in 0..w {
        border.push(lum[x]);
        border.push(lum[(h - 1) * w + x]);
    }
    for y in 0..h {
        border.push(lum[y * w]);
        border.push(lum[y * w + w - 1]);
    }
    let bg = median(border);
    // Foreground differs from the backdrop by a margin either way.
    let mut fg: Vec<bool> = lum.iter().map(|&v| (v - bg).abs() > 0.08).collect();

    // Flood-fill background from the borders through non-foreground pixels.
    let mut is_bg = vec![false; w * h];
    let mut stack = Vec::new();
    for x in 0..w {
        stack.push(x);
        stack.push((h - 1) * w + x);
    }
    for y in 0..h {
        stack.push(y * w);
        stack.push(y * w + w - 1);
    }
    while let Some(i) = stack.pop() {
        if is_bg[i] || fg[i] {
            continue;
        }
        is_bg[i] = true;
        let (x, y) = (i % w, i / w);
        if x > 0 {
            stack.push(i - 1);
        }
        if x + 1 < w {
            stack.push(i + 1);
        }
        if y > 0 {
            stack.push(i - w);
        }
        if y + 1 < h {
            stack.push(i + w);
        }
    }
    // Anything not reached is foreground (garment + enclosed holes).
    for (i, f) in fg.iter_mut().enumerate() {
        *f = *f || !is_bg[i];
    }

    // Size floor scales with resolution: specks vanish, every garment view
    // (even a narrow side view at ~5% of the frame) survives.
    let min_area = ((w * h) / 2000).max(64);
    sweep_small_components(&mut fg, w, h, min_area);
    fg
}

/// One 4-connected foreground blob: pixel count plus bounding box
/// (x1/y1 exclusive).
#[derive(Debug, Clone)]
struct Component {
    area: usize,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    /// Set once per-view symmetrization judged this view near-symmetric.
    symmetrized: bool,
}

/// Label every 4-connected foreground component. Returns the per-pixel
/// label image (0 = background, 1-based component ids) plus one
/// [`Component`] per id (index 0 unused).
fn label_components(fg: &[bool], w: usize, h: usize) -> (Vec<u32>, Vec<Component>) {
    let mut labels = vec![0u32; w * h];
    let mut comps: Vec<Component> = vec![Component {
        area: 0,
        x0: 0,
        y0: 0,
        x1: 0,
        y1: 0,
        symmetrized: false,
    }]; // 1-based
    let mut next = 0u32;
    for i in 0..w * h {
        if !fg[i] || labels[i] != 0 {
            continue;
        }
        next += 1;
        let mut stack = vec![i];
        let (mut area, mut x0, mut y0, mut x1, mut y1) = (0usize, w, h, 0usize, 0usize);
        while let Some(j) = stack.pop() {
            if !fg[j] || labels[j] != 0 {
                continue;
            }
            labels[j] = next;
            let (x, y) = (j % w, j / w);
            area += 1;
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x + 1);
            y1 = y1.max(y + 1);
            if x > 0 {
                stack.push(j - 1);
            }
            if x + 1 < w {
                stack.push(j + 1);
            }
            if y > 0 {
                stack.push(j - w);
            }
            if y + 1 < h {
                stack.push(j + w);
            }
        }
        comps.push(Component {
            area,
            x0,
            y0,
            x1,
            y1,
            symmetrized: false,
        });
    }
    (labels, comps)
}

/// Zero connected components smaller than `min_area` px (area opening on the
/// mask). Unlike keep-largest, every garment view survives multi-view inputs.
fn sweep_small_components(fg: &mut [bool], w: usize, h: usize, min_area: usize) {
    let (labels, comps) = label_components(fg, w, h);
    for (i, f) in fg.iter_mut().enumerate() {
        if *f && comps[labels[i] as usize].area < min_area {
            *f = false;
        }
    }
}

/// Mirror-average each garment view around its own vertical center axis.
/// Views already near-symmetric (front/back) get the wrinkle-averaging the
/// old global pass provided; asymmetric views (side/profile) pass through
/// untouched, since mirroring them would fabricate a phantom silhouette.
/// `labels` is the label image for the same mask revision; the mirror union
/// never steals pixels from a neighbouring view, so no re-label is needed.
/// Per-view success lands in `comps[i].symmetrized` for the linework pass.
fn symmetrize_components(
    mask: &mut [bool],
    labels: &[u32],
    comps: &mut [Component],
    w: usize,
    h: usize,
) {
    // Max relative area growth admitted from the mirror union: symmetric
    // fronts/backs stay far below this, side views grow far above it.
    const BASE_MAX_GROWTH: f32 = 0.15;
    for (id, comp) in comps.iter_mut().enumerate().skip(1) {
        let label = id as u32;
        let (x0, x1, y0, y1) = (comp.x0, comp.x1, comp.y0, comp.y1);
        if x1 <= x0 + 1 || y1 <= y0 || x1 > w || y1 > h {
            continue;
        }
        // Distance from component centre to the nearest image boundary.
        // Components near the bottom hem (flaps, pockets) get reduced mirror
        // growth to avoid stealing neighbour pixels and breaking flap edges.
        let comp_center_y = (y0 + y1) as f32 / 2.0;
        let dist_to_bottom = h as f32 - comp_center_y;
        let dist_to_top = comp_center_y;
        let hem_buffer = dist_to_bottom.min(dist_to_top);
        // If the component is within 30% of the image height from either edge,
        // reduce the mirror growth proportionally.
        let rel_growth = if hem_buffer / (h as f32) < 0.30 {
            // Growth scales from 5% at the edge to BASE_MAX_GROWTH at the 30% threshold.
            let t = hem_buffer / (h as f32); // 0.0 at edge, 0.3 at threshold

            0.05 + (BASE_MAX_GROWTH - 0.05) * (t / 0.30)
        } else {
            BASE_MAX_GROWTH
        };
        // Dry run: count the union growth without writing. Mirror of (x,y)
        // is (x0+x1-1-x, y). Pixels owned by a neighbouring view are skipped.
        let mut added = 0usize;
        for y in y0..y1 {
            for x in x0..x1 {
                let i = y * w + x;
                if mask[i] || (labels[i] != 0 && labels[i] != label) {
                    continue;
                }
                if mask[y * w + (x0 + x1 - 1 - x)] {
                    added += 1;
                }
            }
        }
        if added as f32 > comp.area as f32 * rel_growth {
            continue; // asymmetric view (e.g. side/profile near hem): leave alone
        }
        for y in y0..y1 {
            for x in x0..x1 {
                let i = y * w + x;
                if mask[i] || (labels[i] != 0 && labels[i] != label) {
                    continue;
                }
                if mask[y * w + (x0 + x1 - 1 - x)] {
                    mask[i] = true;
                }
            }
        }
        comp.symmetrized = true;
    }
}

/// Mirror-max inside each symmetrized view's own bounding box: each mirrored
/// pair takes the darker (stronger-line) value. Asymmetric views are skipped.
fn symmetrize_gray_max_components(g: &mut [u8], comps: &[Component], w: u32, h: u32) {
    let (w, h) = (w as usize, h as usize);
    for comp in comps.iter().skip(1).filter(|c| c.symmetrized) {
        let (x0, x1, y0, y1) = (
            comp.x0.min(w),
            comp.x1.min(w),
            comp.y0.min(h),
            comp.y1.min(h),
        );
        if x1 <= x0 + 1 || y1 <= y0 {
            continue;
        }
        for y in y0..y1 {
            for x in x0..(x0 + x1) / 2 {
                let (a, b) = (y * w + x, y * w + (x0 + x1 - 1 - x));
                let v = g[a].min(g[b]);
                g[a] = v;
                g[b] = v;
            }
        }
    }
}

/// Area-average downscale for large inputs.
fn downscale_area(rgb: &RgbImage, nw: u32, nh: u32) -> RgbImage {
    use image::Rgb;
    let (sw, sh) = (rgb.width() as usize, rgb.height() as usize);
    let (nw, nh) = (nw as usize, nh as usize);
    let src = rgb.as_raw();
    let mut out = RgbImage::new(nw as u32, nh as u32);
    for y in 0..nh {
        let y0 = y * sh / nh;
        let y1 = ((y + 1) * sh / nh).max(y0 + 1);
        for x in 0..nw {
            let x0 = x * sw / nw;
            let x1 = ((x + 1) * sw / nw).max(x0 + 1);
            let (mut r, mut g, mut b, mut n) = (0u32, 0u32, 0u32, 0u32);
            for sy in y0..y1 {
                for sx in x0..x1 {
                    let i = (sy * sw + sx) * 3;
                    r += src[i] as u32;
                    g += src[i + 1] as u32;
                    b += src[i + 2] as u32;
                    n += 1;
                }
            }
            out.put_pixel(
                x as u32,
                y as u32,
                Rgb([(r / n) as u8, (g / n) as u8, (b / n) as u8]),
            );
        }
    }
    out
}

fn dilate(mask: &[bool], w: u32, h: u32, r: usize) -> Vec<bool> {
    let (w, h) = (w as usize, h as usize);
    let mut out = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            if !mask[y * w + x] {
                continue;
            }
            let y0 = y.saturating_sub(r);
            let y1 = (y + r + 1).min(h);
            let x0 = x.saturating_sub(r);
            let x1 = (x + r + 1).min(w);
            for yy in y0..y1 {
                for xx in x0..x1 {
                    out[yy * w + xx] = true;
                }
            }
        }
    }
    out
}

/// Open then close: removes boundary notches and bumps so the traced
/// silhouette becomes a few clean curves instead of jitter.
fn smooth_mask(mask: &[bool], w: u32, h: u32, r: usize) -> Vec<bool> {
    let opened = dilate(&erode(mask, w, h, r), w, h, r);
    erode(&dilate(&opened, w, h, r), w, h, r)
}

fn erode(mask: &[bool], w: u32, h: u32, r: usize) -> Vec<bool> {
    let (w, h) = (w as usize, h as usize);
    let mut out = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let y0 = y.saturating_sub(r);
            let y1 = (y + r + 1).min(h);
            let x0 = x.saturating_sub(r);
            let x1 = (x + r + 1).min(w);
            let mut all = true;
            for yy in y0..y1 {
                for xx in x0..x1 {
                    if !mask[yy * w + xx] {
                        all = false;
                        break;
                    }
                }
                if !all {
                    break;
                }
            }
            out[y * w + x] = all;
        }
    }
    out
}

/// Gaussian blur, pyramidal for large sigmas: blur at half resolution and
/// upscale. ~4x faster per level with negligible effect on XDoG output.
fn fast_blur(gray: &GrayImage, sigma: f32) -> GrayImage {
    if sigma <= 2.0 {
        return gauss_blur(gray, sigma);
    }
    let (w, h) = (gray.width().max(2), gray.height().max(2));
    let small = image::imageops::resize(gray, w / 2, h / 2, image::imageops::FilterType::Nearest);
    let blurred = gauss_blur(&small, sigma / 2.0);
    image::imageops::resize(&blurred, w, h, image::imageops::FilterType::Nearest)
}

/// Separable Gaussian blur with truncated kernel. (The generic ops blur was
/// the pipeline bottleneck at ~1.6s per conversion; this is milliseconds.)
fn gauss_blur(gray: &GrayImage, sigma: f32) -> GrayImage {
    let (w, h) = (gray.width() as usize, gray.height() as usize);
    let r = ((sigma * 3.0).ceil() as usize).max(1);
    let mut k = Vec::with_capacity(2 * r + 1);
    let mut sum = 0.0f32;
    for i in -(r as i32)..=r as i32 {
        let v = (-(i * i) as f32 / (2.0 * sigma * sigma)).exp();
        k.push(v);
        sum += v;
    }
    for v in k.iter_mut() {
        *v /= sum;
    }
    let src = gray.as_raw();
    let mut tmp = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (j, &kv) in k.iter().enumerate() {
                let xx = (x as i32 + j as i32 - r as i32).clamp(0, w as i32 - 1) as usize;
                acc += src[y * w + xx] as f32 * kv;
            }
            tmp[y * w + x] = acc.round().clamp(0.0, 255.0) as u8;
        }
    }
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (j, &kv) in k.iter().enumerate() {
                let yy = (y as i32 + j as i32 - r as i32).clamp(0, h as i32 - 1) as usize;
                acc += tmp[yy * w + x] as f32 * kv;
            }
            out[y * w + x] = acc.round().clamp(0.0, 255.0) as u8;
        }
    }
    GrayImage::from_raw(w as u32, h as u32, out).expect("blur size")
}

/// Extended Difference-of-Gaussians: dark stylized lines on white.
/// `strength` 0..=1 maps to the epsilon threshold (lower eps = more lines).
///
/// Color-aware: the DoG valley response is computed per R/G/B channel and
/// OR-ed (a line registers when ANY channel dips). Luminance-only XDoG is
/// blind to near-isoluminant edges — gold buttons on blue denim, white
/// contrast stitching — which all move chroma far more than luminance.
/// XDoG output: full-res binary for previews plus vector chains in small-px.
struct XdogOut {
    full: Vec<u8>,
    chains: Vec<Vec<(f32, f32)>>,
    sw: u32,
    sh: u32,
}

fn xdog_lines(rgb: &RgbImage, strength: f32) -> XdogOut {
    let (w, h) = (rgb.width(), rgb.height());
    let (wu, hu) = (w as usize, h as usize);
    // Compute the response at reduced resolution (edges survive downscaling)
    // with hand-rolled sampling: the generic ops resize is ~1s at 2MP.
    let sigma = 1.4f32;
    let scale = (800.0 / w.max(h) as f32).min(1.0);
    let (sw, sh) = (
        ((w as f32 * scale).round() as u32).max(1),
        ((h as f32 * scale).round() as u32).max(1),
    );
    let tau = 0.98f32;
    let phi = 20.0f32;
    // eps window is tight: interior DoG floor sits near +0.01, so eps must
    // stay negative; -0.15 keeps only the strongest edges, -0.008 everything.
    let eps = (-0.15 + 0.20 * strength.clamp(0.0, 1.0)).min(-0.008);
    let (swu, shu) = (sw as usize, sh as usize);
    // Per-channel valleys: each channel is blurred at both sigmas, the
    // soft-threshold response computed per channel, and the darkest (min)
    // response wins each pixel. 6 small blurs instead of 2; fast_blur keeps
    // the whole response stage in the tens of ms.
    let mut small_out = vec![255u8; swu * shu];
    for ch in 0..3 {
        let small = sample_channel(rgb, wu, hu, sw, sh, ch);
        let b1 = fast_blur(&small, sigma * scale);
        let b2 = fast_blur(&small, sigma * 4.0 * scale);
        let (rb1, rb2) = (b1.as_raw(), b2.as_raw());
        for (i, o) in small_out.iter_mut().enumerate() {
            let d = rb1[i] as f32 / 255.0 - tau * rb2[i] as f32 / 255.0;
            // XDoG soft threshold: values below eps become dark lines.
            let v = if d >= eps {
                1.0
            } else {
                1.0 + (phi * (d - eps)).tanh()
            };
            let v = (v.clamp(0.0, 1.0) * 255.0) as u8;
            if v < *o {
                *o = v;
            }
        }
    }
    // Hysteresis: confident lines seed, faint lines survive only when
    // connected to confident ones. Joins dotted seams, drops lone noise.
    // Then an area opening removes remaining tiny specks.
    let high = 210u8;
    let low = (225.0 + 10.0 * strength.clamp(0.0, 1.0)) as u8;
    let min_size = (30.0 - 20.0 * strength.clamp(0.0, 1.0)) as usize;
    let mut kept = hysteresis(&small_out, swu, shu, high, low);
    // [T1 speckle filter] removed - using sweep_small_components min_area threshold instead
    sweep_small(&mut kept, swu, shu, min_size);
    // Small filled details (buttons, eyelets, labels) thin to a dot and get
    // swept, so they can never survive the skeleton path. Emit their outer
    // boundary loops as closed chains instead.
    let blobs = blob_outlines(&kept, swu, shu);
    sweep_small(&mut kept, swu, shu, min_size);
    // Close before thinning: joins dotted seams and 1-2px gaps into continuous
    // strokes (the old filled-blob pass got this from close_u8 r1). Without
    // it the skeleton shatters into swept-away fragments.
    let kept = erode(&dilate(&kept, sw, sh, 1), sw, sh, 1);
    let mut bin = vec![255u8; swu * shu];
    for (i, &k) in kept.iter().enumerate() {
        if k {
            bin[i] = 0;
        }
    }
    let mut skel = zhang_suen(&bin, swu, shu);
    // Minimal pre-filtering: kill only pixel dots here. Dotted seams must
    // reach the graph-level merge intact (it joins by chord fit); size
    // filtering happens on final arc length instead.
    sweep_small(&mut skel, swu, shu, 4);
    // Spur pruning: delete 1-2px nubs that only add junction splits.
    prune_spurs(&mut skel, swu, shu, 3);
    let mut chains = trace_chains(&skel, swu, shu);
    // Rejoin dotted-seam fragments split at junctions / small gaps.
    merge_collinear(&mut chains, 6.0);
    // Closed blob loops never merge (their chord fit is huge); append after.
    chains.extend(blobs);
    let chains = chains
        .into_iter()
        .filter(|c| c.len() >= 2)
        .map(|c| simplify_dp(&c, 1.0))
        .filter(|c| arc_len(c) >= 3.0)
        .collect::<Vec<_>>();
    // Full-res binary for the sidebar preview + debug dumps.
    let full = {
        let mut b = vec![255u8; swu * shu];
        for (i, &k) in skel.iter().enumerate() {
            if k {
                b[i] = 0;
            }
        }
        if sw == w && sh == h {
            b
        } else {
            upscale_nearest_u8(&b, sw, sh, w, h)
        }
    };
    XdogOut {
        full,
        chains,
        sw,
        sh,
    }
}

/// Spur pruning: delete short branches rooted at a junction (staircase
/// artifacts on diagonals, ragged-edge twigs). Isolated dashes have an
/// endpoint at both ends, so they survive. Repeat passes; each removal can
/// reveal a new short spur. The junction root pixel itself is kept.
fn prune_spurs(skel: &mut [bool], w: usize, h: usize, max_len: usize) {
    const DX8: [i32; 8] = [1, 1, 0, -1, -1, -1, 0, 1];
    const DY8: [i32; 8] = [0, 1, 1, 1, 0, -1, -1, -1];
    let at = |x: i32, y: i32, s: &[bool]| -> bool {
        x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h && s[y as usize * w + x as usize]
    };
    let ncount = |x: i32, y: i32, s: &[bool]| -> usize {
        (0..8).filter(|&k| at(x + DX8[k], y + DY8[k], s)).count()
    };
    for _ in 0..3 {
        let mut changed = false;
        for y in 0..h {
            for x in 0..w {
                if !skel[y * w + x] || ncount(x as i32, y as i32, skel) != 1 {
                    continue;
                }
                let mut branch = vec![(x as i32, y as i32)];
                let (mut cx, mut cy) = (x as i32, y as i32);
                let (mut px, mut py) = (-1, -1);
                // Incoming step direction; prefer the straightest neighbor so
                // the walk doesn't shortcut diagonally onto crossing lines.
                let (mut dx, mut dy) = (0i32, 0i32);
                loop {
                    let mut nxt: Option<(i32, i32)> = None;
                    let mut best_dot = i32::MIN;
                    for k in 0..8 {
                        let (nx, ny) = (cx + DX8[k], cy + DY8[k]);
                        if (nx != px || ny != py) && at(nx, ny, skel) {
                            let dot = DX8[k] * dx + DY8[k] * dy;
                            if nxt.is_none() || dot > best_dot {
                                best_dot = dot;
                                nxt = Some((nx, ny));
                            }
                        }
                    }
                    match nxt {
                        None => break,
                        Some((nx, ny)) => {
                            (dx, dy) = (nx - cx, ny - cy);
                            (px, py) = (cx, cy);
                            (cx, cy) = (nx, ny);
                            branch.push((cx, cy));
                            if ncount(cx, cy, skel) != 2 {
                                break;
                            }
                        }
                    }
                }
                // Twig = short branch ending at a junction. Pop the root so
                // the junction (shared with other branches) survives.
                if branch.len() <= max_len + 1 && ncount(cx, cy, skel) >= 3 {
                    branch.pop();
                    for &(bx, by) in &branch {
                        skel[by as usize * w + bx as usize] = false;
                    }
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
}

/// Walk the skeleton into polyline chains (small-px float coords).
/// Splits at junctions (round caps rejoin them visually); loops close.
fn trace_chains(skel: &[bool], w: usize, h: usize) -> Vec<Vec<(f32, f32)>> {
    const DX8: [i32; 8] = [1, 1, 0, -1, -1, -1, 0, 1];
    const DY8: [i32; 8] = [0, 1, 1, 1, 0, -1, -1, -1];
    let at = |x: i32, y: i32| -> bool {
        x >= 0
            && y >= 0
            && (x as usize) < w
            && (y as usize) < h
            && skel[y as usize * w + x as usize]
    };
    let ncount =
        |x: i32, y: i32| -> usize { (0..8).filter(|&k| at(x + DX8[k], y + DY8[k])).count() };
    // Walk from (sx,sy) through unvisited pixels. Junctions/endpoints stop
    // the walk (already pushed); the start pixel always takes one step.
    let walk = |sx: i32, sy: i32, visited: &mut [bool]| -> Vec<(f32, f32)> {
        let mut path = vec![(sx as f32, sy as f32)];
        let (mut cx, mut cy) = (sx, sy);
        loop {
            if ncount(cx, cy) != 2 && (cx != sx || cy != sy) {
                break;
            }
            let mut nxt = None;
            for k in 0..8 {
                let (nx, ny) = (cx + DX8[k], cy + DY8[k]);
                if at(nx, ny) && !visited[ny as usize * w + nx as usize] {
                    nxt = Some((nx, ny));
                    break;
                }
            }
            let (nx, ny) = match nxt {
                Some(p) => p,
                None => break,
            };
            path.push((nx as f32, ny as f32));
            visited[ny as usize * w + nx as usize] = true;
            (cx, cy) = (nx, ny);
        }
        path
    };
    let mut visited = vec![false; w * h];
    let mut chains: Vec<Vec<(f32, f32)>> = Vec::new();
    // 1. endpoint walks.
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if skel[i] && !visited[i] && ncount(x as i32, y as i32) == 1 {
                visited[i] = true;
                chains.push(walk(x as i32, y as i32, &mut visited));
            }
        }
    }
    // 2. junction fans: one chain per unvisited direction, rooted at J.
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if !skel[i] || visited[i] || ncount(x as i32, y as i32) < 3 {
                continue;
            }
            visited[i] = true;
            for k in 0..8 {
                let (nx, ny) = (x as i32 + DX8[k], y as i32 + DY8[k]);
                if at(nx, ny) && !visited[ny as usize * w + nx as usize] {
                    // Mark the walk start (walk() only marks stepped pixels).
                    visited[ny as usize * w + nx as usize] = true;
                    let mut c = vec![(x as f32, y as f32)];
                    c.extend(walk(nx, ny, &mut visited));
                    chains.push(c);
                }
            }
        }
    }
    // 3. leftover loops: walk until closed.
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if skel[i] && !visited[i] {
                visited[i] = true;
                let mut c = walk(x as i32, y as i32, &mut visited);
                if c.len() > 2 {
                    let (fx, fy) = c[0];
                    let (lx, ly) = c[c.len() - 1];
                    if (fx - lx).abs() <= 1.5 && (fy - ly).abs() <= 1.5 {
                        c.push((fx, fy));
                    }
                }
                chains.push(c);
            }
        }
    }
    // Dots (isolated pixels) carry no linework; drop them at the source.
    chains.into_iter().filter(|c| c.len() >= 2).collect()
}

/// Greedily merge chain ends that nearly touch and continue (near-)straight:
/// thinning splits lines at every junction pixel, and staircase diagonals
/// shatter into short fragments that turn sharply. Primary test: the
/// concatenation must fit its end-to-end chord (staircase fits a line; a
/// collar V does not). Curved lapel/seam fragments fail that test, so a
/// secondary test accepts tangent-aligned joins (end directions agree within
/// ~32 deg) with a looser chord bound. Sharp corners stay split and rejoin
/// visually via round caps.
fn merge_collinear(chains: &mut Vec<Vec<(f32, f32)>>, gap: f32) {
    /// Max perpendicular deviation of points from the end-to-end chord.
    fn chord_dev(pts: &[(f32, f32)]) -> f32 {
        if pts.len() <= 2 {
            return 0.0;
        }
        let (ax, ay) = pts[0];
        let (bx, by) = pts[pts.len() - 1];
        let (dx, dy) = (bx - ax, by - ay);
        let den = (dx * dx + dy * dy).sqrt().max(1e-6);
        pts.iter()
            .map(|&(px, py)| (dy * px - dx * py + bx * ay - by * ax).abs() / den)
            .fold(0.0, f32::max)
    }
    const FIT_TOL: f32 = 1.5;
    /// Looser chord bound for tangent-aligned joins (curved seams/lapels).
    const CURVE_TOL: f32 = 3.0;
    /// Long jumps (up to 3.5x gap) for gapped fold/stitch dashes: only when
    /// nearly exactly collinear (dot > 0.92, dev <= 2.0). Greedy min-dev
    /// selection still prefers true continuations over parallel-line rivals.
    const LONG_REACH_MULT: f32 = 3.5;
    const LONG_DOT: f32 = 0.92;
    const LONG_DEV: f32 = 2.0;
    /// Min cosine between the end directions at the join (~32 deg).
    const TANGENT_DOT: f32 = 0.85;
    /// Unit direction of the last/first `span` points at a chain end, in
    /// oriented coords where the join sits at the tail / head respectively.
    fn end_dir(pts: &[(f32, f32)], tail: bool) -> (f32, f32) {
        let n = pts.len();
        let k = (n - 1).clamp(1, 3);
        let (ax, ay, bx, by) = if tail {
            let (ax, ay) = pts[n - 1 - k];
            let (bx, by) = pts[n - 1];
            (ax, ay, bx, by)
        } else {
            let (ax, ay) = pts[0];
            let (bx, by) = pts[k];
            (ax, ay, bx, by)
        };
        let (dx, dy) = (bx - ax, by - ay);
        let len = (dx * dx + dy * dy).sqrt().max(1e-6);
        (dx / len, dy / len)
    }
    loop {
        // (i, ie, j, je, dev), minimizing dev among ends within gap.
        let mut best: Option<(usize, bool, usize, bool, f32)> = None;
        for i in 0..chains.len() {
            if chains[i].len() < 2 {
                continue;
            }
            for j in (i + 1)..chains.len() {
                if chains[j].len() < 2 {
                    continue;
                }
                for &ie in &[false, true] {
                    for &je in &[false, true] {
                        let pi = if ie {
                            *chains[i].last().unwrap()
                        } else {
                            chains[i][0]
                        };
                        let pj = if je {
                            *chains[j].last().unwrap()
                        } else {
                            chains[j][0]
                        };
                        let dist = ((pi.0 - pj.0).powi(2) + (pi.1 - pj.1).powi(2)).sqrt();
                        if dist > gap * LONG_REACH_MULT {
                            continue;
                        }
                        let dist = ((pi.0 - pj.0).powi(2) + (pi.1 - pj.1).powi(2)).sqrt();
                        if dist > gap * LONG_REACH_MULT {
                            continue;
                        }
                        let skip = usize::from(dist < 0.75);
                        // Tangent fallback directions, measured at the
                        // junction on the oriented halves: i arrives along
                        // ti, j leaves along tj.
                        let mut cand: Vec<(f32, f32)> = if ie {
                            chains[i].clone()
                        } else {
                            chains[i].iter().rev().cloned().collect()
                        };
                        let other: Vec<(f32, f32)> = if je {
                            chains[j].iter().rev().cloned().collect()
                        } else {
                            chains[j].clone()
                        };
                        let (tix, tiy) = end_dir(&cand, true);
                        let (tjx, tjy) = end_dir(&other, false);
                        // Skip short chain merges that would create diagonal
                        // artifacts: if both chains are short (< 6px) and their
                        // end directions are somewhat perpendicular (dot < 0.5),
                        // they likely represent unrelated fragments.
                        let i_short = chains[i].len() < 6;
                        let j_short = chains[j].len() < 6;
                        let dirs_perp = tix * tjx + tiy * tjy < 0.5;
                        if i_short && j_short && dirs_perp {
                            continue;
                        }
                        cand.extend(other.into_iter().skip(skip));
                        let dev = chord_dev(&cand);
                        // Aligned ends continue one curve: accept under a
                        // looser chord bound for lapel/seam arcs, with a
                        // longer reach for stitch pitch. The plain chord
                        // test keeps the original tight gap.
                        let aligned = tix * tjx + tiy * tjy > TANGENT_DOT && dev <= CURVE_TOL;
                        // Seam-aware join: accept even when tangent directions
                        // differ by more than 30° (dot < 0.866), as hem stitches
                        // often cross princess seams with direction changes. If the
                        // chord deviation is small and the gap is reasonable, join
                        // them as a seam junction.
                        let seam_ok = dev <= 2.0 && dist <= 8.0 && tix * tjx + tiy * tjy > 0.6;
                        let chord_ok = dist <= gap && dev <= FIT_TOL;
                        // Long jump across a response gap: near-exact
                        // collinearity only; min-dev ordering still serves
                        // true continuations first.
                        let long_ok = tix * tjx + tiy * tjy > LONG_DOT && dev <= LONG_DEV;
                        if (chord_ok || aligned || seam_ok || long_ok)
                            && best.map(|b| dev < b.4).unwrap_or(true)
                        {
                            best = Some((i, ie, j, je, dev));
                        }
                    }
                }
            }
        }
        let (i, ie, j, je, _) = match best {
            Some(b) => b,
            None => break,
        };
        // Rebuild: orient i so the join is its tail, j so the join is its
        // head, then concatenate (skipping a duplicated junction pixel).
        if !ie {
            chains[i].reverse();
        }
        if je {
            chains[j].reverse();
        }
        let mut merged = std::mem::take(&mut chains[i]);
        let other = std::mem::take(&mut chains[j]);
        let (lx, ly) = *merged.last().unwrap();
        let skip = ((other[0].0 - lx).powi(2) + (other[0].1 - ly).powi(2)).sqrt() < 0.75;
        merged.extend(other.into_iter().skip(usize::from(skip)));
        chains[i] = merged;
        chains.remove(j);
    }
    chains.retain(|c| c.len() >= 2);
}

/// Outer boundary loops of small compact response components, in small-px
/// coords (closed: first point repeated at the end). Thin seam fragments are
/// excluded by the min-side gate; large regions by the max-side gate; wisps
/// by the solidity gate. Units are small-px so the gates behave the same at
/// any input resolution.
fn blob_outlines(kept: &[bool], w: usize, h: usize) -> Vec<Vec<(f32, f32)>> {
    let (labels, comps) = label_components(kept, w, h);
    let mut out = Vec::new();
    for (id, c) in comps.iter().enumerate().skip(1) {
        let (bw, bh) = (c.x1 - c.x0, c.y1 - c.y0);
        let bbox_area = (bw * bh).max(1);
        if c.area < 24 || bw.min(bh) < 6 || bw.max(bh) > 64 {
            continue;
        }
        if c.area * 100 < bbox_area * 45 {
            continue;
        }
        let boundary = moore_outline(&labels, id as u32, c, w, h);
        if boundary.len() >= 4 {
            // Buttons trace as ragged loops; a clean fitted circle reads as
            // a tech-pack button, while non-circular blobs keep their shape.
            let button_trace = fit_circle_or(boundary);
            // If the fitted circle is actually D-shaped (circularity < 0.70),
            // replace with an ellipse that preserves the button's shape.
            if button_trace.len() >= 4 {
                let circ = circularity(&button_trace);
                if circ < 0.70 && is_near_circular_area(c.area, (bw, bh)) {
                    // Fit an 8-point ellipse and replace the trace
                    out.push(ellipse_trace(&button_trace));
                } else {
                    out.push(button_trace);
                }
            }
        }
    }
    out
}

/// Kasa circle fit on a closed boundary loop: when the loop is circular
/// (radial RMSE < 12% of radius, near-square bbox) replace the ragged trace
/// with a clean 14-gon; otherwise return the loop unchanged.
fn fit_circle_or(loop_pts: Vec<(f32, f32)>) -> Vec<(f32, f32)> {
    // Drop the closing duplicate for the fit.
    let pts = if loop_pts.len() > 2 {
        &loop_pts[..loop_pts.len() - 1]
    } else {
        return loop_pts;
    };
    let n = pts.len() as f32;
    let (mx, my) = (
        pts.iter().map(|p| p.0).sum::<f32>() / n,
        pts.iter().map(|p| p.1).sum::<f32>() / n,
    );
    // Kasa fit: solve [suu suv; suv svv] [uc;vc] = rhs/2 for the center.
    let (mut suu, mut suv, mut svv) = (0.0f32, 0.0f32, 0.0f32);
    let (mut bx, mut by) = (0.0f32, 0.0f32);
    for &(px, py) in pts {
        let (u, v) = (px - mx, py - my);
        suu += u * u;
        suv += u * v;
        svv += v * v;
        let r2 = u * u + v * v;
        bx += u * r2;
        by += v * r2;
    }
    let det = suu * svv - suv * suv;
    if det.abs() < 1e-6 {
        return loop_pts;
    }
    let (uc, vc) = (
        (bx * svv - by * suv) / det / 2.0,
        (suu * by - suv * bx) / det / 2.0,
    );
    let (cx, cy) = (uc + mx, vc + my);
    let r = pts
        .iter()
        .map(|&(px, py)| ((px - cx).powi(2) + (py - cy).powi(2)).sqrt())
        .sum::<f32>()
        / n;
    if r < 2.0 {
        return loop_pts;
    }
    let rmse = (pts
        .iter()
        .map(|&(px, py)| (((px - cx).powi(2) + (py - cy).powi(2)).sqrt() - r).powi(2))
        .sum::<f32>()
        / n)
        .sqrt();
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for &(px, py) in pts {
        x0 = x0.min(px);
        y0 = y0.min(py);
        x1 = x1.max(px);
        y1 = y1.max(py);
    }
    let (bw, bh) = (x1 - x0 + 1.0, y1 - y0 + 1.0);
    if rmse / r > 0.12 || (bw / bh - 1.0).abs() > 0.35 {
        return loop_pts;
    }
    // Clean 14-gon, closed.
    let mut poly: Vec<(f32, f32)> = (0..14)
        .map(|k| {
            let a = k as f32 * std::f32::consts::TAU / 14.0;
            (cx + r * a.cos(), cy + r * a.sin())
        })
        .collect();
    poly.push((cx + r, cy));
    poly
}

/// Circle circularity = 4π·area / perimeter². Values near 1.0 are circular,
/// values near 0.0 are line-like, < 0.70 indicates D-shaped or worse.
fn circularity(pts: &[(f32, f32)]) -> f32 {
    // Compute polygon area via shoelace formula
    let mut area2 = 0.0f32;
    let n = pts.len();
    for i in 0..n {
        let j = (i + 1) % n;
        area2 += pts[i].0 * pts[j].1 - pts[j].0 * pts[i].1;
    }
    let area = area2.abs() / 2.0f32;
    // Compute perimeter
    let mut perim = 0.0f32;
    for i in 0..n {
        let j = (i + 1) % n;
        let dx = pts[j].0 - pts[i].0;
        let dy = pts[j].1 - pts[i].1;
        perim += (dx * dx + dy * dy).sqrt();
    }
    if perim < 1e-6 || area < 1e-6 {
        return 0.0;
    }
    4.0 * std::f32::consts::PI * area / (perim * perim)
}

/// Return true if the component's area is within ±20% of a circle with the
/// same bounding-box area (i.e. the component is not a radically different
/// shape that would make ellipse fitting meaningless).
fn is_near_circular_area(area: usize, bbox: (usize, usize)) -> bool {
    let (bw, bh) = bbox;
    let circle_area = (bw * bh) as f32 * 0.785; // π/4 ≈ 0.785, area of circle inscribed in bbox
    let ratio = area as f32 / circle_area;
    (0.8..=1.2).contains(&ratio)
}

/// Fit an 8-point ellipse to a boundary loop and return a closed trace.
/// The ellipse is centered at the loop's centroid with semi-axes computed
/// from the second-moment matrix; returns the original loop if fitting fails.
fn ellipse_trace(loop_pts: &[(f32, f32)]) -> Vec<(f32, f32)> {
    let n = loop_pts.len();
    if n < 4 {
        return loop_pts.to_vec();
    }
    // Use all but the last point (assumed duplicate of first)
    let pts = if n > 1 { &loop_pts[..n - 1] } else { loop_pts };
    let m = pts.len();
    let (mx, my) = (
        pts.iter().map(|p| p.0).sum::<f32>() / m as f32,
        pts.iter().map(|p| p.1).sum::<f32>() / m as f32,
    );
    let sxx = pts.iter().map(|&(x, _)| (x - mx) * (x - mx)).sum::<f32>() / m as f32;
    let syy = pts.iter().map(|&(_, y)| (y - my) * (y - my)).sum::<f32>() / m as f32;
    let sxy = pts.iter().map(|&(x, y)| (x - mx) * (y - my)).sum::<f32>() / m as f32;
    let t = (sxx - syy) / 2.0;
    let d = ((sxx - syy) / 2.0).powi(2) + sxy * sxy;
    let lambda1 = (t + d.sqrt()) / 2.0;
    let lambda2 = (t - d.sqrt()) / 2.0;
    let a = lambda1.sqrt().max(1.0); // semi-major axis
    let b = lambda2.sqrt().max(1.0); // semi-minor axis
    let angle = if sxy.abs() < 1e-6 {
        0.0
    } else {
        (sxy / (sxx - syy + 1e-6)).atan() / 2.0
    };
    let mut poly: Vec<(f32, f32)> = (0..=7)
        .map(|k| {
            let theta = k as f32 * std::f32::consts::PI / 4.0 + angle;
            (mx + a * theta.cos(), my + b * theta.sin())
        })
        .collect();
    // Close the polygon
    poly.push(poly[0]);
    poly
}
/// Moore-neighbor outer boundary trace of one 4-connected component
/// (Jacob's stopping criterion). Returns a closed loop in pixel coords.
fn moore_outline(labels: &[u32], id: u32, c: &Component, w: usize, h: usize) -> Vec<(f32, f32)> {
    // Clockwise neighbor order for y-down images, starting index included.
    const DX: [i32; 8] = [-1, -1, 0, 1, 1, 1, 0, -1]; // W NW N NE E SE S SW
    const DY: [i32; 8] = [0, -1, -1, -1, 0, 1, 1, 1];
    let at = |x: i32, y: i32| -> bool {
        x >= c.x0 as i32
            && y >= c.y0 as i32
            && x < c.x1 as i32
            && y < c.y1 as i32
            && x >= 0
            && y >= 0
            && (x as usize) < w
            && (y as usize) < h
            && labels[y as usize * w + x as usize] == id
    };
    // Start at the leftmost pixel of the topmost row: the pixel above it is
    // background, so backtrack-from-north is valid.
    let (mut sx, sy) = (c.x1, c.y0);
    for x in c.x0..c.x1 {
        if at(x as i32, sy as i32) {
            sx = x;
            break;
        }
    }
    if sx == c.x1 {
        return Vec::new();
    }
    let (sx, sy) = (sx as i32, sy as i32);
    let mut boundary = vec![(sx as f32, sy as f32)];
    // Backtrack direction: north of start (background by construction).
    let (mut cx, mut cy) = (sx, sy);
    // Direction index of `backtrack` as seen from `current`.
    let mut back_dir = 2; // N
    let (mut fx, mut fy) = (sx, sy);
    let mut first = true;
    loop {
        // Scan clockwise starting with the neighbor after backtrack.
        let mut stepped = false;
        for k in 1..=8 {
            let dir = (back_dir + k) % 8;
            let (nx, ny) = (cx + DX[dir], cy + DY[dir]);
            if at(nx, ny) {
                if first {
                    (fx, fy) = (nx, ny);
                    first = false;
                } else if cx == sx && cy == sy && nx == fx && ny == fy {
                    // Re-entered start via the first step: loop closed.
                    boundary.push((sx as f32, sy as f32));
                    return boundary;
                }
                (cx, cy) = (nx, ny);
                back_dir = (dir + 4) % 8;
                boundary.push((cx as f32, cy as f32));
                stepped = true;
                break;
            }
        }
        if !stepped || boundary.len() > c.area * 4 + 16 {
            // Isolated pixel (no fg neighbor) or runaway: close what we have.
            boundary.push((sx as f32, sy as f32));
            return boundary;
        }
    }
}

/// Polyline arc length in chain (small-px) units.
fn arc_len(c: &[(f32, f32)]) -> f32 {
    c.windows(2)
        .map(|w| ((w[1].0 - w[0].0).powi(2) + (w[1].1 - w[0].1).powi(2)).sqrt())
        .sum()
}

/// Douglas-Peucker simplification (recursive; chains are short).
fn simplify_dp(pts: &[(f32, f32)], tol: f32) -> Vec<(f32, f32)> {
    if pts.len() <= 2 {
        return pts.to_vec();
    }
    let (ax, ay) = pts[0];
    let (bx, by) = pts[pts.len() - 1];
    let (dx, dy) = (bx - ax, by - ay);
    let den = (dx * dx + dy * dy).sqrt();
    let mut max_d = 0.0f32;
    let mut idx = 0;
    for (i, &(px, py)) in pts.iter().enumerate().skip(1).take(pts.len() - 2) {
        let d = if den == 0.0 {
            ((px - ax).powi(2) + (py - ay).powi(2)).sqrt()
        } else {
            (dy * px - dx * py + bx * ay - by * ax).abs() / den
        };
        if d > max_d {
            max_d = d;
            idx = i;
        }
    }
    if max_d <= tol {
        return vec![pts[0], pts[pts.len() - 1]];
    }
    let mut left = simplify_dp(&pts[..=idx], tol);
    let right = simplify_dp(&pts[idx..], tol);
    left.pop();
    left.extend(right);
    left
}

/// Zhang-Suen thinning: reduce foreground (<128) to a 1px 8-connected
/// skeleton. Iterates two sub-cycles until no pixel changes (capped).
fn zhang_suen(v: &[u8], w: usize, h: usize) -> Vec<bool> {
    let mut fg: Vec<bool> = v.iter().map(|&px| px < 128).collect();
    // Neighbor offsets in p2..p9 order: N, NE, E, SE, S, SW, W, NW.
    const DX: [i32; 8] = [0, 1, 1, 1, 0, -1, -1, -1];
    const DY: [i32; 8] = [-1, -1, 0, 1, 1, 1, 0, -1];
    for _ in 0..100 {
        let mut removed_any = false;
        for step in 0..2 {
            let mut remove = Vec::new();
            for y in 1..h.saturating_sub(1) {
                for x in 1..w.saturating_sub(1) {
                    let i = y * w + x;
                    if !fg[i] {
                        continue;
                    }
                    let mut n = [false; 8];
                    for (k, px) in n.iter_mut().enumerate() {
                        let nx = x as i32 + DX[k];
                        let ny = y as i32 + DY[k];
                        *px = fg[ny as usize * w + nx as usize];
                    }
                    let b = n.iter().filter(|&&b| b).count();
                    if !(2..=6).contains(&b) {
                        continue;
                    }
                    let mut a = 0;
                    for k in 0..8 {
                        if !n[k] && n[(k + 1) % 8] {
                            a += 1;
                        }
                    }
                    if a != 1 {
                        continue;
                    }
                    // p2=N, p4=E, p6=S, p8=W in n[0..8] order.
                    let (p2, p4, p6, p8) = (n[0], n[2], n[4], n[6]);
                    let ok = if step == 0 {
                        !(p2 && p4 && p6) && !(p4 && p6 && p8)
                    } else {
                        !(p2 && p4 && p8) && !(p2 && p6 && p8)
                    };
                    if ok {
                        remove.push(i);
                    }
                }
            }
            if !remove.is_empty() {
                removed_any = true;
                for i in remove {
                    fg[i] = false;
                }
            }
        }
        if !removed_any {
            break;
        }
    }
    fg
}

/// Keep pixels darker than `low` that connect (8-way) to a pixel darker
/// than `high`.
fn hysteresis(v: &[u8], w: usize, h: usize, high: u8, low: u8) -> Vec<bool> {
    let mut kept = vec![false; w * h];
    let mut stack = Vec::new();
    for (i, &px) in v.iter().enumerate() {
        if px < high {
            kept[i] = true;
            stack.push(i);
        }
    }
    while let Some(i) = stack.pop() {
        let (x, y) = (i % w, i / w);
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                    continue;
                }
                let j = ny as usize * w + nx as usize;
                if !kept[j] && v[j] < low {
                    kept[j] = true;
                    stack.push(j);
                }
            }
        }
    }
    kept
}

/// Zero connected components smaller than `min_size` px (area opening).
fn sweep_small(kept: &mut [bool], w: usize, h: usize, min_size: usize) {
    let mut seen = vec![false; w * h];
    for i in 0..w * h {
        if !kept[i] || seen[i] {
            continue;
        }
        let mut comp = Vec::new();
        let mut stack = vec![i];
        seen[i] = true;
        while let Some(j) = stack.pop() {
            comp.push(j);
            let (x, y) = (j % w, j / w);
            if x > 0 && kept[j - 1] && !seen[j - 1] {
                seen[j - 1] = true;
                stack.push(j - 1);
            }
            if x + 1 < w && kept[j + 1] && !seen[j + 1] {
                seen[j + 1] = true;
                stack.push(j + 1);
            }
            if y > 0 && kept[j - w] && !seen[j - w] {
                seen[j - w] = true;
                stack.push(j - w);
            }
            if y + 1 < h && kept[j + w] && !seen[j + w] {
                seen[j + w] = true;
                stack.push(j + w);
            }
        }
        if comp.len() < min_size {
            for j in comp {
                kept[j] = false;
            }
        }
    }
}

/// Box-average one R/G/B channel straight from the RGB image down to small
/// size (box average, not nearest point: nearest sampling aliases the edge
/// phase row-to-row, which dithers the threshold into dotted lines).
fn sample_channel(rgb: &RgbImage, w: usize, h: usize, sw: u32, sh: u32, ch: usize) -> GrayImage {
    // Box-average (not nearest point): nearest sampling aliases the edge
    // phase row-to-row, which dithers the threshold into dotted lines.
    let (sw, sh) = (sw as usize, sh as usize);
    let raw_px = rgb.as_raw();
    let mut raw = vec![0u8; sw * sh];
    for y in 0..sh {
        let y0 = y * h / sh;
        let y1 = ((y + 1) * h / sh).max(y0 + 1);
        for x in 0..sw {
            let x0 = x * w / sw;
            let x1 = ((x + 1) * w / sw).max(x0 + 1);
            let mut acc = 0u32;
            let mut n = 0u32;
            for sy in y0..y1 {
                for sx in x0..x1 {
                    acc += raw_px[(sy * w + sx) * 3 + ch] as u32;
                    n += 1;
                }
            }
            raw[y * sw + x] = (acc / n) as u8;
        }
    }
    GrayImage::from_raw(sw as u32, sh as u32, raw).expect("small channel")
}

/// Nearest-upscale a u8 buffer (fast path around the slow generic resize).
fn upscale_nearest_u8(src: &[u8], sw: u32, sh: u32, w: u32, h: u32) -> Vec<u8> {
    let (sw, sh, w, h) = (sw as usize, sh as usize, w as usize, h as usize);
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        let sy = (y * sh / h).min(sh - 1);
        for x in 0..w {
            out[y * w + x] = src[sy * sw + (x * sw / w).min(sw - 1)];
        }
    }
    out
}

fn mask_to_color(mask: &[bool], w: u32, h: u32) -> ColorImage {
    let mut pixels = Vec::with_capacity((w * h * 4) as usize);
    for &b in mask {
        let v = if b { 0u8 } else { 255u8 };
        pixels.extend_from_slice(&[v, v, v, 255]);
    }
    ColorImage {
        pixels,
        width: w as usize,
        height: h as usize,
    }
}

/// Downscaled PNG for the UI sidebar (None if encoding fails).
fn png_thumb_gray(g: &[u8], w: u32, h: u32) -> Option<Vec<u8>> {
    let full = GrayImage::from_raw(w, h, g.to_vec())?;
    let thumb = image::imageops::thumbnail(&full, 480, 480);
    let mut buf = Vec::new();
    image::DynamicImage::ImageLuma8(thumb)
        .write_to(&mut Cursor::new(&mut buf), ImageFormat::Png)
        .ok()?;
    Some(buf)
}

fn png_thumb_mask(mask: &[bool], w: u32, h: u32) -> Option<Vec<u8>> {
    png_thumb_gray(
        &mask
            .iter()
            .map(|&b| if b { 255u8 } else { 0 })
            .collect::<Vec<_>>(),
        w,
        h,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageFormat, Rgb, RgbImage};
    use std::io::Cursor;

    /// Synthetic gray T-shirt on white: torso + sleeves + collar + hem seam.
    fn synth_tee_rgb() -> RgbImage {
        let (w, h) = (120u32, 140u32);
        let mut img = RgbImage::from_pixel(w, h, Rgb([255, 255, 255]));
        let tee = Rgb([150, 150, 150]);
        let dark = Rgb([60, 60, 60]);
        let rect = |img: &mut RgbImage, x0, y0, x1, y1, c| {
            for y in y0..y1 {
                for x in x0..x1 {
                    img.put_pixel(x, y, c);
                }
            }
        };
        rect(&mut img, 40, 30, 80, 120, tee); // torso
        rect(&mut img, 18, 34, 40, 56, tee); // left sleeve
        rect(&mut img, 80, 34, 102, 56, tee); // right sleeve
                                              // collar: dark arc dip in the middle top
        for x in 50..70 {
            let dx = (x as i32 - 60) as f32;
            let y = 30 + (dx * dx / 14.0) as u32;
            for yy in 30..y + 2 {
                img.put_pixel(x, yy, dark);
            }
        }
        rect(&mut img, 42, 112, 78, 114, dark); // hem seam
        rect(&mut img, 36, 36, 38, 54, dark); // left sleeve seam
        rect(&mut img, 82, 36, 84, 54, dark); // right sleeve seam
        img
    }

    fn synth_tee_png() -> Vec<u8> {
        let img = synth_tee_rgb();
        let mut buf = Vec::new();
        img.write_to(&mut Cursor::new(&mut buf), ImageFormat::Png)
            .unwrap();
        buf
    }

    #[test]
    fn mask_finds_garment_not_backdrop() {
        let rgb = synth_tee_rgb();
        let lum = luminance(&rgb);
        let mask = foreground_mask(&lum, 120, 140);
        let at = |x, y| mask[(y * 120 + x) as usize];
        assert!(!at(0, 0), "corner is backdrop");
        assert!(!at(119, 139), "corner is backdrop");
        assert!(at(60, 80), "torso center is garment");
        assert!(at(25, 45), "sleeve is garment");
        let n: usize = mask.iter().filter(|&&b| b).count();
        assert!((2000..9000).contains(&n), "fg size sane, got {n}");
    }

    #[test]
    fn symmetrize_mirrors_each_view_in_its_own_box() {
        // Near-symmetric view: 4x2 block + one extra pixel mirrors the pixel
        // across and reports symmetrized.
        let (w, h) = (9usize, 4usize);
        let mut m = vec![false; w * h];
        for y in 1..3 {
            for x in 2..6 {
                m[y * w + x] = true;
            }
        }
        m[3 * w + 2] = true;
        let (labels, mut comps) = label_components(&m, w, h);
        assert_eq!(comps.len(), 2, "one view");
        symmetrize_components(&mut m, &labels, &mut comps, w, h);
        assert!(comps[1].symmetrized, "near-symmetric view averaged");
        assert!(m[3 * w + 5], "extra pixel mirrored within its own box");
        assert!(!m[3 * w + 8], "mirror never leaves the view's box");
    }

    #[test]
    fn symmetrize_leaves_side_views_untouched() {
        // L shape (side-profile-like): mirroring the bar would grow the area
        // ~58%, so the view must pass through unchanged.
        let (w, h) = (9usize, 8usize);
        let mut m = vec![false; w * h];
        for y in 0..8 {
            m[y * w + 2] = true;
        }
        for x in 2..7 {
            m[7 * w + x] = true;
        }
        let before = m.clone();
        let (labels, mut comps) = label_components(&m, w, h);
        symmetrize_components(&mut m, &labels, &mut comps, w, h);
        assert!(!comps[1].symmetrized, "asymmetric view flagged");
        assert_eq!(m, before, "asymmetric view untouched");
    }

    #[test]
    fn sweep_keeps_every_view_drops_specks() {
        // Two garment views + one speck: both views survive, speck goes.
        let (w, h) = (30usize, 10usize);
        let mut m = vec![false; w * h];
        for y in 2..7 {
            for x in 2..7 {
                m[y * w + x] = true;
            }
            for x in 20..25 {
                m[y * w + x] = true;
            }
        }
        m[0] = true;
        sweep_small_components(&mut m, w, h, 12);
        let n: usize = m.iter().filter(|&&b| b).count();
        assert_eq!(n, 50, "both views kept, speck dropped, got {n}");
        let (_, comps) = label_components(&m, w, h);
        assert_eq!(comps.len(), 3, "two views labelled");
    }

    #[test]
    fn erode_shrinks_by_radius() {
        // 5x5 solid block in 9x9: radius-1 erosion leaves the inner 3x3.
        let mut m = vec![false; 81];
        for y in 2..7 {
            for x in 2..7 {
                m[y * 9 + x] = true;
            }
        }
        let e = erode(&m, 9, 9, 1);
        assert!(e[4 * 9 + 4], "center survives");
        assert!(!e[2 * 9 + 2], "original corner eaten");
        assert!(!e[0], "outside stays out");
        assert_eq!(e.iter().filter(|&&b| b).count(), 9);
    }

    #[test]
    fn thinning_reduces_bar_to_centerline() {
        // 9x9 canvas, 3px-wide horizontal bar -> single center row.
        let mut m = vec![255u8; 81];
        for y in 3..6 {
            for x in 1..8 {
                m[y * 9 + x] = 0;
            }
        }
        let t = zhang_suen(&m, 9, 9);
        // Center span survives (blunt ends may shorten asymmetrically by a
        // pixel — known directional bias of two-subcycle thinning).
        for x in 3..6 {
            assert!(t[4 * 9 + x], "center survives at {x}");
        }
        assert!(!t[3 * 9 + 4] && !t[5 * 9 + 4], "outer rows eaten");
    }

    #[test]
    fn smooth_mask_keeps_centered_bulk() {
        // 20x20 block centered in 30x30 + lone speck: open+close keeps the
        // block exactly and drops the speck.
        let mut m = vec![false; 900];
        for y in 5..25 {
            for x in 5..25 {
                m[y * 30 + x] = true;
            }
        }
        m[0] = true;
        let s = smooth_mask(&m, 30, 30, 2);
        assert_eq!(s.iter().filter(|&&b| b).count(), 400);
    }

    #[test]
    fn xdog_catches_isoluminant_chroma_edges() {
        // Gold-on-denim button: chroma step with a near-flat luminance field
        // (luminance-only XDoG is blind to it). Denim [70,100,170] lum=0.387,
        // button [150,90,40] lum=0.389: |dLum| < 0.01, but R jumps +80 and
        // B drops -130 across the rim.
        let (w, h) = (120u32, 140u32);
        let mut img = RgbImage::from_pixel(w, h, Rgb([255, 255, 255]));
        let denim = Rgb([70, 100, 170]);
        let gold = Rgb([150, 90, 40]);
        for y in 40..120 {
            for x in 30..90 {
                img.put_pixel(x, y, denim);
            }
        }
        for y in 65..95 {
            for x in 45..75 {
                let dx = x as i32 - 60;
                let dy = y as i32 - 80;
                if dx * dx + dy * dy <= 100 {
                    img.put_pixel(x, y, gold);
                }
            }
        }
        let lum: Vec<f32> = img
            .pixels()
            .map(|p| (0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32) / 255.0)
            .collect();
        let (mut lo, mut hi) = (1.0f32, 0.0f32);
        for y in 65..95 {
            for x in 45..75 {
                let v = lum[(y * w + x) as usize];
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        assert!(
            hi - lo < 0.02,
            "test field must be isoluminant, got {lo:.3}..{hi:.3}"
        );
        let xd = xdog_lines(&img, 0.6);
        let near = xd
            .chains
            .iter()
            .filter(|c| {
                !c.is_empty()
                    && c.iter().any(|&(px, py)| {
                        (40.0..=80.0).contains(&px) && (60.0..=100.0).contains(&py)
                    })
            })
            .count();
        assert!(near > 0, "button rim must produce linework");
    }

    #[test]
    fn flat_pipeline_produces_flat_svg() {
        let png = synth_tee_png();
        let out = convert_flat_bytes(&png, &FlatOptions::default()).unwrap();
        assert_eq!((out.width, out.height), (120, 140));
        assert!(out.svg.starts_with("<svg"), "single composed svg root");
        assert!(
            out.svg.contains("id=\"silhouette\""),
            "has silhouette group"
        );
        assert!(out.svg.contains("id=\"details\""), "has details group");
        assert!(out.svg.contains("stroke-width"), "silhouette is stroked");
        assert!(
            out.svg.contains("fill=\"none\""),
            "details are stroked polylines"
        );
        assert!(!out.svg.contains("stroke-dasharray"), "solid by default");
        // Dashed mode emits the stitch convention.
        let out_d = convert_flat_bytes(
            &png,
            &FlatOptions {
                stitch_dashed: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(
            out_d.svg.contains("stroke-dasharray=\"7 4\""),
            "dashed stitches"
        );
        assert!(
            out.path_count >= 2,
            "silhouette + at least one detail, got {}",
            out.path_count
        );
        // No vtracer xml preamble leaks into the composed file.
        assert!(!out.svg.contains("Generator"), "no nested vtracer header");
    }

    #[test]
    fn chains_split_at_junctions() {
        // T shape (9x9): vertical x4 rows 1..8 + arms (2..7,4).
        // Junction at (4,4) fans into separate chains.
        let mut m = vec![false; 81];
        for y in 1..8 {
            m[y * 9 + 4] = true;
        }
        for x in 2..7 {
            m[4 * 9 + x] = true;
        }
        let chains = trace_chains(&m, 9, 9);
        assert!(chains.len() >= 3, "junction fans out, got {}", chains.len());
        for c in &chains {
            assert!(c.len() >= 2, "no dot chains");
        }
    }

    #[test]
    fn dp_collapses_straight_runs() {
        let line: Vec<(f32, f32)> = (0..20).map(|x| (x as f32, 3.0)).collect();
        let s = simplify_dp(&line, 1.0);
        assert_eq!(s.len(), 2, "straight -> endpoints, got {s:?}");
        assert_eq!(s[0], (0.0, 3.0));
        assert_eq!(s[1], (19.0, 3.0));
    }

    #[test]
    fn spurs_pruned_dash_kept() {
        // 20px line with a 2px twig at x10 + an isolated 5px dash.
        // The twig tip is removed (its 1px base nub may remain; the tracer's
        // arc-length filter drops such nubs and merges across them).
        let (w, h) = (30usize, 12usize);
        let mut m = vec![false; w * h];
        for x in 2..22 {
            m[5 * w + x] = true;
        }
        m[6 * w + 10] = true;
        m[7 * w + 10] = true;
        for x in 24..29 {
            m[8 * w + x] = true;
        }
        prune_spurs(&mut m, w, h, 5);
        assert!(!m[7 * w + 10], "twig tip removed");
        assert!(m[5 * w + 10], "junction root survives");
        assert!(m[5 * w + 3] && m[5 * w + 21], "main line intact");
        assert!(m[8 * w + 26], "isolated dash kept");
    }

    #[test]
    fn staircase_merges_v_corner_does_not() {
        // Staircase fragments turn 90 deg but still fit one chord (dev 0.89).
        let mut chains = vec![
            vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
            vec![(2.0, 0.0), (2.0, 1.0), (2.0, 2.0), (3.0, 2.0), (4.0, 2.0)],
        ];
        merge_collinear(&mut chains, 3.0);
        assert_eq!(chains.len(), 1, "staircase merges, got {}", chains.len());
        // V corner (dev 3.5) must stay split for a sharp collar point.
        let mut corner = vec![
            vec![(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)],
            vec![(11.0, 1.0), (16.0, 6.0)],
        ];
        merge_collinear(&mut corner, 3.0);
        assert_eq!(
            corner.len(),
            2,
            "V corner stays split, got {}",
            corner.len()
        );
    }

    #[test]
    fn tangent_arc_fragments_merge() {
        // Two fragments of a r=20 arc (5 deg steps). Combined chord dev is
        // ~1.9, above FIT_TOL, but end tangents agree (~15 deg apart), so the
        // fallback must join them into one lapel-like curve.
        let mut chains = vec![
            vec![
                (20.000, 0.000),
                (19.924, 1.743),
                (19.696, 3.473),
                (19.319, 5.176),
                (18.794, 6.840),
            ],
            vec![
                (17.321, 10.000),
                (16.383, 11.471),
                (15.321, 12.856),
                (14.142, 14.142),
                (12.856, 15.321),
            ],
        ];
        merge_collinear(&mut chains, 6.0);
        assert_eq!(
            chains.len(),
            1,
            "arc joins via tangents, got {}",
            chains.len()
        );
        assert_eq!(chains[0].len(), 10);
    }

    #[test]
    fn collinear_ends_merge_corners_dont() {
        let mut chains = vec![
            vec![(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)],
            vec![(11.0, 0.0), (16.0, 0.0), (21.0, 0.0)],
            vec![(10.0, 1.0), (10.0, 6.0)],
        ];
        merge_collinear(&mut chains, 4.0);
        assert_eq!(chains.len(), 2, "one merge, got {}", chains.len());
        let long = chains.iter().find(|c| c.len() == 6).expect("merged chain");
        assert_eq!(long[0], (0.0, 0.0));
        assert_eq!(long[5], (21.0, 0.0));
    }

    #[test]
    fn onmodel_rejected_with_clear_error() {
        let png = synth_tee_png();
        let err = convert_flat_bytes(
            &png,
            &FlatOptions {
                input: FlatInput::OnModel,
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(err.to_string().contains("Phase-2"), "got: {err}");
    }
}
