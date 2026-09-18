//! Clothing photo -> tech-pack flat sketch (draft quality).
//!
//! Phase 1 is fully deterministic and targets flat-lay / ghost-mannequin
//! photos on plain backdrops:
//!
//! 1. background keying (backdrop color estimated from border pixels)
//! 2. XDoG stylized line extraction for seams, folds, trims
//! 3. mirror symmetrization around the vertical center axis
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
    if std::env::var("IM2VEC_FLAT_DEBUG").is_ok() {
        eprintln!("decode+resize: {} ms", t0.elapsed().as_millis());
    }
    convert_flat_rgb(&rgb, opts)
}

fn convert_flat_rgb(rgb: &RgbImage, opts: &FlatOptions) -> Result<FlatOutput> {
    let (w, h) = (rgb.width(), rgb.height());
    let lum = luminance(rgb);
    let mut stages: Vec<FlatStage> = Vec::new();

    // 1. foreground mask via backdrop keying.
    let t = Instant::now();
    let mut mask = smooth_mask(&foreground_mask(&lum, w, h), w, h, 2);
    if opts.symmetrize {
        symmetrize_mask(&mut mask, w, h);
    }
    stage(&mut stages, "background keying", t);
    if !mask.iter().any(|&b| b) {
        bail!("no garment found — flat mode needs a plain, bright backdrop behind the garment");
    }

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
    let mut xd = xdog_lines(&lum, w, h, opts.detail_strength);
    let mut lines = xd.full.clone();
    for (i, v) in lines.iter_mut().enumerate() {
        if !region[i] {
            *v = 255;
        }
    }
    if opts.symmetrize_lines {
        symmetrize_gray_max(&mut lines, w, h);
        for c in xd.chains.iter_mut() {
            for p in c.iter_mut() {
                p.0 = xd.sw as f32 - 1.0 - p.0;
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
/// dropped by keeping the largest connected component.
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

    keep_largest_component(&mut fg, w, h);
    fg
}

/// Zero all but the largest 4-connected foreground component.
fn keep_largest_component(fg: &mut [bool], w: usize, h: usize) {
    let mut labels = vec![0u32; w * h];
    let mut sizes: Vec<usize> = vec![0]; // 1-based
    let mut next = 0u32;
    for i in 0..w * h {
        if !fg[i] || labels[i] != 0 {
            continue;
        }
        next += 1;
        let mut stack = vec![i];
        let mut size = 0;
        while let Some(j) = stack.pop() {
            if !fg[j] || labels[j] != 0 {
                continue;
            }
            labels[j] = next;
            size += 1;
            let (x, y) = (j % w, j / w);
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
        sizes.push(size);
    }
    if next == 0 {
        return;
    }
    let best = sizes
        .iter()
        .enumerate()
        .skip(1)
        .max_by_key(|&(_, &s)| s)
        .map(|(l, _)| l as u32)
        .unwrap();
    for (i, f) in fg.iter_mut().enumerate() {
        if *f && labels[i] != best {
            *f = false;
        }
    }
}

fn symmetrize_mask(mask: &mut [bool], w: u32, h: u32) {
    let (w, h) = (w as usize, h as usize);
    for y in 0..h {
        for x in 0..w / 2 {
            let (a, b) = (y * w + x, y * w + (w - 1 - x));
            let v = mask[a] || mask[b];
            mask[a] = v;
            mask[b] = v;
        }
    }
}

/// Mirror-max: each mirrored pair takes the darker (stronger-line) value.
fn symmetrize_gray_max(g: &mut [u8], w: u32, h: u32) {
    let (w, h) = (w as usize, h as usize);
    for y in 0..h {
        for x in 0..w / 2 {
            let (a, b) = (y * w + x, y * w + (w - 1 - x));
            let v = g[a].min(g[b]);
            g[a] = v;
            g[b] = v;
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
/// XDoG output: full-res binary for previews plus vector chains in small-px.
struct XdogOut {
    full: Vec<u8>,
    chains: Vec<Vec<(f32, f32)>>,
    sw: u32,
    sh: u32,
}

fn xdog_lines(lum: &[f32], w: u32, h: u32, strength: f32) -> XdogOut {
    let (wu, hu) = (w as usize, h as usize);
    // Compute the response at reduced resolution (edges survive downscaling)
    // with hand-rolled sampling: the generic ops resize is ~1s at 2MP.
    let sigma = 1.4f32;
    let scale = (800.0 / w.max(h) as f32).min(1.0);
    let (sw, sh) = (
        ((w as f32 * scale).round() as u32).max(1),
        ((h as f32 * scale).round() as u32).max(1),
    );
    let debug = std::env::var("IM2VEC_FLAT_DEBUG").is_ok();
    let t = Instant::now();
    let small = sample_gray(lum, wu, hu, sw, sh);
    let b1 = fast_blur(&small, sigma * scale);
    let b2 = fast_blur(&small, sigma * 4.0 * scale);
    if debug {
        eprintln!(
            "xdog blurs: {} ms ({}x{} scale {scale:.2})",
            t.elapsed().as_millis(),
            w,
            h
        );
    }
    let tau = 0.98f32;
    let phi = 20.0f32;
    // eps window is tight: interior DoG floor sits near +0.01, so eps must
    // stay negative; -0.15 keeps only the strongest edges, -0.008 everything.
    let eps = (-0.15 + 0.20 * strength.clamp(0.0, 1.0)).min(-0.008);
    let (swu, shu) = (sw as usize, sh as usize);
    let (rb1, rb2) = (b1.as_raw(), b2.as_raw());
    let mut small_out = vec![255u8; swu * shu];
    for (i, o) in small_out.iter_mut().enumerate() {
        let d = rb1[i] as f32 / 255.0 - tau * rb2[i] as f32 / 255.0;
        // XDoG soft threshold: values below eps become dark lines.
        let v = if d >= eps {
            1.0
        } else {
            1.0 + (phi * (d - eps)).tanh()
        };
        *o = (v.clamp(0.0, 1.0) * 255.0) as u8;
    }
    // Hysteresis: confident lines seed, faint lines survive only when
    // connected to confident ones. Joins dotted seams, drops lone noise.
    // Then an area opening removes remaining tiny specks.
    let high = 210u8;
    let low = (225.0 + 10.0 * strength.clamp(0.0, 1.0)) as u8;
    let min_size = (30.0 - 20.0 * strength.clamp(0.0, 1.0)) as usize;
    let mut kept = hysteresis(&small_out, swu, shu, high, low);
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
    merge_collinear(&mut chains, 4.0);
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
/// shatter into short fragments that turn sharply, so tangent alignment is
/// the wrong test. Instead join the pair whose concatenation best fits its
/// end-to-end chord (staircase fits a line; a collar V does not). Sharp
/// corners stay split and rejoin visually via round caps.
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
    const FIT_TOL: f32 = 1.0;
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
                        if dist > gap {
                            continue;
                        }
                        // Orient i so the join is its tail, j so the join is
                        // its head, then test the chord fit.
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
                        let skip = usize::from(dist < 0.75);
                        cand.extend(other.into_iter().skip(skip));
                        let dev = chord_dev(&cand);
                        if dev <= FIT_TOL && best.map(|b| dev < b.4).unwrap_or(true) {
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

/// Nearest-sample a u8 gray image straight from the float luminance buffer.
fn sample_gray(lum: &[f32], w: usize, h: usize, sw: u32, sh: u32) -> GrayImage {
    // Box-average (not nearest point): nearest sampling aliases the edge
    // phase row-to-row, which dithers the threshold into dotted lines.
    let (sw, sh) = (sw as usize, sh as usize);
    let mut raw = vec![0u8; sw * sh];
    for y in 0..sh {
        let y0 = y * h / sh;
        let y1 = ((y + 1) * h / sh).max(y0 + 1);
        for x in 0..sw {
            let x0 = x * w / sw;
            let x1 = ((x + 1) * w / sw).max(x0 + 1);
            let mut acc = 0.0f32;
            let mut n = 0u32;
            for sy in y0..y1 {
                for sx in x0..x1 {
                    acc += lum[sy * w + sx];
                    n += 1;
                }
            }
            raw[y * sw + x] = ((acc / n as f32).clamp(0.0, 1.0) * 255.0) as u8;
        }
    }
    GrayImage::from_raw(sw as u32, sh as u32, raw).expect("small gray")
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
    fn symmetrize_is_mirror_exact() {
        let (w, h) = (9usize, 4usize);
        let mut m = vec![false; w * h];
        m[0] = true; // asymmetric speck, top-left
        m[2 * w + 6] = true;
        symmetrize_mask(&mut m, w as u32, h as u32);
        for y in 0..h {
            for x in 0..w {
                assert_eq!(
                    m[y * w + x],
                    m[y * w + (w - 1 - x)],
                    "mirror mismatch at {x},{y}"
                );
            }
        }
        assert!(m[8], "speck mirrored to top-right");
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
