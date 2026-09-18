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
    let mut mask = foreground_mask(&lum, w, h);
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
    let sil = convert_image(&sil_img, w, h, &trace_opts(opts.speckle))?;
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

    // 3. detail pass: XDoG lines strictly inside the garment. The region is
    // eroded so inner lines never hug (and double) the silhouette outline.
    let t = Instant::now();
    let region = erode(&mask, w, h, 2);
    let mut lines = xdog_lines(&lum, w, h, opts.detail_strength);
    for (i, v) in lines.iter_mut().enumerate() {
        if !region[i] {
            *v = 255;
        }
    }
    if opts.symmetrize_lines {
        symmetrize_gray_max(&mut lines, w, h);
    }
    stage(&mut stages, "XDoG linework", t);
    let t = Instant::now();
    let det_img = gray_to_color(&lines, w, h);
    let det = convert_image(&det_img, w, h, &trace_opts(opts.speckle))?;
    stage(&mut stages, "trace details", t);
    let det_paths = extract_paths(&det.svg);

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
    svg.push_str("</g><g id=\"details\">");
    let n_det = det_paths.len();
    for p in &det_paths {
        svg.push_str(p);
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

fn trace_opts(speckle: usize) -> ConvertOptions {
    ConvertOptions {
        preset: ImPreset::Mono, // forces binary clustering
        mode: "spline".into(),
        hierarchical: "stacked".into(),
        filter_speckle: speckle,
        simplify: Some(1.0),
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
fn xdog_lines(lum: &[f32], w: u32, h: u32, strength: f32) -> Vec<u8> {
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
    let mut bin = vec![255u8; swu * shu];
    for (i, &k) in kept.iter().enumerate() {
        if k {
            bin[i] = 0;
        }
    }
    if sw == w && sh == h {
        return bin;
    }
    upscale_nearest_u8(&bin, sw, sh, w, h)
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
    let (sw, sh) = (sw as usize, sh as usize);
    let mut raw = vec![0u8; sw * sh];
    for y in 0..sh {
        let sy = (y * h / sh).min(h - 1);
        for x in 0..sw {
            let sx = (x * w / sw).min(w - 1);
            raw[y * sw + x] = (lum[sy * w + sx].clamp(0.0, 1.0) * 255.0) as u8;
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

fn gray_to_color(g: &[u8], w: u32, h: u32) -> ColorImage {
    let mut pixels = Vec::with_capacity((w * h * 4) as usize);
    for &v in g {
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
            out.path_count >= 2,
            "silhouette + at least one detail, got {}",
            out.path_count
        );
        // No vtracer xml preamble leaks into the composed file.
        assert!(!out.svg.contains("Generator"), "no nested vtracer header");
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
