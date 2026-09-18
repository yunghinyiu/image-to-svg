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
use image::{GrayImage, RgbImage};

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
            outline_width: 2.0,
            detail_strength: 0.6,
            speckle: 4,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FlatOutput {
    pub svg: String,
    pub width: u32,
    pub height: u32,
    pub path_count: usize,
    pub svg_bytes: usize,
}

pub fn convert_flat_bytes(bytes: &[u8], opts: &FlatOptions) -> Result<FlatOutput> {
    if opts.input == FlatInput::OnModel {
        bail!("on-model photos need the Phase-2 ML segmenter (segformer clothes, ONNX) which is not bundled yet — use flat-lay / ghost-mannequin photos for now");
    }
    let img = image::load_from_memory(bytes).context("decode image (png/jpg/webp/...)")?;
    let mut rgb = img.to_rgb8();
    if rgb.width().max(rgb.height()) > MAX_SIDE {
        let scale = MAX_SIDE as f32 / rgb.width().max(rgb.height()) as f32;
        let (nw, nh) = (
            (rgb.width() as f32 * scale).round() as u32,
            (rgb.height() as f32 * scale).round() as u32,
        );
        rgb = image::imageops::resize(
            &rgb,
            nw.max(1),
            nh.max(1),
            image::imageops::FilterType::Triangle,
        );
    }
    convert_flat_rgb(&rgb, opts)
}

fn convert_flat_rgb(rgb: &RgbImage, opts: &FlatOptions) -> Result<FlatOutput> {
    let (w, h) = (rgb.width(), rgb.height());
    let lum = luminance(rgb);

    // 1. foreground mask via backdrop keying.
    let mut mask = foreground_mask(&lum, w, h);
    if opts.symmetrize {
        symmetrize_mask(&mut mask, w, h);
    }
    if !mask.iter().any(|&b| b) {
        bail!("no garment found — flat mode needs a plain, bright backdrop behind the garment");
    }

    // 2. silhouette pass: black garment on white.
    let sil_img = mask_to_color(&mask, w, h);
    let sil = convert_image(&sil_img, w, h, &trace_opts(opts.speckle))?;
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

    // 3. detail pass: XDoG lines inside (slightly dilated) garment region.
    let region = dilate(&mask, w, h, 2);
    let mut lines = xdog_lines(&lum, w, h, opts.detail_strength);
    for (i, v) in lines.iter_mut().enumerate() {
        if !region[i] {
            *v = 255;
        }
    }
    if opts.symmetrize {
        symmetrize_gray_max(&mut lines, w, h);
    }
    let det_img = gray_to_color(&lines, w, h);
    let det = convert_image(&det_img, w, h, &trace_opts(opts.speckle))?;
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

/// Extended Difference-of-Gaussians: dark stylized lines on white.
/// `strength` 0..=1 maps to the epsilon threshold (lower eps = more lines).
fn xdog_lines(lum: &[f32], w: u32, h: u32, strength: f32) -> Vec<u8> {
    let (wu, hu) = (w as usize, h as usize);
    let raw: Vec<u8> = lum
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 255.0) as u8)
        .collect();
    let gray = GrayImage::from_raw(w, h, raw).expect("gray buffer size");
    let sigma = 1.4f32;
    let b1 = image::imageops::blur(&gray, sigma);
    let b2 = image::imageops::blur(&gray, sigma * 4.0);
    let tau = 0.98f32;
    let phi = 20.0f32;
    // eps window is tight: interior DoG floor sits near +0.01, so eps must
    // stay negative; -0.15 keeps only the strongest edges, -0.008 everything.
    let eps = (-0.15 + 0.20 * strength.clamp(0.0, 1.0)).min(-0.008);
    let mut out = vec![255u8; wu * hu];
    for (i, o) in out.iter_mut().enumerate() {
        let d = b1.as_raw()[i] as f32 / 255.0 - tau * b2.as_raw()[i] as f32 / 255.0;
        // XDoG soft threshold: values below eps become dark lines.
        let v = if d >= eps {
            1.0
        } else {
            1.0 + (phi * (d - eps)).tanh()
        };
        *o = (v.clamp(0.0, 1.0) * 255.0) as u8;
    }
    // Binarize for a clean trace: lines black, rest white.
    for v in out.iter_mut() {
        *v = if *v < 235 { 0 } else { 255 };
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
