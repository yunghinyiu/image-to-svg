//! im2vec-core: logo-first raster -> vector pipeline built on `vtracer`.
//!
//! Pipeline stages:
//!   decode (any format via `image`) -> vtracer segment -> fit curves -> SVG
//!
//! We intentionally depend on `vtracer` (MIT, pure Rust) rather than forking
//! on day one. If we outgrow it (custom gradient meshes, ML segmentation),
//! vtracer's pluggable `Pipeline`/`Frontend` traits give us a seam to fork into.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use vtracer::{Clustering, Config, FitMode, Hierarchical, Preset};

/// Re-exported so sibling crates (e.g. `im2vec-flat`) can feed pixels in.
pub use vtracer::ColorImage;

/// High-level preset tuned for our roadmap: logos -> illustration -> photo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ImPreset {
    /// Flat logos, icons, line-art. Small files, crisp edges.
    #[default]
    Logo,
    /// Flat illustrations, cartoons. More colors.
    Illustration,
    /// Photos / gradients / shadows. Larger files, keeps detail.
    Photo,
    /// Pure black-and-white line art.
    Mono,
}

impl ImPreset {
    fn to_vtracer_preset(self) -> Preset {
        match self {
            ImPreset::Logo | ImPreset::Illustration => Preset::Poster,
            ImPreset::Photo => Preset::Photo,
            ImPreset::Mono => Preset::Bw,
        }
    }
}

/// Tunable conversion knobs exposed in CLI + web UI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvertOptions {
    pub preset: ImPreset,
    /// Curve fitting: spline (smooth, recommended), polygon (sharp), pixel (pixel-art).
    pub mode: String,
    /// stacked = compact layered paths (default), cutout = seam-free mosaic.
    pub hierarchical: String,
    /// Discard patches smaller than X px side length. Higher = cleaner, less detail.
    pub filter_speckle: usize,
    /// Significant bits per RGB channel (1..=8). Lower = fewer colors.
    pub color_precision: i32,
    /// Color difference between gradient layers. Higher = fewer layers.
    pub gradient_step: i32,
    /// Max colors (auto-quantize). None = unlimited.
    pub max_colors: Option<usize>,
    /// Curve simplification tolerance in px. None = off. Try 1.0-2.0 for logos.
    pub simplify: Option<f64>,
    /// SVG coordinate decimal places. Lower = smaller file.
    pub path_precision: u32,
    /// "" = preset default; "color-cluster" | "binary" | "watershed"
    pub clustering: String,
    /// None = preset default (128); higher = more, smaller regions
    pub watershed_detail: Option<u32>,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        Self {
            preset: ImPreset::Logo,
            mode: "spline".into(),
            hierarchical: "stacked".into(),
            filter_speckle: 4,
            color_precision: 6,
            gradient_step: 16,
            max_colors: Some(16),
            simplify: Some(1.0),
            path_precision: 2,
            clustering: String::new(),
            watershed_detail: None,
        }
    }
}

impl ConvertOptions {
    pub fn for_preset(preset: ImPreset) -> Self {
        match preset {
            ImPreset::Logo => Self {
                preset,
                max_colors: Some(8),
                filter_speckle: 4,
                simplify: Some(1.0),
                ..Default::default()
            },
            ImPreset::Illustration => Self {
                preset,
                max_colors: Some(16),
                filter_speckle: 4,
                simplify: Some(1.0),
                ..Default::default()
            },
            ImPreset::Photo => Self {
                preset,
                max_colors: None,
                color_precision: 6,
                gradient_step: 8,
                filter_speckle: 2,
                simplify: None,
                clustering: "watershed".into(),
                watershed_detail: Some(160),
                ..Default::default()
            },
            ImPreset::Mono => Self {
                preset,
                mode: "spline".into(),
                filter_speckle: 4,
                simplify: Some(1.0),
                ..Default::default()
            },
        }
    }

    fn to_vtracer_config(&self) -> Config {
        let mut cfg = Config::from_preset(self.preset.to_vtracer_preset());

        cfg.mode = match self.mode.as_str() {
            "polygon" => FitMode::Polygon,
            "pixel" => FitMode::Pixel,
            _ => FitMode::Spline,
        };
        cfg.hierarchical = match self.hierarchical.as_str() {
            "cutout" => Hierarchical::Cutout,
            _ => Hierarchical::Stacked,
        };
        if self.preset == ImPreset::Mono {
            cfg.clustering = Clustering::Binary;
        } else {
            cfg.clustering = match self.clustering.as_str() {
                "binary" => Clustering::Binary,
                "watershed" => Clustering::Watershed,
                _ => Clustering::ColorCluster,
            };
        }
        if let Some(d) = self.watershed_detail {
            cfg.watershed_detail = d;
        }
        cfg.filter_speckle = self.filter_speckle;
        cfg.color_precision = self.color_precision.clamp(1, 8);
        cfg.layer_difference = self.gradient_step.clamp(0, 255);
        cfg.max_colors = self.max_colors;
        cfg.simplify = self.simplify;
        cfg.path_precision = Some(self.path_precision);
        cfg.optimize = 1;
        cfg
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ConvertOutput {
    pub svg: String,
    pub width: u32,
    pub height: u32,
    pub path_count: usize,
    pub svg_bytes: usize,
}

pub fn decode_to_color_image(bytes: &[u8]) -> Result<(ColorImage, u32, u32)> {
    let dyn_img = image::load_from_memory(bytes).context("decode image (png/jpg/webp/...)")?;
    let rgba = dyn_img.to_rgba8();
    let (w, h) = (rgba.width(), rgba.height());
    Ok((
        ColorImage {
            pixels: rgba.into_raw(),
            width: w as usize,
            height: h as usize,
        },
        w,
        h,
    ))
}

pub fn convert_bytes(bytes: &[u8], opts: &ConvertOptions) -> Result<ConvertOutput> {
    let (img, w, h) = decode_to_color_image(bytes)?;
    convert_image(&img, w, h, opts)
}

pub fn convert_image(
    img: &ColorImage,
    w: u32,
    h: u32,
    opts: &ConvertOptions,
) -> Result<ConvertOutput> {
    let cfg = opts.to_vtracer_config();
    let pipeline = cfg
        .build()
        .map_err(|e| anyhow::anyhow!("vtracer build: {e:?}"))?;
    let svg = pipeline
        .to_svg(img)
        .map_err(|e| anyhow::anyhow!("vtracer convert: {e:?}"))?;
    Ok(ConvertOutput {
        path_count: count_paths(&svg),
        svg_bytes: svg.len(),
        svg,
        width: w,
        height: h,
    })
}

fn count_paths(svg: &str) -> usize {
    svg.matches("<path").count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_rgba(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 4]) -> ColorImage {
        let mut pixels = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                pixels.extend_from_slice(&f(x, y));
            }
        }
        ColorImage {
            pixels,
            width: w as usize,
            height: h as usize,
        }
    }

    #[test]
    fn logo_circle_converts() {
        // 64x64 white canvas, black circle in middle -> must yield >=1 path
        let img = test_rgba(64, 64, |x, y| {
            let dx = x as i32 - 32;
            let dy = y as i32 - 32;
            if dx * dx + dy * dy < 20 * 20 {
                [0, 0, 0, 255]
            } else {
                [255, 255, 255, 255]
            }
        });
        let opts = ConvertOptions::for_preset(ImPreset::Logo);
        let out = convert_image(&img, 64, 64, &opts).expect("convert");
        assert!(out.svg.contains("<svg"), "svg root missing");
        assert!(out.path_count >= 1, "expected paths, got:\n{}", out.svg);
    }
}
