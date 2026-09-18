use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use im2vec_core::{convert_bytes, ConvertOptions, ImPreset};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum PresetArg {
    Logo,
    Illustration,
    Photo,
    Mono,
    /// Clothing photo -> tech-pack flat sketch (flat-lay only for now).
    Flat,
}

impl From<PresetArg> for ImPreset {
    fn from(p: PresetArg) -> Self {
        match p {
            PresetArg::Logo => ImPreset::Logo,
            PresetArg::Illustration => ImPreset::Illustration,
            PresetArg::Photo => ImPreset::Photo,
            PresetArg::Mono => ImPreset::Mono,
            PresetArg::Flat => ImPreset::Logo, // unused: flat routes to im2vec-flat
        }
    }
}

/// Convert any raster image to a high-quality SVG.
#[derive(Parser, Debug)]
#[command(name = "im2vec", version)]
struct Args {
    /// Input image (png/jpg/webp/...)
    input: PathBuf,
    /// Output SVG path
    #[arg(short, long)]
    output: PathBuf,
    /// Preset tuned for image type
    #[arg(long, value_enum, default_value = "logo")]
    preset: PresetArg,
    /// Curve mode: spline | polygon | pixel
    #[arg(long, default_value = "spline")]
    mode: String,
    /// stacked (compact) | cutout (seam-free mosaic)
    #[arg(long, default_value = "stacked")]
    hierarchical: String,
    #[arg(long, default_value_t = 4)]
    filter_speckle: usize,
    #[arg(long)]
    max_colors: Option<usize>,
    #[arg(long)]
    simplify: Option<f64>,
    #[arg(long, default_value_t = 2)]
    path_precision: u32,
    /// color_precision bits (1-8), gradient_step layer diff
    #[arg(long)]
    color_precision: Option<i32>,
    #[arg(long)]
    gradient_step: Option<i32>,
    /// color-cluster (default) | binary | watershed
    #[arg(long, default_value = "")]
    clustering: String,
    #[arg(long)]
    watershed_detail: Option<u32>,
    /// flat preset only: flatlay | on-model (on-model needs Phase-2 ML segmenter)
    #[arg(long, default_value = "flatlay")]
    flat_input: String,
    /// flat preset only: mirror-average around the vertical center axis
    #[arg(long, default_value_t = true)]
    symmetrize: bool,
    /// flat preset only: also mirror the detail linework (off: logos/pockets stay put)
    #[arg(long, default_value_t = false)]
    symmetrize_lines: bool,
    /// flat preset only: silhouette outline stroke width in px
    #[arg(long, default_value_t = 2.0)]
    outline_width: f32,
    /// flat preset only: 0..=1, higher keeps weaker lines (fabric folds)
    #[arg(long, default_value_t = 0.6)]
    detail_strength: f32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let t = Instant::now();

    let bytes =
        std::fs::read(&args.input).with_context(|| format!("read {}", args.input.display()))?;

    if matches!(args.preset, PresetArg::Flat) {
        let flat = im2vec_flat::FlatOptions {
            input: im2vec_flat::FlatInput::parse(&args.flat_input),
            symmetrize: args.symmetrize,
            symmetrize_lines: args.symmetrize_lines,
            outline_width: args.outline_width,
            detail_strength: args.detail_strength,
            speckle: args.filter_speckle,
        };
        let out = im2vec_flat::convert_flat_bytes(&bytes, &flat)?;
        std::fs::write(&args.output, &out.svg)
            .with_context(|| format!("write {}", args.output.display()))?;
        eprintln!(
            "im2vec flat: {}x{} -> {} paths, {} bytes svg in {:?} -> {}",
            out.width,
            out.height,
            out.path_count,
            out.svg_bytes,
            t.elapsed(),
            args.output.display()
        );
        return Ok(());
    }

    let preset: ImPreset = args.preset.into();
    let mut opts = ConvertOptions::for_preset(preset);
    opts.mode = args.mode;
    opts.hierarchical = args.hierarchical;
    opts.filter_speckle = args.filter_speckle;
    if args.max_colors.is_some() {
        opts.max_colors = args.max_colors;
    }
    if args.simplify.is_some() {
        opts.simplify = args.simplify;
    }
    opts.path_precision = args.path_precision;
    if let Some(v) = args.color_precision {
        opts.color_precision = v;
    }
    if let Some(v) = args.gradient_step {
        opts.gradient_step = v;
    }
    if !args.clustering.is_empty() {
        opts.clustering = args.clustering;
    }
    if args.watershed_detail.is_some() {
        opts.watershed_detail = args.watershed_detail;
    }

    let out = convert_bytes(&bytes, &opts)?;
    std::fs::write(&args.output, &out.svg)
        .with_context(|| format!("write {}", args.output.display()))?;

    eprintln!(
        "im2vec: {}x{} -> {} paths, {} bytes svg in {:?} -> {}",
        out.width,
        out.height,
        out.path_count,
        out.svg_bytes,
        t.elapsed(),
        args.output.display()
    );
    Ok(())
}
