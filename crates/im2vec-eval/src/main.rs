//! `im2vec-eval`: run the flat pipeline on a sample input, compare against a
//! reference technical flat, and write metrics + visual diffs.
//!
//! ```sh
//! cargo run -p im2vec-eval -- \
//!   --input samples/blazer/input.png \
//!   --target samples/blazer/target.png \
//!   --out-dir docs/eval --name baseline
//! ```
//!
//! Writes `<out-dir>/<name>.svg`, `<out-dir>/<name>-report.json`,
//! `<out-dir>/<name>-diff.png` (input | ours | reference) and
//! `<out-dir>/<name>-onion.png` (aligned overlay: reference black, ours red).

use anyhow::{Context, Result};
use clap::Parser;
use im2vec_eval::run_eval;
use im2vec_flat::FlatOptions;
use std::path::PathBuf;

/// Evaluate flat-pipeline output against a reference technical flat.
#[derive(Parser, Debug)]
#[command(name = "im2vec-eval", version)]
struct Args {
    /// Input garment photo (png/jpg/webp)
    #[arg(long)]
    input: PathBuf,
    /// Reference technical flat (black line art on white)
    #[arg(long)]
    target: PathBuf,
    /// Directory for report artifacts
    #[arg(long)]
    out_dir: PathBuf,
    /// Report name prefix (e.g. "baseline")
    #[arg(long, default_value = "baseline")]
    name: String,
    /// flat preset: 0..=1, higher keeps weaker lines
    #[arg(long, default_value_t = 0.6)]
    detail_strength: f32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let input =
        std::fs::read(&args.input).with_context(|| format!("read {}", args.input.display()))?;
    let target =
        std::fs::read(&args.target).with_context(|| format!("read {}", args.target.display()))?;
    std::fs::create_dir_all(&args.out_dir)
        .with_context(|| format!("mkdir {}", args.out_dir.display()))?;

    let opts = FlatOptions {
        detail_strength: args.detail_strength,
        ..FlatOptions::default()
    };
    let art = run_eval(&input, &target, &opts)?;
    let r = &art.report;

    std::fs::write(args.out_dir.join(format!("{}.svg", args.name)), &art.svg)?;
    std::fs::write(
        args.out_dir.join(format!("{}-report.json", args.name)),
        serde_json::to_string_pretty(r)?,
    )?;
    std::fs::write(
        args.out_dir.join(format!("{}-diff.png", args.name)),
        &art.diff_png,
    )?;
    std::fs::write(
        args.out_dir.join(format!("{}-onion.png", args.name)),
        &art.onion_png,
    )?;

    println!(
        "1:1 eval — {} vs {}",
        args.input.display(),
        args.target.display()
    );
    println!("  silhouette IoU ............ {:.3}", r.silhouette_iou);
    println!(
        "  chamfer ................... {:.1} px (norm {:.4})",
        r.chamfer_px, r.chamfer_normalized
    );
    println!(
        "    ours->target ............ {:.1} px",
        r.chamfer_ours_to_target_px
    );
    println!(
        "    target->ours ............ {:.1} px",
        r.chamfer_target_to_ours_px
    );
    println!(
        "  ink ratio (ours/target) ... {:.2} ({} vs {} px)",
        r.ink_ratio_ours_to_target, r.ink_pixels_ours, r.ink_pixels_target
    );
    println!("  paths ..................... {}", r.path_count);
    println!(
        "  buttons ................... ours {} / target {} / matched {} (err {}, tol {:.0} px)",
        r.buttons_ours,
        r.buttons_target,
        r.buttons_matched,
        r.button_position_error_px
            .map(|e| format!("{e:.1} px"))
            .unwrap_or_else(|| "n/a".to_string()),
        r.button_match_tolerance_px
    );
    println!("  artifacts -> {}", args.out_dir.display());
    Ok(())
}
