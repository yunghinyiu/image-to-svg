//! Per-view silhouette IoU diagnostic (Phase 2).
//!
//! Splits our mask and the reference silhouette into left-to-right garment
//! views, aligns each view pair, and reports IoU under uniform and
//! anisotropic alignment.
//!
//! ```sh
//! cargo run -q -p im2vec-eval --example perview -- \
//!   --input samples/blazer/input.png --target samples/blazer/target.png
//! ```

use im2vec_eval::{
    per_view_silhouette_iou, resize_max_side, silhouette_from_lineart, EVAL_MAX_SIDE,
};
use im2vec_flat::{flat_garment_mask, FlatOptions};

fn main() {
    let mut input = None;
    let mut target = None;
    let mut no_sym = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--input" => input = args.next(),
            "--target" => target = args.next(),
            "--no-sym" => no_sym = true,
            _ => {}
        }
    }
    let (input, target) = (input.expect("--input"), target.expect("--target"));

    let opts = FlatOptions {
        symmetrize: !no_sym,
        ..FlatOptions::default()
    };
    let input_png = std::fs::read(&input).unwrap();
    let mask = flat_garment_mask(&input_png, &opts).unwrap();

    let target_png = std::fs::read(&target).unwrap();
    let target_gray = image::load_from_memory(&target_png).unwrap().to_luma8();
    let target_small = resize_max_side(
        &target_gray,
        EVAL_MAX_SIDE,
        image::imageops::FilterType::Lanczos3,
    );
    let target_sil = silhouette_from_lineart(&target_small);

    let ours_small = resize_max_side(&mask, EVAL_MAX_SIDE, image::imageops::FilterType::Nearest);

    let views = per_view_silhouette_iou(&ours_small, &target_sil);
    println!(
        "views: ours={} target={} (paired left-to-right)",
        views.len(),
        views.len()
    );
    for (k, v) in views.iter().enumerate() {
        println!(
            "view {k}: uniform IoU {:.4} | aniso IoU {:.4} (sx {:.3} sy {:.3}) | ours {:?} target {:?}",
            v.iou_uniform, v.iou_aniso, v.aniso_sx, v.aniso_sy, v.our_box, v.target_box
        );
    }
}
