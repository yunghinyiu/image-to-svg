//! Shape-from-shading prototype runner.
//!
//! ```sh
//! cargo run -q -p im2vec-flat --example shading_proto
//! ```
//!
//! Outputs:
//! - /tmp/shading-normal.png: RGB normal map
//! - /tmp/shading-curvature.png: curvature (fold) heatmap

use image::GenericImageView;

fn main() {
    let input_png = std::fs::read("samples/blazer/input.png").unwrap();
    let img = image::load_from_memory(&input_png).unwrap();

    // Downscale to pipeline size (1600x900) for consistency
    let scaled = img.resize(1600, 900, image::imageops::FilterType::Lanczos3);
    let gray = scaled.to_luma8();
    let (w, h) = gray.dimensions();

    println!("Estimating normals for {}x{}...", w, h);
    let normals = im2vec_flat::shading::estimate_normals(&gray);

    println!("Computing curvature...");
    let curv = im2vec_flat::shading::curvature_from_normals(&normals, w as usize, h as usize);

    println!("Rendering...");
    let normal_map = im2vec_flat::shading::render_normal_map(&normals, w, h);
    normal_map.save("/tmp/shading-normal.png").unwrap();

    let curv_map = im2vec_flat::shading::render_curvature(&curv, w, h);
    curv_map.save("/tmp/shading-curvature.png").unwrap();

    // Stats
    let max_c = curv.iter().cloned().fold(0.0f32, f32::max);
    let mean_c = curv.iter().sum::<f32>() / curv.len() as f32;
    println!("Curvature: max={:.3}, mean={:.4}", max_c, mean_c);
    println!("Wrote /tmp/shading-normal.png and /tmp/shading-curvature.png");
}
