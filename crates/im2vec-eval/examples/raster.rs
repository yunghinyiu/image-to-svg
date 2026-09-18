//! Rasterize an SVG file to PNG: cargo run -p im2vec-eval --example raster -- in.svg out.png [max_side]
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: raster <in.svg> <out.png> [max_side]");
        std::process::exit(2);
    }
    let max_side: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1600);
    let svg = std::fs::read_to_string(&args[1]).expect("read svg");
    let img = im2vec_eval::rasterize_svg(&svg, max_side).expect("rasterize");
    image::DynamicImage::ImageRgba8(img)
        .to_rgb8()
        .save(&args[2])
        .expect("save png");
    println!("wrote {}", args[2]);
}
