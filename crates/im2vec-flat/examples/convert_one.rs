//! One-off conversion: image file -> SVG file (for ad-hoc garment tests).
//! Usage: cargo run -q -p im2vec-flat --example convert_one -- <input.png> <out.svg>
use std::fs;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input = PathBuf::from(args.get(1).expect("usage: convert_one <input> <out.svg>"));
    let out = PathBuf::from(args.get(2).expect("usage: convert_one <input> <out.svg>"));
    let bytes = fs::read(&input).expect("read input");
    let opts = im2vec_flat::FlatOptions::default();
    let flat = im2vec_flat::convert_flat_bytes(&bytes, &opts).expect("convert");
    fs::write(&out, &flat.svg).expect("write svg");
    eprintln!(
        "wrote {} ({}x{}, {} paths, {} bytes)",
        out.display(),
        flat.width,
        flat.height,
        flat.path_count,
        flat.svg_bytes
    );
}
