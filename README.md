# im2vec

Convert PNG, JPG, and WebP images to SVG.

I made it for logos first. Drop in a flat logo and you get back a small SVG with clean edges. It also handles illustrations, photos, and black and white line art.

It runs on [vtracer](https://github.com/visioncortex/VTracer) in pure Rust.

## Quickstart

You need Rust and Cargo installed.

```bash
cargo run -p im2vec-cli -- input.png -o output.svg --preset logo
```

Pick the preset that matches your image. Use logo for icons and flat marks, illustration for cartoons and flat art with more colors, photo for pictures with gradients and shadows, mono for black and white line art.

## Web preview

If you want to see the result before you save, run the local preview.

```bash
cargo run -p im2vec-web
```

Open http://127.0.0.1:5173, drop in an image, and it converts right away. Move the sliders to tune it, compare side by side, and download the SVG when you like it.

