# im2vec — image to vector SVG (Rust)

Logo-first raster → vector tracer. Any PNG/JPG/WebP in, high-quality SVG out.
Built on [`vtracer`](https://github.com/visioncortex/VTracer) (MIT, pure Rust) — we wrap it with logo-tuned presets, a CLI, and a dev-server preview UI. We fork only if we outgrow it.

## Layout

- `crates/im2vec-core` — decode + `ConvertOptions` (logo/illustration/photo/mono) + `convert_bytes`
- `crates/im2vec-cli` — `im2vec input.png -o output.svg --preset logo`
- `crates/im2vec-web` — dev server at `http://127.0.0.1:5173`: paste / drag-drop / browse an image, it auto-converts; tune sliders, side-by-side compare, download the SVG only if you like it
- `samples/` — drop test logos here (not committed if large)

## Quickstart

```bash
cargo test -p im2vec-core
cargo run -p im2vec-cli -- samples/logo.png -o /tmp/out.svg --preset logo
PORT=5173 cargo run -p im2vec-web
# open http://127.0.0.1:5173
```

## Roadmap slices

1. ✅ Slice 0: workspace + core + CLI + web shell
2. ✅ Slice 1: logo pipeline tuning (speckle, simplify, palette snapping)
3. ✅ Slice 2/3: illustration + photo mode (watershed segmentation preserves gradients/shadows)
4. ✅ Web UX: paste / drag-drop auto-convert, opt-in SVG download
5. Slice 4: quality harness (render-back diff MSE/SSIM, A/B comparator)
6. Slice 5: perf (rayon, Session caching for slider interactivity) + export PDF/EPS
