# 1:1 Checklist — blazer tech-pack parity

**What "1:1" means:** our SVG output for `samples/blazer/input.png` is
visually indistinguishable from `samples/blazer/target.png` — the same
silhouette, the same construction details, the same line conventions
(solid = seam/silhouette, dashed = topstitching), the same clean symbols.
The reference is a rule-based redrawing, so 1:1 is achievable without ML:
every line in the target *means* something.

## How to measure

```sh
# quantitative metrics + visual diffs -> docs/eval/
cargo run -p im2vec-eval -- \
  --input samples/blazer/input.png \
  --target samples/blazer/target.png \
  --out-dir docs/eval --name baseline

# interactive visual diff (side-by-side + onion-skin overlay)
cargo run -p im2vec-web   # then open http://127.0.0.1:5173/eval
```

All metrics align our output to the reference with a similarity transform
(uniform scale + translation fitted to the two garment bounding boxes)
before comparing — the reference is a redrawing with slightly different
global proportions, so raw canvas position is not compared. See
`crates/im2vec-eval/src/lib.rs` for the exact protocol.

## Quantitative metrics (auto)

| Metric | Baseline 2026-09-18 | Phase target | Notes |
|---|---|---|---|
| silhouette IoU | 0.857 | ≥ 0.97 front view | ✗ mask vs. reference silhouette, bbox-aligned |
| chamfer (symmetric) | 8.5 px | ≤ 3.0 px @1024 | ✗ mean nearest-neighbour ink distance |
| chamfer ours→target | 9.4 px | — | high = we draw lines the ref lacks (noise) |
| chamfer target→ours | 7.6 px | — | high = we miss ref lines (missing details) |
| ink ratio ours/target | 1.84 | 0.8 – 1.2 | ✗ >>1 = texture/wrinkle noise; <<1 = missing detail |
| path count | 212 | 30 – 60 | ✗ ref-equivalent is a few dozen meaningful paths |
| buttons: ours / ref / matched | 5 / 15 / 3 | 18 / 18 / 18 | ✗ 6 front (2×3) + 4 side cuff + 8 back cuff (4 per sleeve); detector recall ~15/18 on ref |
| button position error | 10.0 px | ≤ 5 px @1024 | ✗ mean over matched pairs |

## Qualitative checklist (human-rated per view)

Mark: `☐` not checked · `☑` pass · `☒` fail

| # | Criterion | Front | Side | Back | Notes |
|---|---|---|---|---|---|
| 1 | Silhouette matches: lapel peaks sharp, collar, hem, cuffs | ☐ | ☐ | ☐ | no rounded tips, no merged views |
| 2 | 6 front buttons in 2×3 grid, uniform circles, 4-hole detail | ☐ | — | — | |
| 3 | 2 flap pockets: clean rounded rect + dashed inset stitching | ☐ | — | — | |
| 4 | Princess seams present, solid, smooth | ☐ | — | — | |
| 5 | 4 cuff buttons per sleeve, uniform | ☐ | ☐ | ☐ | |
| 6 | Hem stitch line (dashed) | ☐ | ☐ | ☐ | |
| 7 | Collar stitch line (dashed) | ☐ | — | ☐ | |
| 8 | Line semantics: solid = seam/silhouette, dashed = topstitching | ☐ | ☐ | ☐ | no uniform-stroke look |
| 9 | Zero denim texture / wrinkle noise lines | ☐ | ☐ | ☐ | |
| 10 | Front/back perfectly symmetric | ☐ | — | ☐ | |
| 11 | Side view untouched (not mirrored, no phantom sleeve) | — | ☐ | — | |
| 12 | Back label reproduced (dashed rect + diagonal hatch) | — | — | ☐ | or explicit decision to omit |
| 13 | Blind test: unidentifiable vs. reference | ☐ | ☐ | ☐ | final sign-off |

## Baseline record (2026-09-18)

- Pipeline: default flat preset (`detail_strength 0.6`), no pipeline code changes
  (only additive `im2vec-flat` mask API for measurement).
- Artifacts: `docs/eval/baseline.svg`, `baseline-report.json`, `baseline-diff.png`, `baseline-onion.png`.
- Reading the baseline: the photo is a **denim** blazer, so wrinkle/texture noise
  dominates (ink ratio 1.84, 212 paths vs ~30–60). The silhouette is close
  (IoU 0.857) but linework is noisy and most buttons are missed (3/18 matched).
  Phases 2–5 address silhouette sharpness, noise-vs-seam classification,
  button/pocket symbols, and stroke hygiene, in that order.
