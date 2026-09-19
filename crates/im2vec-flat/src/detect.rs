//! #29 — direct photo edge detection for template landmarks.
//!
//! Detection *proposes*, #27 search *refines*: each detector estimates a
//! landmark position from the input photo's XDoG chains and reports a
//! confidence. Below [`DETECT_CONFIDENCE_MIN`] the caller falls back to the
//! button-relative heuristic, so weak photo evidence can never regress the
//! output. Symmetry about the center front is a first-class signal: pocket
//! flaps and lapel edges come in mirrored pairs, and a lone gorge seam is
//! mirrored to the other side.

use std::f32::consts::PI;

/// Minimum confidence for a detection to override the heuristic.
pub const DETECT_CONFIDENCE_MIN: f32 = 0.5;

/// Lightweight geometric features for one chain.
#[derive(Clone, Debug)]
struct ChainFeat {
    idx: usize,
    arc_len: f32,
    cx: f32,
    cy: f32,
    x0: f32,
    x1: f32,
    y0: f32,
    y1: f32,
    angle_deg: f32,
    straightness: f32,
}

fn chain_features(chains: &[Vec<(f32, f32)>]) -> Vec<ChainFeat> {
    chains
        .iter()
        .enumerate()
        .filter_map(|(idx, c)| {
            if c.len() < 2 {
                return None;
            }
            let mut arc = 0.0f32;
            let (mut x0, mut x1) = (f32::INFINITY, f32::NEG_INFINITY);
            let (mut y0, mut y1) = (f32::INFINITY, f32::NEG_INFINITY);
            let (mut sx, mut sy) = (0.0f32, 0.0f32);
            for w in c.windows(2) {
                let ((ax, ay), (bx, by)) = (w[0], w[1]);
                arc += (bx - ax).hypot(by - ay);
                for (px, py) in [w[0], w[1]] {
                    x0 = x0.min(px);
                    x1 = x1.max(px);
                    y0 = y0.min(py);
                    y1 = y1.max(py);
                    sx += px;
                    sy += py;
                }
            }
            // (windows double-count interior points; correct the centroid.)
            let n = c.len() as f32;
            sx -= c[1..c.len() - 1].iter().map(|p| p.0).sum::<f32>();
            sy -= c[1..c.len() - 1].iter().map(|p| p.1).sum::<f32>();
            let (fx, fy) = (c[0], c[c.len() - 1]);
            let end_len = (fx.0 - fy.0).hypot(fx.1 - fy.1);
            let angle_deg = (fy.1 - fx.1).atan2(fy.0 - fx.0) * 180.0 / PI;
            Some(ChainFeat {
                idx,
                arc_len: arc,
                cx: sx / n,
                cy: sy / n,
                x0,
                x1,
                y0,
                y1,
                angle_deg,
                straightness: if arc > 1e-6 { end_len / arc } else { 0.0 },
            })
        })
        .collect()
}

/// Deviation of a chain's end-to-end direction from horizontal, in degrees.
fn horiz_dev(f: &ChainFeat) -> f32 {
    let a = f.angle_deg.abs();
    a.min(180.0 - a)
}

/// Deviation from vertical, in degrees.
fn vert_dev(f: &ChainFeat) -> f32 {
    (f.angle_deg.abs() - 90.0).abs()
}

/// A symmetric left/right chain pair.
#[derive(Clone, Copy, Debug)]
pub struct DetectedPair {
    pub left_idx: usize,
    pub right_idx: usize,
    pub confidence: f32,
}

fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Tuning for [`find_horizontal_pair`].
struct HPairConfig {
    len_min: f32,
    len_max: f32,
    angle_tol: f32,
    straight_min: f32,
    zone_hw: f32,
    x_inner: f32,
    x_outer: f32,
    y_sym_tol: f32,
    mirror_tol: f32,
}

/// Generic symmetric horizontal pair finder. Scores candidates on symmetry,
/// zone proximity to `y_expected`, and straightness.
fn find_horizontal_pair(
    feats: &[ChainFeat],
    cx: f32,
    y_expected: f32,
    cfg: &HPairConfig,
) -> Option<(DetectedPair, f32)> {
    let cands: Vec<&ChainFeat> = feats
        .iter()
        .filter(|f| {
            f.arc_len >= cfg.len_min
                && f.arc_len <= cfg.len_max
                && horiz_dev(f) <= cfg.angle_tol
                && f.straightness >= cfg.straight_min
                && (f.cy - y_expected).abs() <= cfg.zone_hw
                && (f.cx - cx).abs() >= cfg.x_inner
                && (f.cx - cx).abs() <= cfg.x_outer
        })
        .collect();
    let (left, right): (Vec<&ChainFeat>, Vec<&ChainFeat>) = cands.iter().partition(|f| f.cx < cx);
    let mut best: Option<(DetectedPair, f32, f32)> = None; // (pair, y, score)
    for l in &left {
        for r in &right {
            if (l.cy - r.cy).abs() > cfg.y_sym_tol {
                continue;
            }
            let mirror_mismatch = ((cx - l.cx) - (r.cx - cx)).abs();
            if mirror_mismatch > cfg.mirror_tol {
                continue;
            }
            let y = (l.cy + r.cy) * 0.5;
            let sym = 1.0 - mirror_mismatch / cfg.mirror_tol;
            let zone = 1.0 - (y - y_expected).abs() / cfg.zone_hw;
            let straight = (l.straightness + r.straightness) * 0.5;
            let score = 0.45 * sym + 0.30 * clamp01(zone) + 0.25 * straight;
            if best.is_none_or(|(_, _, s)| score > s) {
                best = Some((
                    DetectedPair {
                        left_idx: l.idx,
                        right_idx: r.idx,
                        confidence: score,
                    },
                    y,
                    score,
                ));
            }
        }
    }
    best.map(|(p, y, _)| (p, y))
}

/// Gorge (notch) y from horizontal seam candidates near the expected row.
///
/// Built on the garment-agnostic [`detect_horizontal_seams`] ranker (length
/// coverage, extent symmetry about the center front, straightness); a zone
/// term prefers candidates near `y_expected`. A symmetric pair at one height
/// naturally clusters into a single strong candidate, which is what the old
/// pair-then-mirror logic selected. Returns `(y, score)`.
pub fn detect_gorge_y(
    chains: &[Vec<(f32, f32)>],
    cx: f32,
    w: f32,
    y_expected: f32,
) -> Option<(f32, f32)> {
    let zone_hw = 45.0; // the notch zone is tight
    let view = ViewContext::new(
        cx - 0.5 * w,
        y_expected - zone_hw,
        cx + 0.5 * w,
        y_expected + zone_hw,
    );
    detect_horizontal_seams(chains, &view, 30.0)
        .into_iter()
        .map(|c| {
            let zone = 1.0 - (c.pos - y_expected).abs() / zone_hw;
            let score = 0.5 * zone.clamp(0.0, 1.0) + 0.5 * c.confidence;
            (c.pos, score)
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .filter(|&(_, s)| s >= DETECT_CONFIDENCE_MIN)
}

/// Minimum net convergence (px) for a lapel pair: top separation minus bottom
/// separation must be this positive. Blazer lapels measure +27px; jeans
/// pocket openings -37px; parallel fly edges ~0px.
const MIN_LAPEL_CONVERGENCE: f32 = 10.0;

/// Net horizontal convergence of a left/right chain pair going downward:
/// positive when the pair narrows (lapel-like), negative when it splays
/// (pocket-opening-like), near zero when parallel (fly/placket-like).
/// Returns `None` when either chain has too few points to measure.
fn pair_convergence(left: &[(f32, f32)], right: &[(f32, f32)]) -> Option<f32> {
    /// Mean x of points in the top / bottom quarter of a chain's y-range.
    fn ends(chain: &[(f32, f32)]) -> Option<(f32, f32)> {
        if chain.len() < 4 {
            return None;
        }
        let (mut y0, mut y1) = (f32::INFINITY, f32::NEG_INFINITY);
        for &(_, y) in chain {
            y0 = y0.min(y);
            y1 = y1.max(y);
        }
        let band = 0.25 * (y1 - y0);
        if band <= 0.0 {
            return None;
        }
        let (mut tx, mut bx, mut nt, mut nb) = (0.0f32, 0.0f32, 0u32, 0u32);
        for &(x, y) in chain {
            if y <= y0 + band {
                tx += x;
                nt += 1;
            }
            if y >= y1 - band {
                bx += x;
                nb += 1;
            }
        }
        if nt == 0 || nb == 0 {
            return None;
        }
        Some((tx / nt as f32, bx / nb as f32))
    }
    let (lt, lb) = ends(left)?;
    let (rt, rb) = ends(right)?;
    Some((rt - lt) - (rb - lb))
}

/// Lapel edge pair: long, near-vertical chains in the upper front, symmetric
/// about the center front. Returns the pair with confidence; the caller logs
/// it as a structural signal (placement itself is optimized by #27 search).
///
/// A lapel pair must CONVERGE going down (peak wide at top, narrowing to the
/// break). This rejects lookalikes with the same symmetry signature: pocket
/// openings splay outward going down (negative convergence) and fly/placket
/// edges run parallel (near-zero). Measured on the blazer (pair 30/26:
/// +27px) vs jeans pocket openings (pair 20/22: -37px).
pub fn detect_lapel_pair(
    chains: &[Vec<(f32, f32)>],
    cx: f32,
    w: f32,
    y0: f32,
    h: f32,
) -> Option<DetectedPair> {
    let feats = chain_features(chains);
    let cands: Vec<&ChainFeat> = feats
        .iter()
        .filter(|f| {
            f.arc_len >= 90.0
                && vert_dev(f) <= 25.0
                && f.straightness >= 0.75
                && f.cy >= y0
                && f.cy <= y0 + 0.55 * h
                && (f.y1 - f.y0) >= 100.0
                && (f.cx - cx).abs() >= 0.05 * w
                && (f.cx - cx).abs() <= 0.45 * w
        })
        .collect();
    let (left, right): (Vec<&ChainFeat>, Vec<&ChainFeat>) = cands.iter().partition(|f| f.cx < cx);
    let mut best: Option<(DetectedPair, f32)> = None;
    for l in &left {
        for r in &right {
            let mirror_mismatch = ((cx - l.cx) - (r.cx - cx)).abs();
            if mirror_mismatch > 50.0 {
                continue;
            }
            let overlap = (l.y1.min(r.y1) - l.y0.max(r.y0)).max(0.0);
            if overlap < 60.0 {
                continue;
            }
            // Lapels converge going down; pocket openings splay, flies run
            // parallel. Reject pairs that don't narrow (see pair_convergence).
            let conv = pair_convergence(&chains[l.idx], &chains[r.idx]).unwrap_or(0.0);
            if conv < MIN_LAPEL_CONVERGENCE {
                continue;
            }
            let sym = 1.0 - mirror_mismatch / 50.0;
            let ov = (overlap / 150.0).min(1.0);
            let straight = (l.straightness + r.straightness) * 0.5;
            let score = 0.4 * sym + 0.3 * ov + 0.3 * straight;
            if best.is_none_or(|(_, s)| score > s) {
                best = Some((
                    DetectedPair {
                        left_idx: l.idx,
                        right_idx: r.idx,
                        confidence: score,
                    },
                    score,
                ));
            }
        }
    }
    best.filter(|(_, s)| *s >= DETECT_CONFIDENCE_MIN)
        .map(|(p, _)| p)
}

// ---------------------------------------------------------------------------
// Garment-agnostic detection layer.
//
// The detectors above are thin blazer-specific configs over a symmetric-pair
// finder. The layer below works from photo evidence alone — no blazer
// proportions, no target-measured constants — so a new garment (shirt,
// dress, trousers) gets the same primitives: a view frame, a symmetry axis
// from mask moments, ranked seam candidates, and closure columns from
// buttons. Garment-specific templates then consume these primitives instead
// of hardcoded fractions.
// ---------------------------------------------------------------------------

/// A garment view's frame in output pixels, plus its symmetry axis.
///
/// `axis_x` starts at the bbox center; [`refine_axis_from_mask`] replaces it
/// with the mask-moment centroid, which needs no buttons and works for any
/// garment.
#[derive(Clone, Copy, Debug)]
pub struct ViewContext {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    pub axis_x: f32,
}

impl ViewContext {
    pub fn new(x0: f32, y0: f32, x1: f32, y1: f32) -> Self {
        Self {
            x0,
            y0,
            x1,
            y1,
            axis_x: (x0 + x1) * 0.5,
        }
    }

    pub fn w(&self) -> f32 {
        self.x1 - self.x0
    }

    /// View height. (Foundation for future garment templates; allow(dead_code).)
    #[allow(dead_code)]
    pub fn h(&self) -> f32 {
        self.y1 - self.y0
    }
}

/// Symmetry axis from mask moments: the centroid-x of foreground pixels
/// inside the view bbox. Unlike the button-derived center front, this needs
/// no closure hardware — dresses, tees, and trousers get an axis too.
/// Falls back to the bbox center on empty masks.
pub fn refine_axis_from_mask(mask: &[bool], w: usize, h: usize, view: &ViewContext) -> f32 {
    if w == 0 || h == 0 || mask.len() < w * h {
        return view.axis_x;
    }
    let x0 = view.x0.max(0.0) as usize;
    let x1 = (view.x1.min(w as f32)) as usize;
    let y0 = view.y0.max(0.0) as usize;
    let y1 = (view.y1.min(h as f32)) as usize;
    let (mut sum_x, mut n) = (0u64, 0u64);
    for y in y0..y1 {
        for x in x0..x1 {
            if mask[y * w + x] {
                sum_x += x as u64;
                n += 1;
            }
        }
    }
    if n == 0 {
        view.axis_x
    } else {
        sum_x as f32 / n as f32
    }
}

/// One ranked seam candidate: `pos` is the seam's y (horizontal) or x
/// (vertical); `lo..hi` is its extent along the seam direction.
/// `lo`/`hi`/`total_len` are foundation API for future template consumers
/// (seam extent rendering, photo-measured pocket widths).
#[derive(Clone, Copy, Debug)]
pub struct SeamCandidate {
    pub pos: f32,
    #[allow(dead_code)]
    pub lo: f32,
    #[allow(dead_code)]
    pub hi: f32,
    pub confidence: f32,
    #[allow(dead_code)]
    pub total_len: f32,
}

/// Tolerance (px) for clustering chains into one seam candidate.
const SEAM_CLUSTER_TOL_PX: f32 = 10.0;
/// Max orientation deviation (deg) from the seam direction.
const SEAM_ANGLE_TOL_DEG: f32 = 20.0;

/// Garment-agnostic seam candidates: near-horizontal (or vertical) chains
/// clustered by position and ranked by length coverage, extent symmetry
/// about the view axis, and straightness. Returns all candidates sorted by
/// descending confidence — the caller picks by zone, not the detector.
fn cluster_seam_candidates(
    chains: &[Vec<(f32, f32)>],
    view: &ViewContext,
    horizontal: bool,
    min_len: f32,
) -> Vec<SeamCandidate> {
    rank_seam_candidates(&chain_features(chains), view, horizontal, min_len)
}

/// Seam ranking over precomputed chain features: filter to the view, then
/// cluster by position along the seam normal. Split out so view scoring can
/// rank several sub-regions without recomputing features.
fn rank_seam_candidates(
    feats: &[ChainFeat],
    view: &ViewContext,
    horizontal: bool,
    min_len: f32,
) -> Vec<SeamCandidate> {
    let span = if horizontal { view.w() } else { view.h() };
    if span <= 0.0 {
        return Vec::new();
    }
    let margin = 0.05 * span;
    let mut items: Vec<&ChainFeat> = feats
        .iter()
        .filter(|f| {
            let orient_ok = if horizontal {
                horiz_dev(f) <= SEAM_ANGLE_TOL_DEG
            } else {
                vert_dev(f) <= SEAM_ANGLE_TOL_DEG
            };
            orient_ok
                && f.arc_len >= min_len
                && f.cx >= view.x0 - margin
                && f.cx <= view.x1 + margin
                && f.cy >= view.y0 - margin
                && f.cy <= view.y1 + margin
        })
        .collect();
    // Greedy clustering by position along the seam normal.
    items.sort_by(|a, b| {
        let (pa, pb) = if horizontal {
            (a.cy, b.cy)
        } else {
            (a.cx, b.cx)
        };
        pa.partial_cmp(&pb).unwrap()
    });
    let mut clusters: Vec<Vec<&ChainFeat>> = Vec::new();
    for f in items {
        let p = if horizontal { f.cy } else { f.cx };
        match clusters.last_mut() {
            Some(c) => {
                let q = if horizontal { c[0].cy } else { c[0].cx };
                if (p - q).abs() <= SEAM_CLUSTER_TOL_PX {
                    c.push(f);
                } else {
                    clusters.push(vec![f]);
                }
            }
            None => clusters.push(vec![f]),
        }
    }
    let mut out: Vec<SeamCandidate> = clusters
        .into_iter()
        .map(|c| {
            let total_len: f32 = c.iter().map(|f| f.arc_len).sum();
            let pos: f32 = c.iter().map(|f| f.cy * f.arc_len).sum::<f32>() / total_len.max(1e-6);
            let pos = if horizontal {
                pos
            } else {
                c.iter().map(|f| f.cx * f.arc_len).sum::<f32>() / total_len.max(1e-6)
            };
            let (lo, hi) = c
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), f| {
                    if horizontal {
                        (lo.min(f.x0), hi.max(f.x1))
                    } else {
                        (lo.min(f.y0), hi.max(f.y1))
                    }
                });
            let mid = (lo + hi) * 0.5;
            let axis = if horizontal {
                view.axis_x
            } else {
                (view.y0 + view.y1) * 0.5
            };
            let sym = 1.0 - ((mid - axis).abs() / (0.5 * span)).clamp(0.0, 1.0);
            let straight: f32 =
                c.iter().map(|f| f.straightness * f.arc_len).sum::<f32>() / total_len.max(1e-6);
            let coverage = (total_len / span).min(1.0);
            let confidence = 0.5 * coverage + 0.3 * sym + 0.2 * straight.clamp(0.0, 1.0);
            SeamCandidate {
                pos,
                lo,
                hi,
                confidence: confidence.clamp(0.0, 1.0),
                total_len,
            }
        })
        .collect();
    out.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
    out
}

/// Horizontal seam candidates across the whole view (waistbands, pocket
/// flaps, hems, yokes): ranked, not zone-gated.
pub fn detect_horizontal_seams(
    chains: &[Vec<(f32, f32)>],
    view: &ViewContext,
    min_len: f32,
) -> Vec<SeamCandidate> {
    cluster_seam_candidates(chains, view, true, min_len)
}

/// Vertical seam candidates (plackets, side seams, pleats, fly fronts).
/// Foundation for future garment templates; the current pipeline's seam
/// needs are horizontal, so this is exercised by unit tests for now.
#[allow(dead_code)]
pub fn detect_vertical_seams(
    chains: &[Vec<(f32, f32)>],
    view: &ViewContext,
    min_len: f32,
) -> Vec<SeamCandidate> {
    cluster_seam_candidates(chains, view, false, min_len)
}

/// Foreground symmetry about x=`axis` within the bbox: 1.0 = perfectly
/// mirrored, 0.0 = all foreground on one side (or empty). Compares
/// foreground pixel counts left vs right of the axis, row by row, weighted
/// by row foreground count so sparse rows do not dominate.
fn mask_symmetry(mask: &[bool], w: usize, h: usize, bbox: (f32, f32, f32, f32), axis: f32) -> f32 {
    let (x0, y0, x1, y1) = bbox;
    if w == 0 || h == 0 || mask.len() < w * h {
        return 0.0;
    }
    let y_lo = (y0.max(0.0) as usize).min(h);
    let y_hi = (y1.min(h as f32) as usize).min(h);
    let x_lo = (x0.max(0.0) as usize).min(w);
    let x_hi = (x1.min(w as f32) as usize).min(w);
    if y_lo >= y_hi || x_lo >= x_hi {
        return 0.0;
    }
    let xai = (axis.clamp(x0, x1) as usize).clamp(x_lo, x_hi);
    let (mut sym_sum, mut wsum) = (0.0f32, 0.0f32);
    for y in y_lo..y_hi {
        let row = y * w;
        let mut left = 0u32;
        let mut right = 0u32;
        for x in x_lo..xai {
            if mask[row + x] {
                left += 1;
            }
        }
        for x in xai..x_hi {
            if mask[row + x] {
                right += 1;
            }
        }
        let tot = left + right;
        if tot > 0 {
            let s = 1.0 - (left as f32 - right as f32).abs() / tot as f32;
            sym_sum += s * tot as f32;
            wsum += tot as f32;
        }
    }
    if wsum <= 0.0 {
        0.0
    } else {
        sym_sum / wsum
    }
}

/// Identify (front, back) view component indices from garment-agnostic
/// evidence. Buttons are strong evidence when present but not required:
/// a buttonless garment (tee, dress) still gets a front view from mask
/// symmetry, neckline chain evidence, and relative size. `comps` are
/// (x0, y0, x1, y1, area) with index 0 unused; either return may be None.
pub fn identify_views(
    comps: &[(f32, f32, f32, f32, usize)],
    button_pts: &[(f32, f32)],
    chains: &[Vec<(f32, f32)>],
    mask: &[bool],
    w: usize,
    h: usize,
) -> (Option<usize>, Option<usize>) {
    let feats = chain_features(chains);
    let max_area = comps.iter().skip(1).map(|c| c.4).max().unwrap_or(1).max(1) as f32;
    let mut best: Option<(usize, f32)> = None;
    for (i, &(x0, y0, x1, y1, area)) in comps.iter().enumerate().skip(1) {
        let cw = x1 - x0;
        let ch = y1 - y0;
        if cw < 50.0 || ch < 100.0 {
            continue;
        }
        let n_btn = button_pts
            .iter()
            .filter(|&&(bx, by)| bx >= x0 && bx <= x1 && by >= y0 && by <= y1)
            .count();
        let button_score = (n_btn.min(6) as f32) / 6.0;
        let sym = mask_symmetry(mask, w, h, (x0, y0, x1, y1), (x0 + x1) * 0.5);
        // Neckline evidence: strongest horizontal seam in the upper 30%.
        let upper = ViewContext::new(x0, y0, x1, y0 + 0.30 * ch);
        let neck = rank_seam_candidates(&feats, &upper, true, 40.0)
            .into_iter()
            .next()
            .map(|c| c.confidence)
            .unwrap_or(0.0);
        let size = area as f32 / max_area;
        let score = 0.40 * button_score + 0.25 * sym + 0.20 * neck + 0.15 * size;
        if best.is_none_or(|(_, s)| score > s) {
            best = Some((i, score));
        }
    }
    let front_idx = best.filter(|&(_, s)| s >= 0.10).map(|(i, _)| i);
    // Back = largest remaining component with aspect like front (not a sleeve).
    let mut back_idx = None;
    if let Some(fi) = front_idx {
        let fw = comps[fi].2 - comps[fi].0;
        let mut best_area = 0usize;
        for (i, &(x0, _, x1, _, area)) in comps.iter().enumerate().skip(1) {
            if i == fi {
                continue;
            }
            if x1 - x0 < fw * 0.6 {
                continue; // sleeve/detail view
            }
            if area > best_area {
                best_area = area;
                back_idx = Some(i);
            }
        }
    }
    (front_idx, back_idx)
}

/// A detected neckline: seam row, photo-measured extent, bow depth, and
/// confidence. Everything positional comes from the XDoG chains.
#[derive(Clone, Copy, Debug)]
pub struct Neckline {
    pub y: f32,
    pub x0: f32,
    pub x1: f32,
    /// Bow depth below `y` in px (>= 0), measured from the chains.
    pub depth: f32,
    pub confidence: f32,
}

/// Neckline (collar seam) for buttonless fronts: the strongest horizontal
/// seam candidate in the view's upper region. Must be narrower than the view
/// (a neckline spans a fraction of the chest; a seam spanning the full width
/// is a waistband or yoke). Rejects straight yoke-like seams via the bow
/// gate — a real neckline always curves downward.
/// Minimum neckline width as a fraction of view width (rejects noise).
const MIN_NECKLINE_WIDTH_FRAC: f32 = 0.12;
/// Maximum neckline width as a fraction of the garment's LOCAL width at the
/// seam's height: a seam spanning the body there is a waistband/yoke, not a
/// neckline (jeans waistband measured 0.94 of local width; a crew neckline
/// is ~0.35 of the local chest width).
const MAX_NECKLINE_WIDTH_FRAC: f32 = 0.75;

/// Garment width (mask x-extent) at height `y`, restricted to the view's
/// x-range. Returns 0.0 when the mask row is empty.
fn mask_width_at_y(mask: &[bool], img_w: usize, img_h: usize, y: f32, vx0: f32, vx1: f32) -> f32 {
    if mask.len() != img_w * img_h || img_w == 0 || img_h == 0 {
        return 0.0;
    }
    let row = (y.round().clamp(0.0, img_h as f32 - 1.0)) as usize;
    let x_lo = (vx0.max(0.0) as usize).min(img_w);
    let x_hi = (vx1.min(img_w as f32) as usize).min(img_w);
    let (mut lx0, mut lx1) = (img_w, 0usize);
    for x in x_lo..x_hi {
        if mask[row * img_w + x] {
            lx0 = lx0.min(x);
            lx1 = lx1.max(x);
        }
    }
    if lx1 > lx0 {
        (lx1 - lx0) as f32
    } else {
        0.0
    }
}

/// Minimum gap (px) between mask intervals to count as a real separation
/// (ignores speckle noise when looking for legs).
const MIN_LEG_GAP: usize = 15;

/// Does the view's mask show separated legs (jeans/trousers)? Samples rows
/// in the lower half of the view; a bottom garment has ≥2 disjoint mask
/// intervals (left leg, right leg) on ≥2 of the sampled rows. Used to
/// suppress neckline detection on bottoms — a waistband is not a neckline.
pub fn has_separated_legs(mask: &[bool], img_w: usize, img_h: usize, view: &ViewContext) -> bool {
    if mask.len() != img_w * img_h || img_w == 0 || img_h == 0 {
        return false;
    }
    let mut rows_with_gap = 0;
    for &fy in &[0.60, 0.75, 0.90] {
        let y = view.y0 + fy * view.h();
        let row = (y.round().clamp(0.0, img_h as f32 - 1.0)) as usize;
        let x_lo = (view.x0.max(0.0) as usize).min(img_w);
        let x_hi = (view.x1.min(img_w as f32) as usize).min(img_w);
        // Collect mask intervals on this row, then merge ones separated by
        // less than MIN_LEG_GAP (speckle noise, not a real leg separation).
        let mut intervals: Vec<(usize, usize)> = Vec::new();
        let mut start: Option<usize> = None;
        for x in x_lo..x_hi {
            let m = mask[row * img_w + x];
            match (m, start) {
                (true, None) => start = Some(x),
                (false, Some(s)) => {
                    intervals.push((s, x));
                    start = None;
                }
                _ => {}
            }
        }
        if let Some(s) = start {
            intervals.push((s, x_hi));
        }
        let mut merged = 0;
        let mut prev_end = 0usize;
        for (i, &(s, e)) in intervals.iter().enumerate() {
            if i == 0 || s - prev_end >= MIN_LEG_GAP {
                merged += 1;
            }
            prev_end = e;
        }
        if merged >= 2 {
            rows_with_gap += 1;
        }
    }
    rows_with_gap >= 2
}
pub fn detect_neckline(
    chains: &[Vec<(f32, f32)>],
    view: &ViewContext,
    mask: &[bool],
    img_w: usize,
    img_h: usize,
) -> Option<Neckline> {
    let upper = ViewContext::new(view.x0, view.y0, view.x1, view.y0 + 0.30 * view.h());
    let best = detect_horizontal_seams(chains, &upper, 40.0)
        .into_iter()
        .next()?;
    let x0 = best.lo.max(view.x0);
    let x1 = best.hi.min(view.x1);
    if x1 - x0 < MIN_NECKLINE_WIDTH_FRAC * view.w() {
        return None; // too narrow to be a neckline
    }
    // Local body width at the seam's height (falls back to the view width
    // when the mask row is empty).
    let local_w = mask_width_at_y(mask, img_w, img_h, best.pos, view.x0, view.x1);
    let w_ref = if local_w > 0.5 * view.w() {
        local_w.min(view.w())
    } else {
        view.w()
    };
    if x1 - x0 > MAX_NECKLINE_WIDTH_FRAC * w_ref {
        return None; // spans the body: waistband or yoke, not a neckline
    }
    // Bow depth: deepest chain point within the extent near the seam row.
    let mut depth = 0.0f32;
    for c in chains {
        let mut cmax = f32::NEG_INFINITY;
        for &(x, y) in c {
            if x >= x0 && x <= x1 && (y - best.pos).abs() <= 25.0 {
                cmax = cmax.max(y);
            }
        }
        if cmax.is_finite() {
            depth = depth.max((cmax - best.pos).max(0.0));
        }
    }
    if depth < 2.0 {
        return None; // straight seam, not a neckline
    }
    Some(Neckline {
        y: best.pos,
        x0,
        x1,
        depth: depth.min(0.25 * view.h()),
        confidence: best.confidence,
    })
}

/// One column of closure buttons (a placket column): x position plus the
/// button row ys, sorted top to bottom.
#[derive(Clone, Debug)]
pub struct ButtonColumn {
    pub x: f32,
    pub ys: Vec<f32>,
}

/// Cluster button points (already filtered to the view) into vertical
/// columns. Works for 1-column shirts, 2-column double-breasted fronts, and
/// reports ambiguity (>2 columns) instead of guessing.
pub fn detect_button_columns(points: &[(f32, f32)], view: &ViewContext) -> Vec<ButtonColumn> {
    let x_tol = (0.05 * view.w()).max(16.0);
    let mut pts: Vec<(f32, f32)> = points.to_vec();
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut cols: Vec<ButtonColumn> = Vec::new();
    for (x, y) in pts {
        match cols.last_mut() {
            Some(c) if (x - c.x).abs() <= x_tol => {
                let n = c.ys.len() as f32;
                c.x = (c.x * n + x) / (n + 1.0);
                c.ys.push(y);
            }
            _ => cols.push(ButtonColumn { x, ys: vec![y] }),
        }
    }
    for c in cols.iter_mut() {
        c.ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }
    cols
}

/// Center front from closure columns: one column *is* the placket line;
/// two columns straddle it (double-breasted). Anything else is ambiguous —
/// the caller falls back to the mask axis rather than guessing.
pub fn closure_center(cols: &[ButtonColumn]) -> Option<f32> {
    match cols {
        [] => None,
        [c] => Some(c.x),
        [a, b] => Some((a.x + b.x) * 0.5),
        _ => None,
    }
}

/// Pocket flap detection with photo-measured x centers: y plus the left and
/// right flap centers from the detected chains' extents. Pocket placement
/// traces the photo instead of assuming fixed ±0.25w fractions.
#[derive(Clone, Copy, Debug)]
pub struct PocketFlaps {
    pub y: f32,
    pub left_cx: f32,
    pub right_cx: f32,
    pub confidence: f32,
}

pub fn detect_pocket_flaps(
    chains: &[Vec<(f32, f32)>],
    cx: f32,
    w: f32,
    y_expected: f32,
) -> Option<PocketFlaps> {
    let feats = chain_features(chains);
    let cfg = HPairConfig {
        len_min: 45.0,
        len_max: 220.0,
        angle_tol: 15.0,
        straight_min: 0.65,
        zone_hw: 70.0,
        x_inner: 0.05 * w,
        x_outer: 0.45 * w,
        y_sym_tol: 12.0,
        mirror_tol: 30.0,
    };
    let (pair, y) = find_horizontal_pair(&feats, cx, y_expected, &cfg)?;
    if pair.confidence < DETECT_CONFIDENCE_MIN {
        return None;
    }
    // NOTE: `feats` may skip degenerate chains, so look up by stored idx,
    // not by position.
    let left = feats.iter().find(|f| f.idx == pair.left_idx)?;
    let right = feats.iter().find(|f| f.idx == pair.right_idx)?;
    Some(PocketFlaps {
        y,
        left_cx: (left.x0 + left.x1) * 0.5,
        right_cx: (right.x0 + right.x1) * 0.5,
        confidence: pair.confidence,
    })
}

/// Top edge of the mask at column `x`, scanning up from `y_from`: the last
/// foreground y before the background (or `y_to` if the column stays
/// foreground). Used to snap detected landmarks onto the silhouette.
fn mask_top_at_x(mask: &[bool], img_w: usize, img_h: usize, x: f32, y_from: f32, y_to: f32) -> f32 {
    if mask.len() != img_w * img_h || img_w == 0 || img_h == 0 {
        return y_from;
    }
    let xi = (x.round().clamp(0.0, img_w as f32 - 1.0)) as usize;
    let mut y = y_from.round().clamp(0.0, img_h as f32 - 1.0);
    let stop = y_to.round().clamp(0.0, img_h as f32 - 1.0);
    let mut last_fg = y;
    while y >= stop {
        if mask[(y as usize) * img_w + xi] {
            last_fg = y;
        } else {
            break;
        }
        if y <= 0.0 {
            break;
        }
        y -= 1.0;
    }
    last_fg
}

/// A detected armhole (armscye) side: shoulder tip and underarm pit, both
/// photo-measured. The tip is the topmost armhole-chain point snapped up to
/// the silhouette; the pit is the inboard-most chain point in the armhole
/// band (where the inward curve meets the side seam).
#[derive(Clone, Copy, Debug)]
pub struct ArmholeSide {
    pub tip: (f32, f32),
    pub pit: (f32, f32),
    pub confidence: f32,
}

/// A symmetric armhole pair about the view axis.
#[derive(Clone, Copy, Debug)]
pub struct Armhole {
    pub left: ArmholeSide,
    pub right: ArmholeSide,
    pub confidence: f32,
}

/// Minimum chain arc length (px) for an armhole candidate.
const ARMHOLE_MIN_ARC: f32 = 45.0;
/// Minimum net downward travel (px) of a candidate chain.
const ARMHOLE_MIN_DROP: f32 = 35.0;
/// Maximum upward snap (px) when seating the shoulder tip on the
/// silhouette. The tip only corrects small chain/silhouette offsets; a
/// point deep inside the garment must not snap to the view top.
const ARMHOLE_MAX_TIP_SNAP: f32 = 25.0;
/// Maximum endpoint gap (px) when following an armhole chain down to its
/// continuation. Catches fragmented seams (blazer chain 8 -> 21, 52px)
/// while excluding nearby folds (chain 27 sits 98px above chain 17's end).
const ARMHOLE_FOLLOW_GAP: f32 = 60.0;

/// Shared inputs for armhole detection on one view.
struct ArmholeInput<'a> {
    feats: &'a [ChainFeat],
    chains: &'a [Vec<(f32, f32)>],
    view: &'a ViewContext,
    cx: f32,
    mask: &'a [bool],
    img_w: usize,
    img_h: usize,
}

/// An armhole candidate chain, reduced to its endpoints and arc.
struct ArmholeCand {
    idx: usize,
    top: (f32, f32),
    bottom: (f32, f32),
    arc: f32,
}

/// Detect one armhole side. `side` is -1.0 for the left (x < cx) side, +1.0
/// for the right. Returns the side detection with its confidence.
///
/// The armhole is traced by following chains: start from the topmost
/// substantial chain in the side band (the shoulder tip), then hop to the
/// nearest chain below (catching fragmented seams) until the trail ends.
/// The pit is the bottom of that trail — measured, never the inboard-most
/// point of unrelated chains.
fn detect_armhole_side(inp: &ArmholeInput, side: f32) -> Option<ArmholeSide> {
    let (feats, chains, view, cx, mask, img_w, img_h) = (
        inp.feats, inp.chains, inp.view, inp.cx, inp.mask, inp.img_w, inp.img_h,
    );
    let w = view.w();
    let h = view.h();
    // Side band, kept clear of the center front: lapels, princess seams,
    // and plackets live inside 0.18w; the armscye lives outboard of them.
    let x_lo = cx + side * 0.18 * w;
    let x_hi = cx + side * 0.48 * w;
    let (x_lo, x_hi) = (x_lo.min(x_hi), x_lo.max(x_hi));
    let y_lo = view.y0 + 0.03 * h;
    let y_hi = view.y0 + 0.50 * h;
    let mut cands: Vec<ArmholeCand> = Vec::new();
    for f in feats {
        if f.cx < x_lo || f.cx > x_hi || f.cy < y_lo || f.cy > y_hi {
            continue;
        }
        if f.arc_len < ARMHOLE_MIN_ARC {
            continue;
        }
        let c = match chains.get(f.idx) {
            Some(c) if !c.is_empty() => c,
            _ => continue,
        };
        // Chains store points in arbitrary order; use the y-extreme points.
        let top = c
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .copied()
            .unwrap();
        let bottom = c
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .copied()
            .unwrap();
        if bottom.1 - top.1 < ARMHOLE_MIN_DROP {
            continue;
        }
        cands.push(ArmholeCand {
            idx: f.idx,
            top,
            bottom,
            arc: f.arc_len,
        });
    }
    if cands.is_empty() {
        return None;
    }
    // Tip chain: the highest top. The armhole starts at the shoulder.
    cands.sort_by(|a, b| a.top.1.partial_cmp(&b.top.1).unwrap());
    let tip_cand = &cands[0];
    // Shoulder tip: topmost point, snapped up to the silhouette so the
    // rendered curve starts on the garment edge, not floating inside.
    // Distance-limited: a point deep inside the garment keeps its measured
    // position instead of jumping to the view top.
    let snapped_y = mask_top_at_x(mask, img_w, img_h, tip_cand.top.0, tip_cand.top.1, view.y0);
    let tip = if tip_cand.top.1 - snapped_y <= ARMHOLE_MAX_TIP_SNAP {
        (tip_cand.top.0, snapped_y)
    } else {
        tip_cand.top
    };
    // Follow the trail down: hop to the nearest chain whose top sits just
    // below the current bottom. This joins fragmented seams (8 -> 21) but
    // will not leap to a fold floating above the trail's end (27).
    let mut used = vec![tip_cand.idx];
    let mut bottom = tip_cand.bottom;
    let mut total_arc = tip_cand.arc;
    loop {
        let next = cands
            .iter()
            .filter(|c| !used.contains(&c.idx))
            .map(|c| {
                let d = ((c.top.0 - bottom.0).powi(2) + (c.top.1 - bottom.1).powi(2)).sqrt();
                (c, d)
            })
            .filter(|(c, d)| {
                *d < ARMHOLE_FOLLOW_GAP
                    && c.top.1 >= bottom.1 - 30.0
                    && c.top.1 <= view.y0 + 0.60 * h
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        match next {
            Some((c, _)) => {
                used.push(c.idx);
                bottom = c.bottom;
                total_arc += c.arc;
            }
            None => break,
        }
        if used.len() >= 4 {
            break;
        }
    }
    // Underarm pit: the bottom of the measured trail.
    let pit = bottom;
    // Gates: the pit must sit well below the tip (an armhole has real
    // vertical extent) and above the lower band edge (a full-height side
    // seam is not an armhole). Direction is lenient: a nearly vertical
    // trail is still an armhole, but a strongly outward one is a sleeve
    // outer edge.
    if pit.1 - tip.1 < 0.15 * h {
        return None;
    }
    if pit.1 > view.y0 + 0.65 * h {
        return None;
    }
    if -side * (pit.0 - tip.0) < -10.0 {
        return None;
    }
    let support = (total_arc / (0.35 * h)).min(1.0);
    let shape = ((pit.1 - tip.1) / (0.25 * h)).min(1.0);
    let confidence = (0.6 * support + 0.4 * shape).min(1.0);
    Some(ArmholeSide {
        tip,
        pit,
        confidence,
    })
}

/// Armhole (armscye) pair: the curved seam from each shoulder tip to its
/// underarm pit. Both sides must detect — symmetry about the view axis is a
/// first-class signal, and a lone inward-curving chain is more likely a
/// wrinkle or pocket edge than an armhole. Sleeveless garments, bottoms,
/// and the side view produce no pair and get no armhole linework.
pub fn detect_armhole(
    chains: &[Vec<(f32, f32)>],
    view: &ViewContext,
    cx: f32,
    mask: &[bool],
    img_w: usize,
    img_h: usize,
) -> Option<Armhole> {
    let feats = chain_features(chains);
    let inp = ArmholeInput {
        feats: &feats,
        chains,
        view,
        cx,
        mask,
        img_w,
        img_h,
    };
    let left = detect_armhole_side(&inp, -1.0)?;
    let right = detect_armhole_side(&inp, 1.0)?;
    // Symmetry: tip and pit distances from the axis should mirror.
    let w = view.w().max(1.0);
    let tip_sym =
        1.0 - (((cx - left.tip.0) - (right.tip.0 - cx)).abs() / (0.25 * w)).clamp(0.0, 1.0);
    let pit_sym =
        1.0 - (((cx - left.pit.0) - (right.pit.0 - cx)).abs() / (0.25 * w)).clamp(0.0, 1.0);
    let confidence =
        0.5 * (0.5 * tip_sym + 0.5 * pit_sym) + 0.25 * left.confidence + 0.25 * right.confidence;
    if confidence < DETECT_CONFIDENCE_MIN {
        return None;
    }
    Some(Armhole {
        left,
        right,
        confidence,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two horizontal segments, symmetric about cx=200, at y=120.
    fn pocket_chains() -> Vec<Vec<(f32, f32)>> {
        vec![
            vec![(130.0, 120.0), (190.0, 120.0)],
            vec![(210.0, 120.0), (270.0, 120.0)],
            // Distractor: short vertical wrinkle.
            vec![(200.0, 300.0), (200.0, 320.0)],
        ]
    }

    #[test]
    fn pocket_flaps_reject_single_side() {
        // Only the left chain: no pair, no detection.
        let chains = vec![vec![(130.0, 120.0), (190.0, 120.0)]];
        assert!(detect_pocket_flaps(&chains, 200.0, 400.0, 110.0).is_none());
    }

    #[test]
    fn pocket_flaps_ignore_far_zone() {
        // Pair far outside the expected zone: zone gate rejects.
        let chains = vec![
            vec![(130.0, 300.0), (190.0, 300.0)],
            vec![(210.0, 300.0), (270.0, 300.0)],
        ];
        assert!(detect_pocket_flaps(&chains, 200.0, 400.0, 110.0).is_none());
    }

    #[test]
    fn gorge_single_chain_mirrored() {
        // One clear seam on the right; the detector mirrors it.
        let chains = vec![vec![(230.0, 178.0), (290.0, 180.0)]];
        let (y, conf) = detect_gorge_y(&chains, 200.0, 400.0, 180.0).expect("gorge");
        assert!((y - 179.0).abs() < 2.0, "y={y}");
        assert!(conf >= DETECT_CONFIDENCE_MIN, "conf={conf}");
    }

    #[test]
    fn gorge_prefers_pair_over_lone_chain() {
        // Symmetric pair at y=182 beats a lone chain at y=160.
        let chains = vec![
            vec![(230.0, 182.0), (280.0, 182.0)],
            vec![(120.0, 182.0), (170.0, 182.0)],
            vec![(230.0, 160.0), (300.0, 161.0)],
        ];
        let (y, _) = detect_gorge_y(&chains, 200.0, 400.0, 180.0).expect("gorge");
        assert!((y - 182.0).abs() < 2.0, "y={y}");
    }

    #[test]
    fn lapel_pair_found_on_synthetic() {
        // Two long vertical chains, mirrored about cx=200, CONVERGING going
        // down (lapel-like: peak wide at top, narrowing to the break).
        let chains = vec![
            vec![
                (130.0, 100.0),
                (135.0, 137.0),
                (140.0, 175.0),
                (145.0, 212.0),
                (150.0, 250.0),
            ],
            vec![
                (270.0, 100.0),
                (265.0, 137.0),
                (260.0, 175.0),
                (255.0, 212.0),
                (250.0, 250.0),
            ],
            vec![(200.0, 100.0), (200.0, 400.0)], // center: excluded by x_inner
        ];
        let p = detect_lapel_pair(&chains, 200.0, 400.0, 50.0, 500.0).expect("lapels");
        assert_eq!(p.left_idx, 0);
        assert_eq!(p.right_idx, 1);
        assert!(p.confidence >= DETECT_CONFIDENCE_MIN);
    }

    #[test]
    fn lapel_pair_rejects_splaying_pair() {
        // Jeans pocket openings: mirrored and symmetric, but they DIVERGE
        // going down (measured -37px on the real jeans photo). Not lapels.
        let chains = vec![
            vec![
                (150.0, 100.0),
                (145.0, 137.0),
                (140.0, 175.0),
                (135.0, 212.0),
                (130.0, 250.0),
            ],
            vec![
                (250.0, 100.0),
                (255.0, 137.0),
                (260.0, 175.0),
                (265.0, 212.0),
                (270.0, 250.0),
            ],
        ];
        assert!(detect_lapel_pair(&chains, 200.0, 400.0, 50.0, 500.0).is_none());
    }

    #[test]
    fn lapel_pair_rejects_parallel_pair() {
        // Fly/placket edges: mirrored and symmetric but parallel (zero
        // convergence). Not lapels.
        let chains = vec![
            vec![
                (140.0, 100.0),
                (140.0, 137.0),
                (140.0, 175.0),
                (140.0, 212.0),
                (140.0, 250.0),
            ],
            vec![
                (260.0, 100.0),
                (260.0, 137.0),
                (260.0, 175.0),
                (260.0, 212.0),
                (260.0, 250.0),
            ],
        ];
        assert!(detect_lapel_pair(&chains, 200.0, 400.0, 50.0, 500.0).is_none());
    }

    #[test]
    fn lapel_pair_rejects_non_mirrored() {
        // Right chain far from the mirror position.
        let chains = vec![
            vec![(120.0, 100.0), (122.0, 250.0)],
            vec![(340.0, 110.0), (342.0, 240.0)],
        ];
        assert!(detect_lapel_pair(&chains, 200.0, 400.0, 50.0, 500.0).is_none());
    }

    #[test]
    fn empty_chains_detect_nothing() {
        let empty: Vec<Vec<(f32, f32)>> = vec![];
        assert!(detect_gorge_y(&empty, 200.0, 400.0, 180.0).is_none());
        assert!(detect_lapel_pair(&empty, 200.0, 400.0, 50.0, 500.0).is_none());
    }

    fn test_view() -> ViewContext {
        ViewContext::new(0.0, 0.0, 400.0, 500.0)
    }

    #[test]
    fn pocket_flaps_report_photo_x_centers() {
        let flaps = detect_pocket_flaps(&pocket_chains(), 200.0, 400.0, 110.0).expect("flaps");
        assert!((flaps.y - 120.0).abs() < 1.0, "y={}", flaps.y);
        assert!(
            (flaps.left_cx - 160.0).abs() < 1.0,
            "left={}",
            flaps.left_cx
        );
        assert!(
            (flaps.right_cx - 240.0).abs() < 1.0,
            "right={}",
            flaps.right_cx
        );
        assert!(flaps.confidence >= DETECT_CONFIDENCE_MIN);
    }

    #[test]
    fn pocket_flaps_reject_asymmetric_pair() {
        let chains = vec![
            vec![(130.0, 120.0), (190.0, 120.0)],
            vec![(210.0, 160.0), (270.0, 160.0)],
        ];
        assert!(detect_pocket_flaps(&chains, 200.0, 400.0, 110.0).is_none());
    }

    #[test]
    fn axis_from_mask_uses_centroid() {
        // Foreground block shifted left of bbox center (200): axis follows it.
        let (w, h) = (400usize, 100usize);
        let mut mask = vec![false; w * h];
        for y in 0..100 {
            for x in 100..180 {
                mask[y * w + x] = true;
            }
        }
        let view = ViewContext::new(0.0, 0.0, 400.0, 100.0);
        let axis = refine_axis_from_mask(&mask, w, h, &view);
        assert!((axis - 140.0).abs() < 1.0, "axis={axis}");
    }

    #[test]
    fn axis_from_mask_falls_back_on_empty() {
        let view = ViewContext::new(0.0, 0.0, 400.0, 100.0);
        let axis = refine_axis_from_mask(&vec![false; 400 * 100], 400, 100, &view);
        assert!((axis - 200.0).abs() < 1e-6, "axis={axis}");
    }

    #[test]
    fn horizontal_seams_rank_by_coverage_and_symmetry() {
        // Two horizontal seams: a long symmetric one (y=300) and a short
        // off-center one (y=150). The long one must rank first.
        let chains = vec![
            vec![(50.0, 300.0), (350.0, 300.0)],
            vec![(260.0, 150.0), (330.0, 150.0)],
        ];
        let cands = detect_horizontal_seams(&chains, &test_view(), 30.0);
        assert_eq!(cands.len(), 2, "candidates={cands:?}");
        assert!((cands[0].pos - 300.0).abs() < 1.0, "pos={}", cands[0].pos);
        assert!(cands[0].confidence > cands[1].confidence);
        assert!((cands[0].lo - 50.0).abs() < 1.0 && (cands[0].hi - 350.0).abs() < 1.0);
    }

    #[test]
    fn horizontal_seams_ignore_vertical_chains() {
        let chains = vec![vec![(200.0, 50.0), (200.0, 450.0)]];
        let cands = detect_horizontal_seams(&chains, &test_view(), 30.0);
        assert!(cands.is_empty());
    }

    #[test]
    fn vertical_seams_find_placket() {
        // Long vertical chain near the axis: placket candidate.
        let chains = vec![
            vec![(198.0, 50.0), (202.0, 450.0)],
            vec![(50.0, 100.0), (350.0, 100.0)], // horizontal distractor
        ];
        let cands = detect_vertical_seams(&chains, &test_view(), 100.0);
        assert_eq!(cands.len(), 1, "candidates={cands:?}");
        assert!((cands[0].pos - 200.0).abs() < 3.0, "pos={}", cands[0].pos);
        assert!((cands[0].lo - 50.0).abs() < 1.0 && (cands[0].hi - 450.0).abs() < 1.0);
    }

    #[test]
    fn button_columns_two_col_midpoint() {
        // Double-breasted layout: two columns; center is their midpoint.
        let pts = vec![
            (360.5, 425.5),
            (463.0, 425.5),
            (373.0, 492.0),
            (451.0, 493.0),
        ];
        let cols = detect_button_columns(&pts, &test_view());
        assert_eq!(cols.len(), 2, "cols={cols:?}");
        let cx = closure_center(&cols).expect("center");
        assert!((cx - 411.9).abs() < 1.0, "cx={cx}");
    }

    #[test]
    fn button_columns_single_col_is_placket() {
        // Shirt-style single column: the column itself is the center.
        let pts = vec![(200.0, 100.0), (201.0, 200.0), (199.0, 300.0)];
        let cols = detect_button_columns(&pts, &test_view());
        assert_eq!(cols.len(), 1);
        assert_eq!(cols[0].ys.len(), 3);
        let cx = closure_center(&cols).expect("center");
        assert!((cx - 200.0).abs() < 1.0, "cx={cx}");
    }

    #[test]
    fn button_columns_ambiguous_returns_none() {
        let pts = vec![(100.0, 100.0), (200.0, 100.0), (300.0, 100.0)];
        let cols = detect_button_columns(&pts, &test_view());
        assert_eq!(cols.len(), 3);
        assert!(closure_center(&cols).is_none());
        assert!(closure_center(&[]).is_none());
    }

    // ---- #41: buttonless garment support ----

    /// Fill a rect in a fresh mask.
    fn fill_rect(w: usize, h: usize, x0: usize, x1: usize, y0: usize, y1: usize) -> Vec<bool> {
        let mut m = vec![false; w * h];
        for y in y0..y1.min(h) {
            for x in x0..x1.min(w) {
                m[y * w + x] = true;
            }
        }
        m
    }

    /// Bowed crew-neckline chains: y ~135 at center, ~150 at the edges.
    fn crew_chains() -> Vec<Vec<(f32, f32)>> {
        let mut chains = Vec::new();
        for s in [0, 1] {
            let mut c = Vec::new();
            for i in 0..=14 {
                let x = 180.0 + i as f32 * 10.0 + s as f32 * 5.0;
                let y = 135.0 + ((x - 250.0) / 70.0).powi(2) * 15.0 + s as f32 * 3.0;
                c.push((x, y));
            }
            chains.push(c);
        }
        chains
    }

    #[test]
    fn mask_symmetry_perfect_and_degenerate() {
        let w = 100;
        let h = 100;
        let m = fill_rect(w, h, 20, 80, 10, 90);
        let s = mask_symmetry(&m, w, h, (20.0, 10.0, 80.0, 90.0), 50.0);
        assert!(s > 0.99, "sym={s}");
        let e = vec![false; w * h];
        assert_eq!(mask_symmetry(&e, w, h, (20.0, 10.0, 80.0, 90.0), 50.0), 0.0);
        let o = fill_rect(w, h, 20, 50, 10, 90);
        let s2 = mask_symmetry(&o, w, h, (20.0, 10.0, 80.0, 90.0), 50.0);
        assert!(s2 < 0.05, "sym={s2}");
    }

    #[test]
    fn identify_views_picks_buttoned_front() {
        let comps = vec![
            (0.0, 0.0, 0.0, 0.0, 0usize),
            (100.0, 100.0, 500.0, 700.0, 200000),
            (600.0, 100.0, 1000.0, 700.0, 190000),
        ];
        let buttons = vec![
            (200.0, 300.0),
            (300.0, 300.0),
            (200.0, 400.0),
            (300.0, 400.0),
        ];
        let mask = fill_rect(1100, 800, 100, 500, 100, 700);
        let (front, back) = identify_views(&comps, &buttons, &[], &mask, 1100, 800);
        assert_eq!(front, Some(1));
        assert_eq!(back, Some(2));
    }

    #[test]
    fn identify_views_finds_front_without_buttons() {
        // Buttonless tee: two symmetric comps; only the first has neckline chains.
        let comps = vec![
            (0.0, 0.0, 0.0, 0.0, 0usize),
            (100.0, 100.0, 500.0, 700.0, 200000),
            (600.0, 100.0, 1000.0, 700.0, 195000),
        ];
        let chains = crew_chains();
        let mut mask = fill_rect(1100, 800, 100, 500, 100, 700);
        for y in 100..700 {
            for x in 600..1000 {
                mask[y * 1100 + x] = true;
            }
        }
        let (front, back) = identify_views(&comps, &[], &chains, &mask, 1100, 800);
        assert_eq!(front, Some(1), "neckline evidence should pick comp 1");
        assert_eq!(back, Some(2));
    }

    #[test]
    fn identify_views_none_when_nothing_qualifies() {
        let comps = vec![(0.0, 0.0, 0.0, 0.0, 0usize), (10.0, 10.0, 30.0, 40.0, 500)];
        let mask = vec![false; 100 * 100];
        let (front, back) = identify_views(&comps, &[], &[], &mask, 100, 100);
        assert_eq!(front, None);
        assert_eq!(back, None);
    }

    /// Test mask: filled rectangle (x_lo..x_hi, y_lo..y_hi) in an img_w x img_h
    /// grid, for neckline local-width tests.
    fn rect_mask(
        img_w: usize,
        img_h: usize,
        x_lo: usize,
        x_hi: usize,
        y_lo: usize,
        y_hi: usize,
    ) -> Vec<bool> {
        let mut m = vec![false; img_w * img_h];
        for y in y_lo..y_hi.min(img_h) {
            for x in x_lo..x_hi.min(img_w) {
                m[y * img_w + x] = true;
            }
        }
        m
    }

    #[test]
    fn detect_neckline_finds_crew() {
        let view = ViewContext::new(100.0, 100.0, 500.0, 700.0);
        // Wide body at the seam height: the 145px crew neckline is ~0.36 of
        // the local width, well under the 0.75 waistband gate.
        let mask = rect_mask(600, 800, 100, 500, 130, 160);
        let nl = detect_neckline(&crew_chains(), &view, &mask, 600, 800).expect("neckline");
        assert!((nl.y - 141.0).abs() < 8.0, "y={}", nl.y);
        assert!((nl.x0 - 180.0).abs() < 15.0, "x0={}", nl.x0);
        assert!((nl.x1 - 325.0).abs() < 15.0, "x1={}", nl.x1);
        assert!(nl.depth > 5.0, "depth={}", nl.depth);
        assert!(nl.confidence >= 0.5, "conf={}", nl.confidence);
    }

    #[test]
    fn detect_neckline_rejects_straight_yoke() {
        let view = ViewContext::new(100.0, 100.0, 500.0, 700.0);
        let chains = vec![vec![
            (180.0, 140.0),
            (220.0, 140.0),
            (260.0, 140.0),
            (300.0, 140.0),
        ]];
        let mask = rect_mask(600, 800, 100, 500, 130, 160);
        assert!(detect_neckline(&chains, &view, &mask, 600, 800).is_none());
    }

    #[test]
    fn detect_neckline_rejects_full_width_waistband() {
        // A bowed seam spanning the body at its height (like a jeans
        // waistband: 380px seam on a 400px body = 0.95 of local width) is
        // not a neckline, even though it curves.
        let view = ViewContext::new(100.0, 100.0, 500.0, 700.0);
        let chains = vec![vec![
            (110.0, 140.0),
            (200.0, 143.0),
            (300.0, 146.0),
            (400.0, 143.0),
            (490.0, 140.0),
        ]];
        let mask = rect_mask(600, 800, 100, 500, 130, 160);
        assert!(detect_neckline(&chains, &view, &mask, 600, 800).is_none());
    }

    #[test]
    fn detect_neckline_none_without_chains() {
        let view = ViewContext::new(100.0, 100.0, 500.0, 700.0);
        let mask = rect_mask(600, 800, 100, 500, 130, 160);
        assert!(detect_neckline(&[], &view, &mask, 600, 800).is_none());
    }

    /// Mask with two separated legs (jeans-like): two disjoint intervals on
    /// the lower rows.
    fn legs_mask() -> Vec<bool> {
        let (w, h) = (200usize, 400usize);
        let mut m = vec![false; w * h];
        // Legs: x 20-80 and x 120-180, from y=200 to y=400.
        for y in 200..400 {
            for x in 20..80 {
                m[y * w + x] = true;
            }
            for x in 120..180 {
                m[y * w + x] = true;
            }
        }
        // Torso: x 20-180, y 0-200 (single interval up top).
        for y in 0..200 {
            for x in 20..180 {
                m[y * w + x] = true;
            }
        }
        m
    }

    #[test]
    fn has_separated_legs_finds_jeans() {
        let view = ViewContext::new(0.0, 0.0, 200.0, 400.0);
        assert!(has_separated_legs(&legs_mask(), 200, 400, &view));
    }

    #[test]
    fn has_separated_legs_rejects_single_blob() {
        // Tee-like: single interval on all rows.
        let (w, h) = (200usize, 400usize);
        let mut m = vec![false; w * h];
        for y in 0..400 {
            for x in 50..150 {
                m[y * w + x] = true;
            }
        }
        let view = ViewContext::new(0.0, 0.0, 200.0, 400.0);
        assert!(!has_separated_legs(&m, 200, 400, &view));
    }

    #[test]
    fn has_separated_legs_ignores_noise_gap() {
        // Single blob with a 5px noise gap — merged, not legs.
        let (w, h) = (200usize, 400usize);
        let mut m = vec![false; w * h];
        for y in 0..400 {
            for x in 50..150 {
                if !(98..103).contains(&x) {
                    m[y * w + x] = true;
                }
            }
        }
        let view = ViewContext::new(0.0, 0.0, 200.0, 400.0);
        assert!(!has_separated_legs(&m, 200, 400, &view));
    }

    // ---- armhole (armscye) detection ----

    /// Synthetic armhole pair: left chain goes down and inward (toward
    /// cx=200), right chain mirrors it. Both curved, not straight.
    fn armhole_chains() -> Vec<Vec<(f32, f32)>> {
        vec![
            vec![
                (60.0, 60.0),
                (62.0, 90.0),
                (68.0, 120.0),
                (78.0, 150.0),
                (90.0, 175.0),
            ],
            vec![
                (340.0, 60.0),
                (338.0, 90.0),
                (332.0, 120.0),
                (322.0, 150.0),
                (310.0, 175.0),
            ],
        ]
    }

    /// Mask covering the view's upper-left/right (so tip snap-up works).
    fn armhole_mask() -> Vec<bool> {
        let (w, h) = (400usize, 500usize);
        let mut m = vec![false; w * h];
        for y in 40..400 {
            for x in 40..360 {
                m[y * w + x] = true;
            }
        }
        m
    }

    #[test]
    fn armhole_detects_synthetic_pair() {
        let view = ViewContext::new(0.0, 0.0, 400.0, 500.0);
        let a = detect_armhole(&armhole_chains(), &view, 200.0, &armhole_mask(), 400, 500)
            .expect("armhole pair");
        // Tips at the top of the chains, snapped up to the mask top (y=40).
        assert!((a.left.tip.0 - 60.0).abs() < 2.0, "tip={:?}", a.left.tip);
        assert!((a.left.tip.1 - 40.0).abs() < 2.0, "tip={:?}", a.left.tip);
        assert!((a.right.tip.0 - 340.0).abs() < 2.0, "tip={:?}", a.right.tip);
        // Pits inboard of the tips and well below them.
        assert!(a.left.pit.0 > a.left.tip.0, "pit={:?}", a.left.pit);
        assert!(a.right.pit.0 < a.right.tip.0, "pit={:?}", a.right.pit);
        assert!(a.left.pit.1 - a.left.tip.1 > 60.0);
        assert!(
            a.confidence >= DETECT_CONFIDENCE_MIN,
            "conf={}",
            a.confidence
        );
    }

    #[test]
    fn armhole_rejects_single_side() {
        // Only the left chain: no pair, no detection.
        let view = ViewContext::new(0.0, 0.0, 400.0, 500.0);
        let chains = vec![armhole_chains()[0].clone()];
        assert!(detect_armhole(&chains, &view, 200.0, &armhole_mask(), 400, 500).is_none());
    }

    #[test]
    fn armhole_rejects_outward_chains() {
        // Sleeve outer edges: go down and OUTWARD (away from the axis).
        // Same symmetry, wrong direction — not armholes.
        let view = ViewContext::new(0.0, 0.0, 400.0, 500.0);
        let chains = vec![
            vec![(100.0, 60.0), (90.0, 100.0), (80.0, 140.0), (70.0, 180.0)],
            vec![
                (300.0, 60.0),
                (310.0, 100.0),
                (320.0, 140.0),
                (330.0, 180.0),
            ],
        ];
        assert!(detect_armhole(&chains, &view, 200.0, &armhole_mask(), 400, 500).is_none());
    }

    #[test]
    fn armhole_rejects_straight_side_seam() {
        // Long vertical trails that run past the underarm into the lower
        // body: side seams, not armholes (the 0.65h pit gate rejects). A
        // short vertical trail in the upper band is armhole-like (the
        // blazer's right armhole is nearly vertical) and must NOT be
        // rejected — only the over-long trail is.
        let view = ViewContext::new(0.0, 0.0, 400.0, 500.0);
        let chains = vec![
            vec![
                (100.0, 60.0),
                (101.0, 180.0),
                (102.0, 290.0),
                (103.0, 400.0),
            ],
            vec![
                (300.0, 60.0),
                (299.0, 180.0),
                (298.0, 290.0),
                (297.0, 400.0),
            ],
        ];
        assert!(detect_armhole(&chains, &view, 200.0, &armhole_mask(), 400, 500).is_none());
    }

    #[test]
    fn armhole_none_without_chains() {
        let view = ViewContext::new(0.0, 0.0, 400.0, 500.0);
        let empty: Vec<Vec<(f32, f32)>> = vec![];
        assert!(detect_armhole(&empty, &view, 200.0, &armhole_mask(), 400, 500).is_none());
    }

    #[test]
    fn mask_top_at_x_snaps_to_silhouette() {
        // Foreground y 40..400 at x=120: scanning up from y=100 stops at 40.
        assert!((mask_top_at_x(&armhole_mask(), 400, 500, 120.0, 100.0, 0.0) - 40.0).abs() < 1e-6);
        // Background column: returns the start y unchanged.
        assert!((mask_top_at_x(&armhole_mask(), 400, 500, 10.0, 100.0, 0.0) - 100.0).abs() < 1e-6);
    }
}
