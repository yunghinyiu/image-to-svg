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
            let (mut y0, mut y1) = (f32::INFINITY, f32::NEG_INFINITY);
            let (mut sx, mut sy) = (0.0f32, 0.0f32);
            for w in c.windows(2) {
                let ((ax, ay), (bx, by)) = (w[0], w[1]);
                arc += (bx - ax).hypot(by - ay);
                for (px, py) in [w[0], w[1]] {
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

/// Pocket flap y from a symmetric pair of near-horizontal chains in the
/// lower front. Returns `(y, confidence)`.
pub fn detect_pocket_y(
    chains: &[Vec<(f32, f32)>],
    cx: f32,
    w: f32,
    y_expected: f32,
) -> Option<(f32, f32)> {
    let feats = chain_features(chains);
    let cfg = HPairConfig {
        len_min: 45.0,
        len_max: 220.0,
        angle_tol: 15.0,
        straight_min: 0.65,
        zone_hw: 70.0,
        x_inner: 0.05 * w,
        x_outer: 0.45 * w,
        y_sym_tol: 12.0, // issue #29: symmetric within 12px
        mirror_tol: 30.0,
    };
    find_horizontal_pair(&feats, cx, y_expected, &cfg)
        .filter(|(p, _)| p.confidence >= DETECT_CONFIDENCE_MIN)
        .map(|(p, y)| (y, p.confidence))
}

/// Gorge (notch) y from the gorge seam: a symmetric horizontal pair is
/// preferred; otherwise the best single chain is mirrored to the other side.
/// Returns `(y, confidence)`.
pub fn detect_gorge_y(
    chains: &[Vec<(f32, f32)>],
    cx: f32,
    w: f32,
    y_expected: f32,
) -> Option<(f32, f32)> {
    let feats = chain_features(chains);
    let cfg = HPairConfig {
        len_min: 30.0,
        len_max: 120.0,
        angle_tol: 15.0,
        straight_min: 0.75,
        zone_hw: 45.0, // the notch zone is tight
        x_inner: 0.0,
        x_outer: 0.35 * w,
        y_sym_tol: 10.0,
        mirror_tol: 25.0,
    };
    if let Some((pair, y)) = find_horizontal_pair(&feats, cx, y_expected, &cfg)
        .filter(|(p, _)| p.confidence >= DETECT_CONFIDENCE_MIN)
    {
        return Some((y, pair.confidence));
    }
    // Single-chain fallback: the clearest lone seam, mirrored.
    let mut best: Option<(f32, f32)> = None; // (y, score)
    for f in &feats {
        if f.arc_len < 30.0
            || f.arc_len > 120.0
            || horiz_dev(f) > 15.0
            || f.straightness < 0.8
            || (f.cy - y_expected).abs() > 45.0
            || (f.cx - cx).abs() > 0.35 * w
        {
            continue;
        }
        let zone = 1.0 - (f.cy - y_expected).abs() / 45.0;
        let len_n = (f.arc_len - 30.0) / 90.0;
        let score = 0.4 * clamp01(zone) + 0.3 * f.straightness + 0.3 * clamp01(len_n);
        if best.is_none_or(|(_, s)| score > s) {
            best = Some((f.cy, score));
        }
    }
    best.filter(|&(_, s)| s >= DETECT_CONFIDENCE_MIN)
}

/// Lapel edge pair: long, near-vertical chains in the upper front, symmetric
/// about the center front. Returns the pair with confidence; the caller logs
/// it as a structural signal (placement itself is optimized by #27 search).
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
    fn pocket_pair_found_on_synthetic() {
        let d = detect_pocket_y(&pocket_chains(), 200.0, 400.0, 110.0);
        let (y, conf) = d.expect("should detect the pair");
        assert!((y - 120.0).abs() < 1.0, "y={y}");
        assert!(conf >= DETECT_CONFIDENCE_MIN, "conf={conf}");
    }

    #[test]
    fn pocket_detector_rejects_single_side() {
        // Only the left chain: no pair, no detection.
        let chains = vec![vec![(130.0, 120.0), (190.0, 120.0)]];
        assert!(detect_pocket_y(&chains, 200.0, 400.0, 110.0).is_none());
    }

    #[test]
    fn pocket_detector_rejects_asymmetric_pair() {
        // Right chain 40px lower: exceeds the 12px symmetry tolerance.
        let chains = vec![
            vec![(130.0, 120.0), (190.0, 120.0)],
            vec![(210.0, 160.0), (270.0, 160.0)],
        ];
        assert!(detect_pocket_y(&chains, 200.0, 400.0, 110.0).is_none());
    }

    #[test]
    fn pocket_detector_ignores_far_zone() {
        // Pair far outside the expected zone: zone gate rejects.
        let chains = vec![
            vec![(130.0, 300.0), (190.0, 300.0)],
            vec![(210.0, 300.0), (270.0, 300.0)],
        ];
        assert!(detect_pocket_y(&chains, 200.0, 400.0, 110.0).is_none());
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
        // Two long vertical chains, mirrored about cx=200.
        let chains = vec![
            vec![(120.0, 100.0), (122.0, 250.0)],
            vec![(278.0, 110.0), (280.0, 240.0)],
            vec![(200.0, 100.0), (200.0, 400.0)], // center: excluded by x_inner
        ];
        let p = detect_lapel_pair(&chains, 200.0, 400.0, 50.0, 500.0).expect("lapels");
        assert_eq!(p.left_idx, 0);
        assert_eq!(p.right_idx, 1);
        assert!(p.confidence >= DETECT_CONFIDENCE_MIN);
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
        assert!(detect_pocket_y(&empty, 200.0, 400.0, 110.0).is_none());
        assert!(detect_gorge_y(&empty, 200.0, 400.0, 180.0).is_none());
        assert!(detect_lapel_pair(&empty, 200.0, 400.0, 50.0, 500.0).is_none());
    }
}
