//! #27 — candidate placement search.
//!
//! Button-relative template heuristics land close but leave residual
//! misalignment against the photo. Following the generate-and-select pattern,
//! we generate a small set of candidate placements on the 8px quantization
//! grid, score each by photo edge alignment, and keep the best — gated by a
//! confidence margin so weak edge evidence falls back to the default
//! placement instead of chasing noise.

use crate::template::{gorge_seam_template, lapel_template, render_template};
use crate::template::{Placement, RenderedPath, Template};

/// Radius (px) within which a template sample point counts as "on" a photo edge.
const EDGE_HIT_RADIUS_PX: i32 = 4;
/// Minimum absolute edge-hit fraction for a candidate to be trusted at all.
const MIN_EDGE_HIT_FRAC: f32 = 0.30;
/// Required margin over the default placement before accepting a move.
const ACCEPT_MARGIN: f32 = 0.05;
/// Scoring resample step (px) along template paths.
const SCORE_STEP_PX: f32 = 4.0;

/// Candidate vertical offsets (px) for the pocket flap, on the 8px grid.
const POCKET_DY: [f32; 5] = [-16.0, -8.0, 0.0, 8.0, 16.0];
/// Candidate vertical offsets (px) for the gorge / lapel top.
const GORGE_DY: [f32; 3] = [-8.0, 0.0, 8.0];
/// Candidate width deltas (px) for the lapel pair.
const LAPEL_DW: [f32; 3] = [-8.0, 0.0, 8.0];

/// Boolean edge raster: photo edge pixels dilated by [`EDGE_HIT_RADIUS_PX`],
/// so scoring a template point is an O(1) lookup.
pub struct EdgeMap {
    w: usize,
    h: usize,
    hits: Vec<bool>,
}

impl EdgeMap {
    /// Build from photo edge chains (polylines in output pixel coordinates).
    /// Chain segments are rasterized so diagonal edges have no gaps, then
    /// dilated by the hit radius.
    pub fn from_chains(chains: &[Vec<(f32, f32)>], w: usize, h: usize) -> Self {
        let mut map = Self {
            w,
            h,
            hits: vec![false; w.saturating_mul(h)],
        };
        for chain in chains {
            for seg in chain.windows(2) {
                let (x0, y0) = seg[0];
                let (x1, y1) = seg[1];
                let steps = ((x1 - x0).hypot(y1 - y0).ceil() as i32).max(1);
                for i in 0..=steps {
                    let t = i as f32 / steps as f32;
                    map.mark_dilated(
                        (x0 + (x1 - x0) * t).round() as i32,
                        (y0 + (y1 - y0) * t).round() as i32,
                    );
                }
            }
            if chain.len() == 1 {
                map.mark_dilated(chain[0].0.round() as i32, chain[0].1.round() as i32);
            }
        }
        map
    }

    fn mark_dilated(&mut self, x: i32, y: i32) {
        let r = EDGE_HIT_RADIUS_PX;
        for dy in -r..=r {
            for dx in -r..=r {
                let (px, py) = (x + dx, y + dy);
                if px >= 0 && py >= 0 && (px as usize) < self.w && (py as usize) < self.h {
                    self.hits[py as usize * self.w + px as usize] = true;
                }
            }
        }
    }

    /// True when the point lies within the hit radius of a photo edge.
    pub fn is_edge(&self, x: f32, y: f32) -> bool {
        let (xi, yi) = (x.round() as i32, y.round() as i32);
        if xi < 0 || yi < 0 || (xi as usize) >= self.w || (yi as usize) >= self.h {
            return false;
        }
        self.hits[yi as usize * self.w + xi as usize]
    }
}

/// Resample a polyline at ~`step` px spacing so the edge-hit score is not
/// biased toward densely-sampled curves over sparse straight segments.
fn resample(points: &[(f32, f32)], step: f32) -> Vec<(f32, f32)> {
    let mut out = Vec::new();
    if points.is_empty() {
        return out;
    }
    out.push(points[0]);
    let mut since_last = 0.0f32;
    let mut prev = points[0];
    for &pt in &points[1..] {
        let seg_len = (pt.0 - prev.0).hypot(pt.1 - prev.1);
        if seg_len < 1e-9 {
            prev = pt;
            continue;
        }
        let (dx, dy) = ((pt.0 - prev.0) / seg_len, (pt.1 - prev.1) / seg_len);
        let mut t = step - since_last;
        let (mut bx, mut by) = prev;
        let mut rest = seg_len;
        let mut emitted = false;
        while t <= rest + 1e-6 {
            bx += dx * t;
            by += dy * t;
            out.push((bx, by));
            rest -= t;
            t = step;
            emitted = true;
        }
        since_last = if emitted { rest } else { since_last + seg_len };
        prev = pt;
    }
    out
}

/// Fraction of a rendered template's *solid*-path sample points landing on
/// photo edges. Only solid (structural) paths count: dashed topstitching is
/// decorative and often invisible to XDoG.
pub fn edge_hit_fraction(rendered: &[RenderedPath], edges: &EdgeMap) -> f32 {
    let mut hits = 0usize;
    let mut total = 0usize;
    for rp in rendered.iter().filter(|r| !r.dashed) {
        for p in resample(&rp.points, SCORE_STEP_PX) {
            total += 1;
            if edges.is_edge(p.0, p.1) {
                hits += 1;
            }
        }
    }
    if total == 0 {
        0.0
    } else {
        hits as f32 / total as f32
    }
}

/// Render `template` through each of `placements` and score the union.
fn score_placements(template: &Template, placements: &[Placement], edges: &EdgeMap) -> f32 {
    let mut rendered = Vec::new();
    for p in placements {
        rendered.extend(render_template(template, p));
    }
    edge_hit_fraction(&rendered, edges)
}

/// Grid-search lapel gorge height × width. Returns the winning `(vg, w)`.
/// Falls back to the defaults unless a candidate beats them by
/// [`ACCEPT_MARGIN`] with an absolute score above [`MIN_EDGE_HIT_FRAC`].
/// The gorge seam shares `vg`, so it is scored (and moved) together with the
/// lapels.
pub fn search_lapel(vg_default: f32, vb: f32, frame: &Placement, edges: &EdgeMap) -> (f32, f32) {
    let score = |vg: f32, w: f32| -> f32 {
        let lapel = lapel_template(vg, vb);
        let gorge = gorge_seam_template(vg);
        let placements = [
            Placement {
                w,
                mirror: false,
                ..*frame
            },
            Placement {
                w,
                mirror: true,
                ..*frame
            },
        ];
        let mut rendered = Vec::new();
        for p in &placements {
            rendered.extend(render_template(&lapel, p));
        }
        rendered.extend(render_template(
            &gorge,
            &Placement {
                w,
                mirror: false,
                ..*frame
            },
        ));
        edge_hit_fraction(&rendered, edges)
    };

    let default_score = score(vg_default, frame.w);
    let mut best = (vg_default, frame.w, default_score);
    for &dy in &GORGE_DY {
        for &dw in &LAPEL_DW {
            if dy == 0.0 && dw == 0.0 {
                continue; // default already scored
            }
            let vg = vg_default + dy / frame.h;
            let w = frame.w + dw;
            let s = score(vg, w);
            if s > best.2 {
                best = (vg, w, s);
            }
        }
    }
    if best.2 >= MIN_EDGE_HIT_FRAC && best.2 - default_score >= ACCEPT_MARGIN {
        log_choice(
            "lapel",
            &format!("vg {vg_default:.3}->{:.3}", best.0),
            best.2,
        );
        (best.0, best.1)
    } else {
        (vg_default, frame.w)
    }
}

/// Grid-search the pocket flap Y, shared by both flaps to preserve symmetry.
/// `left_ax` / `right_ax` are the photo-measured flap centers (from
/// [`crate::detect::detect_pocket_flaps`]); the search only moves Y.
/// Returns the winning `ay`, or `ay_default` when no candidate is clearly
/// better.
pub fn search_pocket_y(
    ay_default: f32,
    pocket: &Template,
    left_ax: f32,
    right_ax: f32,
    w: f32,
    h: f32,
    edges: &EdgeMap,
) -> f32 {
    let score = |ay: f32| -> f32 {
        let placements = [
            Placement {
                ax: left_ax,
                ay,
                w,
                h,
                mirror: false,
            },
            Placement {
                ax: right_ax,
                ay,
                w,
                h,
                mirror: false,
            },
        ];
        score_placements(pocket, &placements, edges)
    };

    let default_score = score(ay_default);
    let mut best = (ay_default, default_score);
    for &dy in &POCKET_DY {
        if dy == 0.0 {
            continue; // default already scored
        }
        let s = score(ay_default + dy);
        if s > best.1 {
            best = (ay_default + dy, s);
        }
    }
    if best.1 >= MIN_EDGE_HIT_FRAC && best.1 - default_score >= ACCEPT_MARGIN {
        log_choice(
            "pocket_y",
            &format!("{ay_default:.1}->{:.1}", best.0),
            best.1,
        );
        best.0
    } else {
        ay_default
    }
}

fn log_choice(what: &str, change: &str, score: f32) {
    if std::env::var("IM2VEC_FLAT_DEBUG").is_ok() {
        eprintln!("[search] {what}: {change} (edge-hit {score:.2})");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::template::{TPath, TPoint};

    /// Single solid horizontal line at v=0, spanning u in [-0.09, 0.09].
    fn hline_template() -> Template {
        Template {
            name: "hline",
            paths: vec![TPath {
                points: vec![TPoint::Norm(-0.09, 0.0), TPoint::Norm(0.09, 0.0)],
                dashed: false,
                close: false,
            }],
        }
    }

    fn hline_edge(y: f32) -> Vec<Vec<(f32, f32)>> {
        vec![vec![(0.0, y), (400.0, y)]]
    }

    #[test]
    fn resample_spaces_points_evenly() {
        let pts = resample(&[(0.0, 0.0), (100.0, 0.0)], 4.0);
        assert_eq!(pts.len(), 26); // 0, 4, ..., 100
        assert_eq!(pts[0], (0.0, 0.0));
        assert!((pts[1].0 - 4.0).abs() < 1e-4);
        assert!((pts[25].0 - 100.0).abs() < 1e-4);
    }

    #[test]
    fn resample_carries_remainder_across_segments() {
        // Two 6px segments, 4px step: samples at 0, 4, 8, 12.
        let pts = resample(&[(0.0, 0.0), (6.0, 0.0), (12.0, 0.0)], 4.0);
        let xs: Vec<f32> = pts.iter().map(|p| p.0).collect();
        assert_eq!(xs, vec![0.0, 4.0, 8.0, 12.0]);
    }

    #[test]
    fn edge_hit_fraction_prefers_snapped_template() {
        let edges = EdgeMap::from_chains(&hline_edge(100.0), 400, 300);
        let t = hline_template();
        let snapped = Placement {
            ax: 200.0,
            ay: 100.0,
            w: 400.0,
            h: 200.0,
            mirror: false,
        };
        let offset = Placement {
            ay: 150.0,
            ..snapped
        };
        let s_snap = edge_hit_fraction(&render_template(&t, &snapped), &edges);
        let s_off = edge_hit_fraction(&render_template(&t, &offset), &edges);
        assert!(s_snap > 0.9, "snapped score {s_snap}");
        assert!(s_off < 0.1, "offset score {s_off}");
        assert!(s_snap > s_off);
    }

    #[test]
    fn edge_map_dilates_by_hit_radius() {
        // A single edge pixel at (50, 50); points 4px away still hit, 5px miss.
        let edges = EdgeMap::from_chains(&[vec![(50.0, 50.0)]], 200, 200);
        assert!(edges.is_edge(50.0, 50.0));
        assert!(edges.is_edge(54.0, 50.0));
        assert!(!edges.is_edge(55.0, 50.0));
        assert!(!edges.is_edge(10.0, 10.0));
    }

    #[test]
    fn search_pocket_y_snaps_to_synthetic_edge() {
        // Pocket-style line template; edge at y=120 across both flap sites.
        let edges = EdgeMap::from_chains(&hline_edge(120.0), 400, 300);
        let won = search_pocket_y(104.0, &hline_template(), 100.0, 300.0, 400.0, 200.0, &edges);
        assert!((won - 120.0).abs() < 1e-4, "won {won}, want 120");
    }

    #[test]
    fn search_pocket_y_falls_back_without_edges() {
        let edges = EdgeMap::from_chains(&[], 400, 300);
        let won = search_pocket_y(104.0, &hline_template(), 100.0, 300.0, 400.0, 200.0, &edges);
        assert!((won - 104.0).abs() < 1e-4, "won {won}, want default 104");
    }

    #[test]
    fn search_pocket_y_falls_back_on_weak_evidence() {
        // Edge 40px away: no candidate reaches it; the default must stand.
        let edges = EdgeMap::from_chains(&hline_edge(200.0), 400, 300);
        let won = search_pocket_y(104.0, &hline_template(), 100.0, 300.0, 400.0, 200.0, &edges);
        assert!((won - 104.0).abs() < 1e-4, "won {won}, want default 104");
    }

    #[test]
    fn search_lapel_falls_back_without_edges() {
        let edges = EdgeMap::from_chains(&[], 400, 300);
        let frame = Placement {
            ax: 200.0,
            ay: 50.0,
            w: 400.0,
            h: 500.0,
            mirror: false,
        };
        let (vg, w) = search_lapel(0.09, 0.4, &frame, &edges);
        assert!((vg - 0.09).abs() < 1e-6 && (w - 400.0).abs() < 1e-6);
    }
}
