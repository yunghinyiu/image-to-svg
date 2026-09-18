//! #28: Discrete template parameter representation for structural linework.
//!
//! A [`Template`] captures *structure* (what to draw) as paths in normalized
//! coordinates; a [`Placement`] captures *geometry* (where/how big) as a
//! discrete frame mapping template space to absolute pixels. [`render_template`]
//! maps one through the other into absolute SVG paths.
//!
//! Rationale (cf. OmniSVG, NeurIPS 2025): decoupling structural logic from
//! low-level geometry makes placement *searchable* (#27: generate candidate
//! placements, score against photo edges, keep the best) and *detectable*
//! (#29: photo edge detection outputs a [`Placement`]), and gives any future
//! learning/distillation a discrete parameter space to predict.
//!
//! Coordinate conventions: template `u` is a signed offset from the frame's
//! horizontal anchor in units of frame width `w`; template `v` is a fraction
//! of frame height `h` below the frame top `ay`. Absolute pixel insets that
//! are defined in px rather than proportions (stitching insets) use
//! [`TPoint::NormPx`].
//!
//! Quantization: [`Placement::quantized`] snaps the anchor to an 8px grid
//! ([`PLACEMENT_GRID_PX`]). Rationale: XDoG edge evidence is noisy at
//! single-pixel resolution, so sub-grid placement differences are not
//! resolvable from the photo, and the grid keeps the #27 search space finite.
//! The default conversion path uses exact (unquantized) placements, so this
//! refactor is behavior-preserving.

/// Anchor quantization grid in px for placement search (#27).
/// (Marked allow(dead_code): the search lands in #27; unit-tested below.)
#[allow(dead_code)]
pub const PLACEMENT_GRID_PX: f32 = 8.0;

/// A point in template space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TPoint {
    /// Normalized `(u, v)`: maps to `(ax + s*u*w, ay + v*h)` where `s` is
    /// +1 or -1 from [`Placement::mirror`].
    Norm(f32, f32),
    /// Normalized `(u, v)` plus an absolute pixel offset `(dx, dy)` applied
    /// after mapping. `dx` is mirrored with `u` (stitching insets point
    /// inward regardless of side); `dy` is not.
    NormPx(f32, f32, f32, f32),
}

/// One template path: a polyline in template space plus style flags.
/// Curves are pre-sampled into polylines by the template builders (the
/// pipeline emits polyline paths everywhere).
#[derive(Clone, Debug)]
pub struct TPath {
    pub points: Vec<TPoint>,
    pub dashed: bool,
    /// Append the first point at render time to explicitly close the loop.
    pub close: bool,
}

impl TPath {
    fn solid(points: Vec<TPoint>) -> Self {
        TPath {
            points,
            dashed: false,
            close: false,
        }
    }
    fn dashed(points: Vec<TPoint>) -> Self {
        TPath {
            points,
            dashed: true,
            close: false,
        }
    }
    fn closed(points: Vec<TPoint>, dashed: bool) -> Self {
        TPath {
            points,
            dashed,
            close: true,
        }
    }
}

/// A named structural template: pure structure in normalized coordinates,
/// no absolute pixels (except [`TPoint::NormPx`] stitching insets).
/// (The name is for debugging / future #27 logging; allow(dead_code).)
#[derive(Clone, Debug)]
pub struct Template {
    #[allow(dead_code)]
    pub name: &'static str,
    pub paths: Vec<TPath>,
}

/// Discrete geometric placement: maps template space to absolute coords.
///
/// The frame is `(ax, ay, w, h)` with an optional horizontal mirror about
/// the anchor `ax`. For garment views the anchor is the view's horizontal
/// center and `(w, h)` is the view size, so `u` is a signed fraction of view
/// width from center and `v` is a fraction of view height from the top.
/// (A single uniform `scale` would not capture real templates: lapel widths
/// scale with view width while landmark rows scale with view height.)
#[derive(Clone, Copy, Debug)]
pub struct Placement {
    pub ax: f32,
    pub ay: f32,
    pub w: f32,
    pub h: f32,
    pub mirror: bool,
}

impl Placement {
    /// Snap the anchor to the quantization grid; for #27 candidate search.
    /// Scale/mirror are already discrete by construction.
    /// (Marked allow(dead_code): the search lands in #27; unit-tested below.)
    #[allow(dead_code)]
    pub fn quantized(self) -> Self {
        let q = |v: f32| (v / PLACEMENT_GRID_PX).round() * PLACEMENT_GRID_PX;
        Placement {
            ax: q(self.ax),
            ay: q(self.ay),
            ..self
        }
    }

    /// Map a template point to absolute pixel coordinates.
    pub fn map_point(&self, p: TPoint) -> (f32, f32) {
        let (u, v, dx, dy) = match p {
            TPoint::Norm(u, v) => (u, v, 0.0, 0.0),
            TPoint::NormPx(u, v, dx, dy) => (u, v, dx, dy),
        };
        let s = if self.mirror { -1.0 } else { 1.0 };
        (self.ax + s * (u * self.w + dx), self.ay + v * self.h + dy)
    }
}

/// A rendered template path: absolute pixel polyline plus style.
#[derive(Clone, Debug)]
pub struct RenderedPath {
    pub points: Vec<(f32, f32)>,
    pub dashed: bool,
}

/// Render a template through a placement into absolute paths, preserving
/// template path order.
pub fn render_template(t: &Template, p: &Placement) -> Vec<RenderedPath> {
    t.paths
        .iter()
        .map(|tp| {
            let mut points: Vec<(f32, f32)> = tp.points.iter().map(|&pt| p.map_point(pt)).collect();
            if tp.close {
                if let Some(&p0) = points.first() {
                    points.push(p0);
                }
            }
            RenderedPath {
                points,
                dashed: tp.dashed,
            }
        })
        .collect()
}

/// Sample a cubic Bezier in (u, v) template space into `n` segments.
fn sample_cubic_uv(
    p0: (f32, f32),
    p1: (f32, f32),
    p2: (f32, f32),
    p3: (f32, f32),
    n: usize,
) -> Vec<TPoint> {
    (0..=n)
        .map(|i| {
            let t = i as f32 / n as f32;
            let u = 1.0 - t;
            TPoint::Norm(
                u * u * u * p0.0
                    + 3.0 * u * u * t * p1.0
                    + 3.0 * u * t * t * p2.0
                    + t * t * t * p3.0,
                u * u * u * p0.1
                    + 3.0 * u * u * t * p1.1
                    + 3.0 * u * t * t * p2.1
                    + t * t * t * p3.1,
            )
        })
        .collect()
}

/// Rounded rectangle in (u, v) template space, possibly elliptical
/// (`ru`, `rv` separate) since `u`/`v` units differ in pixels.
fn rounded_rect_uv(
    u0: f32,
    v0: f32,
    uw: f32,
    vh: f32,
    ru: f32,
    rv: f32,
    seg: usize,
) -> Vec<TPoint> {
    let mut pts = Vec::new();
    // (center_u, center_v, start_angle_deg): TR, BR, BL, TL.
    let corners = [
        (u0 + uw - ru, v0 + rv, -90.0),
        (u0 + uw - ru, v0 + vh - rv, 0.0),
        (u0 + ru, v0 + vh - rv, 90.0),
        (u0 + ru, v0 + rv, 180.0),
    ];
    for (ccu, ccv, start_deg) in corners {
        for i in 0..=seg {
            let ang = (start_deg + i as f32 * 90.0 / seg as f32).to_radians();
            pts.push(TPoint::Norm(ccu + ru * ang.cos(), ccv + rv * ang.sin()));
        }
    }
    pts
}

/// Canonical right-side notch lapel; mirror the placement for the left side.
///
/// `vg` / `vb`: gorge and top-button landmark rows as fractions of frame
/// height (from button detection). Proportions transcribed from the
/// reference tech pack: notch at 0.14w, peak at 0.25w / 0.24h, break at
/// 0.09w; dashed topstitching inset 9px; roll line from neck to button.
pub fn lapel_template(vg: f32, _vb: f32) -> Template {
    // Scale down: use compact vb (not detector's tall vb).
    // Target lapel is ~0.15h tall, not 0.32h.
    // Peak at 0.26 (further out than notch_outer 0.16) for the jut.
    let vb_compact = vg + 0.15f32;
    lapel_template_with_peak(vg, vb_compact, 0.26f32)
}

/// Lapel template with explicit peak x position (for curvature-snapped placement).
/// `peak_x` is the normalized x offset from center (0.30 = wide angular lapel).
pub fn lapel_template_with_peak(vg: f32, _vb: f32, peak_x: f32) -> Template {
    // Notched lapel geometry (target-measured):
    // - Gorge: where collar meets lapel at center front
    // - Notch: V-shaped cutout between collar and lapel (the "step")
    // - Peak: outermost point, juts OUTWARD from the notch
    // - Break: where lapel meets the front edge at button level
    //
    // The target shows a distinct notch step, not a smooth V. The peak
    // points outward (horizontally), not downward.
    // Notched lapel with SHARP BREAK at collar-lapel junction (Jev conf 1.0).
    // The break is a distinct angular corner where the collar ends and the
    // lapel begins. The lapel's top edge is STRAIGHT from break to peak.
    // Target shows a pronounced step, not a subtle one.
    // PEAK MUST BE FURTHER OUT than notch_outer to create the jut.
    let break_pt = (0.08f32, vg); // BREAK: collar ends here
                                  // Notch: HORIZONTAL step outward (same Y as break for sharp 90° corner).
    let notch_outer = (0.16f32, vg); // Step outward, NO vertical drop
                                     // Peak: widest point, just below notch (shallow top edge).
    let peak = (peak_x, vg + 0.06f32); // Peak close to notch (shallow angle)
                                       // brk: where lapel meets front edge. NOT at button (vb) — above it.
                                       // Target lapel height ~0.35 (vg=0.08 to brk=0.43).
    let brk = (0.10f32, vg + 0.35f32); // Meets front edge above button
                                       // (Old cubic curve rendered a rounded shield; removed per target.)
    Template {
        name: "lapel",
        paths: vec![
            // BREAK: sharp corner where collar ends, lapel begins.
            // Horizontal step outward (pronounced, not subtle).
            TPath::solid(vec![
                TPoint::Norm(break_pt.0, break_pt.1),
                TPoint::Norm(notch_outer.0, notch_outer.1),
            ]),
            // Lapel top edge: STRAIGHT from break to peak (sharp angle at break).
            TPath::solid(vec![
                TPoint::Norm(notch_outer.0, notch_outer.1),
                TPoint::Norm(peak.0, peak.1),
            ]),
            // Lapel outer edge: peak -> break (straight, angular).
            TPath::solid(vec![
                TPoint::Norm(peak.0, peak.1),
                TPoint::Norm(brk.0, brk.1),
            ]),
            // Dashed topstitching parallel to lapel edges, inset 9px.
            TPath::dashed(vec![
                TPoint::NormPx(notch_outer.0, notch_outer.1, -9.0, 2.0),
                TPoint::NormPx(peak.0, peak.1, -9.0, 0.0),
            ]),
            TPath::dashed(vec![
                TPoint::NormPx(peak.0, peak.1, -9.0, 0.0),
                TPoint::NormPx(brk.0, brk.1, -9.0, 0.0),
            ]),
            // Roll line: break -> brk (the V opening, straight).
            TPath::solid(vec![
                TPoint::Norm(break_pt.0, break_pt.1),
                TPoint::Norm(brk.0, brk.1),
            ]),
        ],
    }
}

/// Gorge seam: collar bottom edge from notch to notch through center.
/// Drawn once (not mirrored).
pub fn gorge_seam_template(vg: f32) -> Template {
    Template {
        name: "gorge-seam",
        paths: vec![TPath::solid(vec![
            TPoint::Norm(-0.14, vg),
            TPoint::Norm(0.0, vg - 0.008),
            TPoint::Norm(0.14, vg),
        ])],
    }
}

/// Pocket flap: rounded rect + dashed inset topstitching, anchored at the frame
/// `(ax, ay)` = `(pocket_center_x, pocket_bottom_y)`.
///
/// Bottom-anchored because the detector (`detect::detect_pocket_y`) keys on the
/// flap's strong bottom edge — the underlay check (photo underneath, trace from
/// it) showed a top-anchored placement draws the flap one flap-height too low.
/// `w`/`h` (frame size in px) are needed for the corner radius, which is
/// circular in pixel space (0.012*w), hence elliptical in (u,v) space.
pub fn pocket_template(w: f32, h: f32) -> Template {
    let (hw, hh) = (0.09f32, 0.06f32); // half-width 0.09w, height 0.06h
    let ru = 0.012f32;
    let rv = 0.012 * w / h;
    let flap = rounded_rect_uv(-hw, -hh, 2.0 * hw, hh, ru, rv, 10);
    // Dashed stitching inset 5px.
    let iu = 5.0 / w;
    let iv = 5.0 / h;
    let riu = 0.008f32;
    let riv = 0.008 * w / h;
    let stitch = rounded_rect_uv(
        -hw + iu,
        -hh + iv,
        2.0 * (hw - iu),
        hh - 2.0 * iv,
        riu,
        riv,
        10,
    );
    Template {
        name: "pocket",
        paths: vec![TPath::closed(flap, false), TPath::closed(stitch, true)],
    }
}

/// Front view: collar band between the lapel notches. The target draws it as
/// a flat band with a topstitched fall edge and a small center label/hanger.
/// `vg` is the gorge (notch) row from the detector; photo measurement shows
/// the actual notch 22px above the detector's y, so we apply that correction.
/// The collar stands 29px above the corrected notch (photo-measured).
/// Spans +/-0.14w to meet the lapel notches exactly.
pub fn front_collar_template(vg: f32) -> Template {
    // Photo evidence (pipeline coords): notch y=156, top y=127.
    // Detector vg maps to y=178; correct by -22px, height 29px.
    // With h~740: 22/740=0.030, 29/740=0.039.
    let vg_corr = vg - 0.030;
    let v_top = vg_corr - 0.039;
    let hw = 0.14f32;
    Template {
        name: "front-collar",
        paths: vec![
            // Collar fall (top edge), slight upward arc at center.
            TPath::solid(vec![
                TPoint::Norm(-hw, v_top + 0.004),
                TPoint::Norm(0.0, v_top),
                TPoint::Norm(hw, v_top + 0.004),
            ]),
            // Collar bottom (neckline seam) — meets the gorge seam.
            TPath::solid(vec![
                TPoint::Norm(-hw, vg_corr),
                TPoint::Norm(0.0, vg_corr - 0.004),
                TPoint::Norm(hw, vg_corr),
            ]),
            // Collar ends (at the notches).
            TPath::solid(vec![
                TPoint::Norm(-hw, v_top + 0.004),
                TPoint::Norm(-hw, vg_corr),
            ]),
            TPath::solid(vec![
                TPoint::Norm(hw, v_top + 0.004),
                TPoint::Norm(hw, vg_corr),
            ]),
            // Dashed topstitching below the fall edge.
            TPath::dashed(vec![
                TPoint::NormPx(-hw, v_top + 0.004, 3.0, 5.0),
                TPoint::NormPx(0.0, v_top, 0.0, 5.0),
                TPoint::NormPx(hw, v_top + 0.004, -3.0, 5.0),
            ]),
            // Center label/hanger: small rect on the neckline.
            TPath::solid(vec![
                TPoint::NormPx(-0.025, vg_corr - 0.004, 0.0, -2.0),
                TPoint::NormPx(0.025, vg_corr - 0.004, 0.0, -2.0),
                TPoint::NormPx(0.025, vg_corr - 0.004, 0.0, 6.0),
                TPoint::NormPx(-0.025, vg_corr - 0.004, 0.0, 6.0),
                TPoint::NormPx(-0.025, vg_corr - 0.004, 0.0, -2.0),
            ]),
        ],
    }
}

/// Back view: collar band (top/bottom edges + sides), dashed topstitching
/// along the collar bottom, and the dashed center back seam.
pub fn back_collar_template() -> Template {
    // #36: collar bottom extended to v=0.13 (was 0.09) so it overlaps the
    // back silhouette instead of floating above it with a gap. The collar
    // sits ON the back; drawn after the silhouette it renders on top.
    // Widened 0.15->0.17 to match the target's broad flat collar.
    Template {
        name: "back-collar",
        paths: vec![
            // Collar top edge.
            TPath::solid(vec![
                TPoint::Norm(-0.17, 0.02 + 0.006),
                TPoint::Norm(0.0, 0.02),
                TPoint::Norm(0.17, 0.02 + 0.006),
            ]),
            // Collar bottom edge (gorge seam).
            TPath::solid(vec![
                TPoint::Norm(-0.17 * 1.05, 0.13),
                TPoint::Norm(0.0, 0.13 - 0.006),
                TPoint::Norm(0.17 * 1.05, 0.13),
            ]),
            // Collar sides.
            TPath::solid(vec![
                TPoint::Norm(-0.17, 0.02 + 0.006),
                TPoint::Norm(-0.17 * 1.05, 0.13),
            ]),
            TPath::solid(vec![
                TPoint::Norm(0.17, 0.02 + 0.006),
                TPoint::Norm(0.17 * 1.05, 0.13),
            ]),
            // Dashed topstitching along collar bottom.
            TPath::dashed(vec![
                TPoint::NormPx(-0.17 * 1.05, 0.13, 4.0, -5.0),
                TPoint::NormPx(0.0, 0.13 - 0.006, 0.0, -5.0),
                TPoint::NormPx(0.17 * 1.05, 0.13, -4.0, -5.0),
            ]),
            // Center back seam (dashed).
            TPath::dashed(vec![
                TPoint::NormPx(0.0, 0.13, 0.0, 8.0),
                TPoint::Norm(0.0, 0.95),
            ]),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placement_maps_norm_to_absolute() {
        let p = Placement {
            ax: 100.0,
            ay: 50.0,
            w: 200.0,
            h: 400.0,
            mirror: false,
        };
        assert_eq!(p.map_point(TPoint::Norm(0.25, 0.5)), (150.0, 250.0));
        // Mirror flips u about the anchor.
        let m = Placement { mirror: true, ..p };
        assert_eq!(m.map_point(TPoint::Norm(0.25, 0.5)), (50.0, 250.0));
    }

    #[test]
    fn placement_mirrors_px_insets_inward() {
        // A -9px stitching inset on the right side becomes +9px when mirrored,
        // so insets always point toward the frame center.
        let p = Placement {
            ax: 100.0,
            ay: 0.0,
            w: 200.0,
            h: 100.0,
            mirror: false,
        };
        let m = Placement { mirror: true, ..p };
        assert_eq!(
            p.map_point(TPoint::NormPx(0.14, 0.1, -9.0, 2.0)),
            (100.0 + 0.14 * 200.0 - 9.0, 12.0)
        );
        assert_eq!(
            m.map_point(TPoint::NormPx(0.14, 0.1, -9.0, 2.0)),
            (100.0 - (0.14 * 200.0 - 9.0), 12.0)
        );
    }

    #[test]
    fn placement_quantized_snaps_to_grid() {
        let p = Placement {
            ax: 103.0,
            ay: 207.0,
            w: 200.0,
            h: 400.0,
            mirror: false,
        }
        .quantized();
        assert_eq!((p.ax, p.ay), (104.0, 208.0));
        // Scale and mirror are untouched (already discrete).
        assert_eq!((p.w, p.h, p.mirror), (200.0, 400.0, false));
    }

    #[test]
    fn render_template_preserves_path_order_and_style() {
        let t = Template {
            name: "test",
            paths: vec![
                TPath::solid(vec![TPoint::Norm(0.0, 0.0), TPoint::Norm(1.0, 1.0)]),
                TPath::dashed(vec![TPoint::Norm(0.5, 0.0)]),
            ],
        };
        let p = Placement {
            ax: 10.0,
            ay: 20.0,
            w: 100.0,
            h: 50.0,
            mirror: false,
        };
        let out = render_template(&t, &p);
        assert_eq!(out.len(), 2);
        assert!(!out[0].dashed);
        assert_eq!(out[0].points, vec![(10.0, 20.0), (110.0, 70.0)]);
        assert!(out[1].dashed);
        assert_eq!(out[1].points, vec![(60.0, 20.0)]);
    }

    #[test]
    fn render_template_closes_loops() {
        let t = Template {
            name: "test",
            paths: vec![TPath::closed(
                vec![
                    TPoint::Norm(0.0, 0.0),
                    TPoint::Norm(1.0, 0.0),
                    TPoint::Norm(1.0, 1.0),
                ],
                false,
            )],
        };
        let p = Placement {
            ax: 0.0,
            ay: 0.0,
            w: 10.0,
            h: 10.0,
            mirror: false,
        };
        let out = render_template(&t, &p);
        assert_eq!(out[0].points.len(), 4);
        assert_eq!(out[0].points[0], out[0].points[3]);
    }

    #[test]
    fn lapel_template_has_expected_path_count() {
        // Notched lapel with break: 1 break step + 2 solid lapel edges
        // + 2 dashed stitching + 1 solid roll = 6 paths.
        let t = lapel_template(0.09, 0.4);
        assert_eq!(t.paths.len(), 6);
        assert_eq!(t.paths.iter().filter(|p| p.dashed).count(), 2);
        // All straight lines have 2 points each.
        for p in &t.paths {
            assert_eq!(p.points.len(), 2);
        }
    }

    #[test]
    fn sample_cubic_uv_endpoints_and_count() {
        let pts = sample_cubic_uv((0.0, 0.0), (0.3, 0.0), (0.3, 0.5), (1.0, 0.5), 10);
        assert_eq!(pts.len(), 11);
        assert_eq!(pts[0], TPoint::Norm(0.0, 0.0));
        assert_eq!(pts[10], TPoint::Norm(1.0, 0.5));
        // Monotonic in u for this curve.
        let us: Vec<f32> = pts
            .iter()
            .map(|p| match p {
                TPoint::Norm(u, _) => *u,
                _ => panic!("expected Norm"),
            })
            .collect();
        for w in us.windows(2) {
            assert!(w[1] >= w[0]);
        }
    }

    #[test]
    fn rounded_rect_uv_stays_in_bounds() {
        let pts = rounded_rect_uv(0.0, 0.0, 0.18, 0.06, 0.012, 0.02, 4);
        // 4 corners * (4+1) points.
        assert_eq!(pts.len(), 20);
        for p in &pts {
            match p {
                TPoint::Norm(u, v) => {
                    assert!((0.0..=0.18).contains(u) && (0.0..=0.06).contains(v));
                }
                _ => panic!("expected Norm"),
            }
        }
    }

    #[test]
    fn pocket_template_renders_closed_loops() {
        let t = pocket_template(500.0, 600.0);
        assert_eq!(t.paths.len(), 2);
        assert!(!t.paths[0].dashed && t.paths[0].close);
        assert!(t.paths[1].dashed && t.paths[1].close);
        let p = Placement {
            ax: 250.0,
            ay: 400.0,
            w: 500.0,
            h: 600.0,
            mirror: false,
        };
        let out = render_template(&t, &p);
        assert_eq!(out.len(), 2);
        // Flap spans 0.18w centered on the anchor.
        let xs: Vec<f32> = out[0].points.iter().map(|&(x, _)| x).collect();
        let (lo, hi) = (
            xs.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
            xs.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
        );
        assert!((lo - (250.0 - 0.09 * 500.0)).abs() < 0.5, "lo={lo}");
        assert!((hi - (250.0 + 0.09 * 500.0)).abs() < 0.5, "hi={hi}");
    }
}
