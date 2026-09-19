//! Tiny MLP chain classifier distilled from Gemini vision labels.
//!
//! A 9 -> 16 -> 2 MLP (ReLU, softmax) trained on 265 Gemini-labeled chains
//! (blazer + jeans). It predicts P(structural) vs P(omit) from geometric and
//! photometric chain features. Used as a noise gate: high-confidence "omit"
//! chains are dropped; structural chains fall through to the heuristic
//! `classify_chain` for seam/stitch/fold styling.
//!
//! Weights are baked in from `mlp_weights.json` (regenerate via
//! `/tmp/distill/train_mlp.py`). Pure Rust, no dependencies beyond serde_json.

use serde::Deserialize;

const WEIGHTS_JSON: &str = include_str!("mlp_weights.json");

#[derive(Deserialize)]
struct MlpWeights {
    feature_mean: Vec<f32>,
    feature_std: Vec<f32>,
    w1: Vec<Vec<f32>>,
    b1: Vec<f32>,
    w2: Vec<Vec<f32>>,
    b2: Vec<f32>,
}

/// P(structural) for a chain from its 9 features.
///
/// Features (must match `train_mlp.py`):
/// [log1p(arc_len), straightness, mean_lum/255, bright_frac, edge_frac,
///  log(bw/bh), log1p(bw*bh)/20, centroid_y/1600, log1p(n_points)]
pub fn p_structural(features: &[f32; 9]) -> f32 {
    let w: MlpWeights = serde_json::from_str(WEIGHTS_JSON).expect("mlp_weights.json");
    // Standardize.
    let mut x = [0.0f32; 9];
    for i in 0..9 {
        x[i] = (features[i] - w.feature_mean[i]) / w.feature_std[i];
    }
    // Hidden: relu(x @ W1 + b1). W1 is [9][16] (input-major).
    let mut h = [0.0f32; 16];
    for (j, h_j) in h.iter_mut().enumerate() {
        let mut s = w.b1[j];
        for (i, x_i) in x.iter().enumerate() {
            s += x_i * w.w1[i][j];
        }
        *h_j = s.max(0.0);
    }
    // Logits: h @ W2 + b2. W2 is [16][2].
    let mut logit0 = w.b2[0];
    let mut logit1 = w.b2[1];
    for (j, h_j) in h.iter().enumerate() {
        logit0 += h_j * w.w2[j][0];
        logit1 += h_j * w.w2[j][1];
    }
    // Softmax P(class 0 = structural).
    let m = logit0.max(logit1);
    let e0 = (logit0 - m).exp();
    let e1 = (logit1 - m).exp();
    e0 / (e0 + e1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlp_loads_and_outputs_probability() {
        let f = [4.0, 0.9, 0.5, 0.3, 0.2, 0.0, 0.5, 0.5, 3.0];
        let p = p_structural(&f);
        assert!((0.0..=1.0).contains(&p), "p={p} not in [0,1]");
    }

    #[test]
    fn mlp_prefers_long_straight_over_tiny_fragment() {
        // Long straight chain should score more structural than a tiny speck.
        let long = [7.0, 0.95, 0.45, 0.1, 0.1, 0.0, 0.8, 0.5, 5.0];
        let tiny = [1.0, 0.5, 0.5, 0.1, 0.0, 0.0, 0.1, 0.5, 1.5];
        assert!(
            p_structural(&long) > p_structural(&tiny),
            "long={} tiny={}",
            p_structural(&long),
            p_structural(&tiny)
        );
    }
}
