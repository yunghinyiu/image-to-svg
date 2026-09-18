//! Shape-from-shading prototype: estimate surface normals from luminance.
//!
//! The idea: pixel brightness encodes surface orientation (Lambertian
//! shading). By analyzing luminance gradients we recover a normal map,
//! then curvature from the normal field reveals folds, seams, and the
//! true 3D structure that 2D edge detection misses.
//!
//! This is a prototype to validate the approach before integrating
//! into the pipeline. Jev will judge whether the detected structures
//! are correct.

use image::{GrayImage, RgbImage};

/// Estimate surface normals from a luminance image.
///
/// Simplified shape-from-shading:
/// - Assumes Lambertian reflectance, light from top-front
/// - Normal x,y proportional to negative luminance gradient
/// - Normal z = 1 (facing camera), normalized
///
/// Returns (nx, ny, nz) per pixel as f32.
pub fn estimate_normals(lum: &GrayImage) -> Vec<(f32, f32, f32)> {
    let (w, h) = lum.dimensions();
    let w = w as usize;
    let h = h as usize;
    let mut normals = Vec::with_capacity(w * h);

    // Sobel kernels for gradient estimation
    for y in 0..h {
        for x in 0..w {
            // Clamped sampling for borders
            let xm = x.saturating_sub(1);
            let xp = (x + 1).min(w - 1);
            let ym = y.saturating_sub(1);
            let yp = (y + 1).min(h - 1);

            let get = |xx: usize, yy: usize| -> f32 {
                lum.get_pixel(xx as u32, yy as u32)[0] as f32 / 255.0
            };

            // Sobel X
            let gx = (get(xp, ym) + 2.0 * get(xp, y) + get(xp, yp)
                - get(xm, ym)
                - 2.0 * get(xm, y)
                - get(xm, yp))
                / 8.0;
            // Sobel Y
            let gy = (get(xm, yp) + 2.0 * get(x, yp) + get(xp, yp)
                - get(xm, ym)
                - 2.0 * get(x, ym)
                - get(xp, ym))
                / 8.0;

            // Surface slopes away from brightness gradient.
            // Scale factor k controls sensitivity; tuned empirically.
            let k = 2.0;
            let nx = -gx * k;
            let ny = -gy * k;
            let nz = 1.0;
            let len = (nx * nx + ny * ny + nz * nz).sqrt();
            normals.push((nx / len, ny / len, nz / len));
        }
    }
    normals
}

/// Compute curvature (fold strength) from a normal map.
///
/// High curvature = rapid normal change = fold, crease, or seam.
/// Returns per-pixel curvature magnitude.
pub fn curvature_from_normals(normals: &[(f32, f32, f32)], w: usize, h: usize) -> Vec<f32> {
    let mut curv = vec![0.0f32; w * h];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            let (nx, ny, nz) = normals[i];
            // Normal variation in x and y directions
            let (nx_r, ny_r, nz_r) = normals[i + 1];
            let (nx_d, ny_d, nz_d) = normals[i + w];
            let dx = ((nx_r - nx).powi(2) + (ny_r - ny).powi(2) + (nz_r - nz).powi(2)).sqrt();
            let dy = ((nx_d - nx).powi(2) + (ny_d - ny).powi(2) + (nz_d - nz).powi(2)).sqrt();
            curv[i] = dx + dy;
        }
    }
    curv
}

/// Render a normal map as an RGB image for visualization.
/// Maps (nx, ny, nz) in [-1,1] to [0,255].
pub fn render_normal_map(normals: &[(f32, f32, f32)], w: u32, h: u32) -> RgbImage {
    let mut img = RgbImage::new(w, h);
    for (i, &(nx, ny, nz)) in normals.iter().enumerate() {
        let x = (i as u32) % w;
        let y = (i as u32) / w;
        img.put_pixel(
            x,
            y,
            image::Rgb([
                ((nx * 0.5 + 0.5) * 255.0) as u8,
                ((ny * 0.5 + 0.5) * 255.0) as u8,
                (nz * 255.0) as u8,
            ]),
        );
    }
    img
}

/// Render curvature as a grayscale heatmap.
pub fn render_curvature(curv: &[f32], w: u32, h: u32) -> GrayImage {
    let max_c = curv.iter().cloned().fold(0.0f32, f32::max).max(1e-6);
    let mut img = GrayImage::new(w, h);
    for (i, &c) in curv.iter().enumerate() {
        let x = (i as u32) % w;
        let y = (i as u32) / w;
        // Square root for better dynamic range
        let v = ((c / max_c).sqrt() * 255.0) as u8;
        img.put_pixel(x, y, image::Luma([v]));
    }
    img
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normals_face_camera_on_flat() {
        // Uniform luminance -> zero gradient -> normals face camera
        let img = GrayImage::from_pixel(10, 10, image::Luma([128u8]));
        let normals = estimate_normals(&img);
        for &(nx, ny, nz) in &normals {
            assert!(nx.abs() < 1e-6);
            assert!(ny.abs() < 1e-6);
            assert!((nz - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn curvature_zero_on_flat() {
        let img = GrayImage::from_pixel(10, 10, image::Luma([128u8]));
        let normals = estimate_normals(&img);
        let curv = curvature_from_normals(&normals, 10, 10);
        for &c in &curv {
            assert!(c < 1e-6);
        }
    }
}
