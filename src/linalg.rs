use std::simd::f32x4;

use crate::dataset::Dataset;

#[inline]
pub fn l2_distance(s1: &[f32], s2: &[f32]) -> f32 {
    f32::sqrt(
        s1.iter()
            .zip(s2.iter())
            .map(|(i, j)| f32::powi(i - j, 2))
            .sum(),
    )
}
// (X-Y)**2 element wise
#[allow(dead_code)]
#[inline]
pub fn l2_distance_simd(xs: &Vec<f32>, ys: &Vec<f32>) -> f32 {
    assert_eq!(xs.len(), ys.len());
    let size = xs.len();
    let chunks: usize = size / 4;
    let mut simd_res: Vec<f32> = vec![0.0; size];

    // todo ; compute the leftover elements
    for i in 3 * chunks..size {
        simd_res[i] = (xs[i] - ys[i]).powi(2);
    }

    let simd_px = xs.as_ptr() as *const f32x4;
    let simd_py = ys.as_ptr() as *const f32x4;
    let simd_r: *mut std::simd::Simd<f32, 4> = simd_res.as_mut_ptr() as *mut f32x4;
    for i in 0..chunks {
        let i = i as isize;
        unsafe {
            let simd_sub = *simd_px.offset(i) - *simd_py.offset(i);
            *simd_r.offset(i) = simd_sub * simd_sub;
        }
    }
    f32::sqrt(simd_res.iter().sum())
}

pub fn compute_centroid(samples: &Vec<&[f32]>) -> Vec<f32> {
    let mut centroid: Vec<f32> = Vec::new();
    // todo : figureout how to idiomatically return
    if samples.is_empty() {
        return centroid;
    }
    // todo : how to get dimension from sample ??
    for i in 0..samples[0].len() {
        let mut temp: f32 = 0.0;
        for sample in samples {
            temp += sample[i];
        }
        centroid.push(temp as f32 / samples.len() as f32);
    }
    centroid
}

mod tests {
    use super::*;

    #[test]
    fn test_l2_distance() {
        let s1: Vec<f32> = vec![0.43835984, -0.11517563, 1.22428137, 0.72168846, -0.89573074];
        assert_eq!(l2_distance(&s1, &s1.clone()), 0.0);

        let s1: Vec<f32> = vec![1.0, 0.0];
        let s2: Vec<f32> = vec![0.0, 0.0];
        assert_eq!(l2_distance(&s1, &s2), 1.0);

        let s1: Vec<f32> = vec![-1.0, 0.0];
        let s2: Vec<f32> = vec![0.0, 0.0];
        assert_eq!(l2_distance(&s1, &s2), 1.0);

        let s1: Vec<f32> = vec![0.12967037, 0.11381539, 0.48476367, -0.51196512, 0.11607633];
        let s2: Vec<f32> = vec![-1.76968614, 0.62421, 1.0612931, -0.283521, 1.25854718];
        let epsilon = 0.000001;
        assert!((l2_distance(&s1, &s2) - 2.3575135930492666).abs() < epsilon);
    }

    #[test]
    fn test_l2_distance_simd() {
        let s1: Vec<f32> = vec![0.43835984, -0.11517563, 1.22428137, 0.72168846, -0.89573074];
        assert_eq!(l2_distance_simd(&s1, &s1.clone()), 0.0);

        let s1: Vec<f32> = vec![1.0, 0.0];
        let s2: Vec<f32> = vec![0.0, 0.0];
        assert_eq!(l2_distance_simd(&s1, &s2), 1.0);

        let s1: Vec<f32> = vec![-1.0, 0.0];
        let s2: Vec<f32> = vec![0.0, 0.0];
        assert_eq!(l2_distance_simd(&s1, &s2), 1.0);

        let s1: Vec<f32> = vec![0.12967037, 0.11381539, 0.48476367, -0.51196512, 0.11607633];
        let s2: Vec<f32> = vec![-1.76968614, 0.62421, 1.0612931, -0.283521, 1.25854718];
        let epsilon = 0.000001;
        assert!((l2_distance_simd(&s1, &s2) - 2.3575135930492666).abs() < epsilon);
    }

    #[test]
    fn test_compute_centroid() {
        let mut samples: Vec<&[f32]> = Vec::new();
        let sample: [f32; 2] = [1.0, 1.0];
        for _ in 0..10 {
            samples.push(&sample);
        }
        let c = compute_centroid(&samples);

        dbg!("{:?}", &c);
        assert!(c == sample)
    }
    #[test]
    fn test_compute_centroid_complex() {
        let mut samples: Vec<&[f32]> = Vec::new();
        let s1: [f32; 2] = [1.0, 0.0];
        let s2: [f32; 2] = [0.0, 1.0];
        samples.push(&s1);
        samples.push(&s2);
        let c = compute_centroid(&samples);

        dbg!("{:?}", &c);
        assert!(c == vec![0.5, 0.5])
    }
}
