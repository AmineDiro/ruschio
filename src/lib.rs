use rand::Rng;
use std::fs::File;
use std::io::{BufReader, Read};

pub trait AsBytes {
    fn from_bytes(bytes: [u8; 4]) -> Self;
}

impl AsBytes for f32 {
    fn from_bytes(bytes: [u8; 4]) -> Self {
        f32::from_le_bytes(bytes)
    }
}

pub fn read_ndarray<T: AsBytes>(file_path: &str, size: usize) -> Option<Vec<T>> {
    let mut data: Vec<T> = Vec::new();
    let f = File::open(file_path).expect("Failed to read file.");
    let mut reader = BufReader::new(f);
    let mut buffer = [0u8; std::mem::size_of::<f32>()];
    loop {
        if let Err(_error) = reader.read_exact(&mut buffer) {
            break;
        }
        data.push(T::from_bytes(buffer))
    }
    if data.len() != size {
        return None;
    }
    Some(data)
}

pub type Sample = Vec<f32>;

#[derive(Debug)]
pub struct Dataset {
    samples: Vec<Sample>,
}

impl Dataset {
    pub fn build(data: Vec<f32>, n_features: usize) -> Self {
        let samples: Vec<Sample> = data
            .chunks(n_features)
            .into_iter()
            .map(|item| item.to_owned())
            .collect();

        Dataset { samples }
    }
}

pub fn l2_distance(s1: &Sample, s2: &Sample) -> f32 {
    f32::sqrt(
        s1.iter()
            .zip(s2.iter())
            .map(|(i, j)| f32::powi(i - j, 2))
            .sum(),
    )
}

pub fn compute_centroid(samples: &Vec<&Sample>) -> Sample {
    let mut centroid: Sample = Vec::new();
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

#[derive(Debug)]
pub struct Kmeans {
    // number of clusters
    n_clusters: usize,
    max_iterations: usize,
}

impl Kmeans {
    pub fn new(n_clusters: usize, max_iterations: Option<usize>) -> Self {
        let max_iterations = if max_iterations.is_none() {
            1000
        } else {
            max_iterations.unwrap()
        };

        Kmeans {
            n_clusters,
            max_iterations: max_iterations,
        }
    }

    pub fn fit(&self, dataset: &Dataset) -> (Vec<Sample>, Vec<usize>) {
        // Pick random K vectors from the dataset
        let mut rng = rand::thread_rng();
        let mut centroids: Vec<Sample> = Vec::with_capacity(self.n_clusters);
        for _k in 0..self.n_clusters {
            let n = rng.gen_range(0..dataset.samples.len());
            centroids.push(dataset.samples[n].clone());
        }

        // FOR MAX_ITER
        let mut pred_classes: Vec<usize> = vec![0; dataset.samples.len()];
        for iter in 0..self.max_iterations {
            // loop over the dataset
            // compute distances to centroids
            // Find minimum distance
            // Assign datapoint to centroids
            let mut assigment: Vec<usize> = vec![0; dataset.samples.len()];
            for (idx, sample) in dataset.samples.iter().enumerate() {
                let (class, _distance) = centroids
                    .iter()
                    .map(|c| l2_distance(c, sample))
                    .enumerate()
                    .min_by(|&(_, i), &(_, j)| i.total_cmp(&j))
                    .unwrap();

                assigment[idx] = class;
            }
            // Stopping criterion : if pred_classes didn't change
            if assigment != pred_classes {
                pred_classes = assigment;
                // recompute cluster centroids
                for k in 0..self.n_clusters {
                    let filtered_samples: Vec<&Sample> = pred_classes
                        .iter()
                        .zip(dataset.samples.iter())
                        .filter(|&(class, _)| *class == k)
                        .map(|(_, item)| item)
                        .collect();
                    centroids[k] = compute_centroid(&filtered_samples);
                }
            } else {
                println!("Converged after {} iterations.", iter);
                break;
            }
        }
        (centroids, pred_classes)
    }
}

pub fn evaluate(predicted: &Vec<usize>, truth: &Vec<usize>) -> f32 {
    let correct: Vec<usize> = predicted
        .iter()
        .zip(truth.iter())
        .filter(|(predicted, truth)| **predicted == **truth)
        .map(|(_, e)| *e)
        .collect();
    correct.len() as f32 / truth.len() as f32
}

#[cfg(test)]
mod tests {
    use std::vec;

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
    fn test_compute_centroid() {
        let mut samples: Vec<&Sample> = Vec::new();
        let sample: Sample = vec![1.0, 1.0];
        for _ in 0..10 {
            samples.push(&sample);
        }
        let c = compute_centroid(&samples);

        dbg!("{:?}", &c);
        assert!(c == sample)
    }
    #[test]
    fn test_compute_centroid_complex() {
        let mut samples: Vec<&Sample> = Vec::new();
        let s1: Sample = vec![1.0, 0.0];
        let s2: Sample = vec![0.0, 1.0];
        samples.push(&s1);
        samples.push(&s2);
        let c = compute_centroid(&samples);

        dbg!("{:?}", &c);
        assert!(c == vec![0.5, 0.5])
    }

    #[test]
    fn test_accuracy() {
        let truth: Vec<usize> = vec![1, 0];
        let pred: Vec<usize> = vec![1, 0];
        assert_eq!(evaluate(&pred, &truth), 1.0);

        let truth: Vec<usize> = vec![1, 0];
        let pred: Vec<usize> = vec![0, 1];
        assert_eq!(evaluate(&pred, &truth), 0.0);

        let truth: Vec<usize> = vec![1, 2];
        let pred: Vec<usize> = vec![1, 0];
        assert_eq!(evaluate(&pred, &truth), 0.5);
    }
}
