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
    if samples.is_empty() {
        return centroid;
    }

    for i in 0..samples[0].len() {
        let mut temp: f32 = 0.0;
        for sample in samples {
            temp += sample[i];
            centroid.push(temp / samples.len() as f32);
        }
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
        let mut centroids: Vec<Sample> = dataset.samples[0..self.n_clusters]
            .iter()
            .map(|s| s.clone())
            .collect();

        // FOR MAX_ITER
        let mut pred_classes: Vec<usize> = vec![0; dataset.samples.len()];
        for _iter in 0..self.max_iterations {
            // loop over the dataset
            // compute distances to centroids
            // Find minimum distance
            // Assign datapoint to centroids
            for (idx, sample) in dataset.samples.iter().enumerate() {
                let class = centroids
                    .iter()
                    .map(|c| l2_distance(c, sample))
                    .enumerate()
                    .min_by(|&(_, i), &(_, j)| i.total_cmp(&j))
                    .unwrap()
                    .0;
                pred_classes[idx] = class;
            }
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
