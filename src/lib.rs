#![feature(portable_simd)]
use rand::Rng;
use rayon::prelude::*;
use std::io::{BufWriter, Error};
use std::{fs::File, io::Write};

pub mod dataset;
mod linalg;

// #[derive(Debug)]
// pub struct Kmeans {
//     // number of clusters
//     n_clusters: usize,
//     max_iterations: usize,
// }

// impl Kmeans {
//     pub fn new(n_clusters: usize, max_iterations: usize) -> Self {
//         Kmeans {
//             n_clusters,
//             max_iterations: max_iterations,
//         }
//     }

//     pub fn fit(&self, dataset: &Dataset<f32>) -> (Vec<Sample>, Vec<usize>) {
//         // Pick random K vectors from the dataset
//         let mut rng = rand::thread_rng();
//         let mut centroids: Vec<Sample> = Vec::with_capacity(self.n_clusters);
//         for _k in 0..self.n_clusters {
//             let n = rng.gen_range(0..dataset.samples.len());
//             centroids.push(dataset.samples[n].clone());
//         }

//         // FOR MAX_ITER
//         let mut pred_classes: Vec<usize> = vec![0; dataset.samples.len()];
//         for iter in 0..self.max_iterations {
//             // loop over the dataset
//             // compute distances to centroids
//             // Find minimum distance
//             // Assign datapoint to centroids
//             let assigment = dataset
//                 .samples
//                 .par_iter()
//                 .map(|sample| {
//                     let (class, _distance) = centroids
//                         .iter()
//                         .map(|c| linalg::l2_distance(c, sample))
//                         .enumerate()
//                         .min_by(|&(_, i), &(_, j)| i.total_cmp(&j))
//                         .unwrap();
//                     class
//                 })
//                 .collect();
//             // Stopping criterion : if pred_classes didn't change
//             if assigment != pred_classes {
//                 pred_classes = assigment;
//                 // recompute cluster centroids
//                 for k in 0..self.n_clusters {
//                     let filtered_samples: Vec<&Sample> = pred_classes
//                         .iter()
//                         .zip(dataset.samples.iter())
//                         .filter(|&(class, _)| *class == k)
//                         .map(|(_, item)| item)
//                         .collect();
//                     centroids[k] = linalg::compute_centroid(&filtered_samples);
//                 }
//             } else {
//                 println!("Converged after {} iterations.", iter);
//                 break;
//             }
//         }
//         (centroids, pred_classes)
//     }
// }

pub fn dump_result(classes: &Vec<usize>, file_path: &str) -> Result<(), Error> {
    let f = File::create(file_path)?;
    let mut writer = BufWriter::new(f);
    let written_bytes: usize = classes
        .iter()
        .map(|e| writer.write(&e.to_le_bytes()).unwrap())
        .sum();
    writer.flush()?;
    assert_eq!(classes.len() * 8, written_bytes);
    Ok(())
}

pub fn evaluate(predicted: &Vec<usize>, truth: &Vec<usize>) -> f32 {
    // TODO: incorrect evaluation
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
    use super::*;
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
