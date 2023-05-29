use ruschio::{read_ndarray, Dataset, Kmeans};

const N_SAMPLES: usize = 1000;
const N_FEATURES: usize = 2;
const K: usize = 3;
const MAX_ITERS: usize = 100_000;

fn main() {
    let samples: Vec<f32> = read_ndarray("tests/blob_data", N_SAMPLES * N_FEATURES).unwrap();
    let mut samples_classes: Vec<f32> = read_ndarray("tests/blob_class", N_SAMPLES).unwrap();
    // Converting the ndarray
    let samples_classes: Vec<usize> = samples_classes.iter_mut().map(|e| *e as usize).collect();

    let dataset = Dataset::build(samples, N_FEATURES);
    let kmean = Kmeans::new(K, Some(MAX_ITERS));

    // Run the algorithm
    let (_centroids, _predicted_classes) = kmean.fit(&dataset, &samples_classes);
}
