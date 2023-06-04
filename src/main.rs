use std::io::{BufWriter, Error};
use std::{fs::File, io::Write};

use ruschio::{evaluate, read_ndarray, Dataset, Kmeans};

const N_SAMPLES: usize = 1000;
const N_FEATURES: usize = 2;
const K: usize = 3;
const MAX_ITERS: usize = 100_000;

fn dump_result(classes: &Vec<usize>, file_path: &str) -> Result<(), Error> {
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

fn main() {
    let samples: Vec<f32> = read_ndarray("tests/blob_data", N_SAMPLES * N_FEATURES).unwrap(); // N_SAMPLES x N_features (2)
    let mut samples_classes: Vec<f32> = read_ndarray("tests/blob_class", N_SAMPLES).unwrap();
    let samples_classes: Vec<usize> = samples_classes.iter_mut().map(|e| *e as usize).collect();

    let dataset = Dataset::build(samples, N_FEATURES);
    let kmean = Kmeans::new(K, Some(MAX_ITERS));

    // Run the algorithm
    let (_centroids, predicted_classes) = kmean.fit(&dataset);

    let acc = evaluate(&predicted_classes, &samples_classes);
    println!("10 Predictions: {:?}", &predicted_classes[0..10]);
    println!("Accuracy : {}", acc);

    dump_result(&predicted_classes, "tests/predicted_classes.data")
        .expect("Error in dumping result");
}
