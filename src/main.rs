use clap::Parser;
use ruschio::*;

/// Naive Kmean algorithm
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Config {
    #[arg(short, long)]
    nsamples: usize,

    #[arg(long, default_value_t = 2)]
    nfeatures: usize,

    #[arg(long, default_value_t = 3)]
    nclusters: usize,

    #[arg(short, long, default_value_t = 1000)]
    max_iterations: usize,
}

fn main() {
    let config = Config::parse();

    println!("{:?}", config);

    let samples: Vec<f32> =
        read_ndarray("tests/blob_data", config.nsamples * config.nfeatures).unwrap(); // N_SAMPLES x N_features (2)
    let mut samples_classes: Vec<f32> = read_ndarray("tests/blob_class", config.nsamples).unwrap();
    let samples_classes: Vec<usize> = samples_classes.iter_mut().map(|e| *e as usize).collect();

    let dataset = Dataset::build(samples, config.nfeatures);
    let kmean = Kmeans::new(config.nclusters, config.max_iterations);

    // Run the algorithm
    let (_centroids, predicted_classes) = kmean.fit(&dataset);

    let acc = evaluate(&predicted_classes, &samples_classes);
    println!("10 Predictions: {:?}", &predicted_classes[0..10]);
    println!("Accuracy : {}", acc);

    dump_result(&predicted_classes, "tests/predicted_classes.data")
        .expect("Error in dumping result");
}
