// struct to iterate over the dataset
use anyhow;
use std::error::Error;
use std::fmt;
use std::fs::File;
use std::io::{BufReader, Read};

#[derive(Debug, Clone)]
pub struct DatasetCreationError;

impl fmt::Display for DatasetCreationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Can't create dataset.")
    }
}

impl Error for DatasetCreationError {
    fn description(&self) -> &str {
        "Can't create dataset."
    }
}
pub struct DatasetIterator<'a, T>
where
    T: 'a,
    T: Sized,
{
    slice: &'a [T],
    nfeatures: usize,
}

impl<'a, T> DatasetIterator<'a, T>
where
    T: 'a,
{
    fn new(slice: &'a [T], nfeatures: usize) -> DatasetIterator<'a, T> {
        DatasetIterator { slice, nfeatures }
    }
}
impl<'a, T> Iterator for DatasetIterator<'a, T> {
    type Item = &'a [T];
    fn next(&mut self) -> Option<Self::Item> {
        if self.slice.is_empty() {
            None
        } else {
            let (first, last) = self.slice.split_at(self.nfeatures);
            self.slice = last;
            Some(first)
        }
    }
}

pub trait AsBytes {
    fn from_bytes(bytes: [u8; 4]) -> Self;
}

impl AsBytes for f32 {
    fn from_bytes(bytes: [u8; 4]) -> Self {
        f32::from_le_bytes(bytes)
    }
}

#[derive(Debug)]
pub struct Dataset<T> {
    pub nfeatures: usize,
    pub nsamples: usize,
    data: Vec<T>,
}

impl<T> Dataset<T>
where
    T: AsBytes,
{
    pub fn build(
        data: Vec<T>,
        nfeatures: usize,
        nsamples: usize,
    ) -> Result<Dataset<T>, DatasetCreationError> {
        if data.len() / nfeatures != nsamples {
            return Err(DatasetCreationError);
        } else {
            let nsamples = data.len() / nfeatures;
            Ok(Dataset {
                nfeatures,
                data,
                nsamples,
            })
        }
    }
    pub fn iter(&self) -> DatasetIterator<'_, T> {
        DatasetIterator::new(&self.data, self.nfeatures)
    }

    pub fn from_file(
        file_path: &str,
        nsamples: usize,
        nfeatures: usize,
    ) -> anyhow::Result<Dataset<T>> {
        let mut data: Vec<T> = Vec::new();
        let f = File::open(file_path)?;

        let mut reader = BufReader::new(f);
        let mut buffer = [0u8; std::mem::size_of::<f32>()];
        loop {
            if let Err(_error) = reader.read_exact(&mut buffer) {
                break;
            }
            data.push(T::from_bytes(buffer))
        }

        Ok(Self::build(data, nfeatures, nsamples)?)
    }
}
