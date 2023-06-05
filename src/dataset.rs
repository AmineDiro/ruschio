// struct to iterate over the dataset
use std::fmt;

#[derive(Debug, Clone)]
pub struct DatasetCreationError;

impl fmt::Display for DatasetCreationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Can't create dataset.")
    }
}

pub struct DatasetIterator<'a, T>
where
    T: 'a,
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

#[derive(Debug)]
pub struct Dataset<T> {
    pub nfeatures: usize,
    pub nsamples: usize,
    data: Vec<T>,
}

impl<T> Dataset<T> {
    pub fn build(data: Vec<T>, nfeatures: usize) -> Result<Dataset<T>, DatasetCreationError> {
        if data.len() % nfeatures != 0 {
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
}
