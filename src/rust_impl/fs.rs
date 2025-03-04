use std::collections::HashMap;

pub mod gs;
pub mod local;
pub mod s3;
// pub mod error;

pub trait StorageBackend {
    async fn list(&self, path: &str) -> Result<Vec<String>, Box<dyn std::error::Error>>;

    async fn upload(&self, local_path: &str, remote_path: &str)
    -> Result<(), Box<dyn std::error::Error>>;

    async fn download(
        &self,
        remote_path: &str,
        local_path: &str,
    ) -> Result<(), Box<dyn std::error::Error>>;

    async fn byte_range_download(
        &self,
        remote_path: &str,
        local_path: &str,
        num_threads: usize,
    ) -> Result<(), Box<dyn std::error::Error>>;

    async fn does_file_exist(&self, path: &str) -> bool;

    async fn delete(&self, path: &str) -> Result<(), Box<dyn std::error::Error>>;
}

///
pub struct StorageDownloader {
    index_to_bytes: HashMap<usize, Vec<u8>>,
    chunk_index: Vec<usize>,
    sample_index: Vec<Vec<usize>>,
    chunk_index_offset: HashMap<usize, Vec<u32>>, // since offsets were stored as np.uint32 in the chunks file
    chunk_index_begin: HashMap<usize, u32>, // for the current order of streaming data, at which index the chunk starts
}

impl StorageDownloader {
    pub fn new() -> Self {
        // Self { index_to_bytes: HashMap::new() }
        return StorageDownloader{
            index_to_bytes: HashMap::new(),
            chunk_index: Vec::new(),
            sample_index: Vec::new(),
            chunk_index_offset: HashMap::new(),
            chunk_index_begin: HashMap::new(),
        };
    }
}
