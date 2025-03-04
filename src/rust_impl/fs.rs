pub mod gs;
pub mod local;
pub mod s3;
// pub mod error;

#[derive(Clone)]
pub enum StorageBackendType {
    Local(local::LocalStorage),
    LocalWithCache(local::LocalStorageWithCache),
    // S3(s3::S3Storage),
    // GS(gs::GSStorage),
}


pub trait StorageBackend {
    fn list(&self, path: &str) -> Result<Vec<String>, Box<dyn std::error::Error>>;

    fn upload(&self, local_path: &str, remote_path: &str)
        -> Result<(), Box<dyn std::error::Error>>;

    fn download(
        &self,
        remote_path: &str,
        local_path: &str,
    ) -> Result<(), Box<dyn std::error::Error>>;

    fn byte_range_download(
        &self,
        remote_path: &str,
        local_path: &str,
        num_threads: usize,
    ) -> Result<(), Box<dyn std::error::Error>>;

    async fn get_bytes_in_range(
        &self,
        url: &str,
        range_start: u32,
        range_end: u32,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>>;

    fn does_file_exist(&self, path: &str) -> bool;

    fn delete(&self, path: &str) -> Result<(), Box<dyn std::error::Error>>;
}
