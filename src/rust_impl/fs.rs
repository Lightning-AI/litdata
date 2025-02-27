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
