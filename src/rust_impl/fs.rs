pub mod gs;
pub mod local;
pub mod s3;
// pub mod error;

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
        path: &str,
        range: (u64, u64),
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>>;

    fn does_file_exist(&self, path: &str) -> bool;

    fn delete(&self, path: &str) -> Result<(), Box<dyn std::error::Error>>;
}
