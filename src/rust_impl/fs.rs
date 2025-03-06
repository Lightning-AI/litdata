pub mod gs;
pub mod local;
pub mod s3;
// pub mod error;

#[derive(Clone, Debug)]
pub enum StorageBackendType {
    Local(local::LocalStorage),
    LocalWithCache(local::LocalStorageWithCache),
    S3(s3::S3Storage),
    // GS(gs::GSStorage),
}

impl StorageBackendType {
    pub async fn get_bytes_in_range(
        &self,
        url: &str,
        range_start: u32,
        range_end: u32,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let result = match self {
            StorageBackendType::Local(local_storage) => {
                local_storage
                    .get_bytes_in_range(url, range_start, range_end)
                    .await
            }
            StorageBackendType::S3(s3_storage) => {
                s3_storage
                    .get_bytes_in_range(url, range_start, range_end)
                    .await
            }
            StorageBackendType::LocalWithCache(local_with_cache) => {
                local_with_cache
                    .get_bytes_in_range(url, range_start, range_end)
                    .await
            }
        };

        result
    }
}

pub trait StorageBackend {
    // async fn list(&self, path: &str) -> Result<Vec<String>, Box<dyn std::error::Error>>;

    // async fn upload(
    //     &self,
    //     local_path: &str,
    //     remote_path: &str,
    // ) -> Result<(), Box<dyn std::error::Error>>;

    // async fn download(
    //     &self,
    //     remote_path: &str,
    //     local_path: &str,
    // ) -> Result<(), Box<dyn std::error::Error>>;

    // async fn byte_range_download(
    //     &self,
    //     remote_path: &str,
    //     local_path: &str,
    //     num_threads: usize,
    // ) -> Result<(), Box<dyn std::error::Error>>;

    async fn get_bytes_in_range(
        &self,
        filename: &str,
        range_start: u32,
        range_end: u32,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>>;

    // async fn does_file_exist(&self, path: &str) -> bool;

    // async fn delete(&self, path: &str) -> Result<(), Box<dyn std::error::Error>>;
}
