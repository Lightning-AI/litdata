use super::fs::StorageBackendType;
use crate::rust_impl::fs::local::{LocalStorage, LocalStorageWithCache};
use crate::rust_impl::fs::s3::S3Storage;

pub fn get_storage_backend(url: &str) -> StorageBackendType {
    if url.starts_with("s3://") {
        return StorageBackendType::S3(S3Storage::new(url.to_string()));
    }
    if url.starts_with("local://") {
        return StorageBackendType::LocalWithCache(LocalStorageWithCache::new(url.to_string()));
    }
    return StorageBackendType::Local(LocalStorage::new(url.to_string()));
}

pub fn get_bucket_name_and_key(url: &str) -> (String, String) {
    if url.starts_with("s3://") == false {
        panic!("Invalid S3 URL: {}", url);
    }
    let url = url.trim_start_matches("s3://");
    let parts = url.splitn(2, '/').collect::<Vec<&str>>();
    (parts[0].to_string(), parts[1].to_string())
}
