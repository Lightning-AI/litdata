use super::fs::StorageBackendType;

pub fn get_storage_backend(url: &str) -> StorageBackendType {
    _ = url;
    panic!("Not implemented");
}

pub fn get_bucket_name_and_key(url: &str) -> (String, String) {
    let url = url.trim_start_matches("s3://");
    let parts = url.splitn(2, '/').collect::<Vec<&str>>();
    (parts[0].to_string(), parts[1].to_string())
}
