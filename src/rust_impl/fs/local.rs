use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use super::StorageBackend;

#[derive(Clone)]
pub struct LocalStorage {
    remote_dir: String,
}

#[allow(dead_code)]
#[allow(unused_variables)]
impl LocalStorage {
    pub fn new(remote_dir: String) -> Self {
        let mut remote_dir = remote_dir;
        if remote_dir.ends_with("/") == false {
            remote_dir = format!("{}/", remote_dir);
        }
        LocalStorage { remote_dir }
    }
}

impl StorageBackend for LocalStorage {
    async fn get_bytes_in_range(
        &self,
        filename: &str,
        _range_start: u32,
        _range_end: u32,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let _url = format!("{}{}", self.remote_dir, filename); // remote_dir ends with "/"

        let mut file = File::open(_url)?; // Open file in binary mode
        let mut buffer: Vec<u8> = vec![0; (_range_end - _range_start) as usize]; // Allocate buffer

        file.seek(SeekFrom::Start(_range_start as u64))?; // Move to start position
        file.read_exact(&mut buffer)?; // Read exact bytes into buffer

        Ok(buffer)
    }
}

// ----- LocalStorageWithCache -----

#[derive(Clone)]
pub struct LocalStorageWithCache {
    loc_stor: LocalStorage,
}

#[allow(dead_code)]
#[allow(unused_variables)]
impl LocalStorageWithCache {
    pub fn new(remote_dir: String) -> Self {
        let remote_dir_without_local = remote_dir.replace("local:", "");
        LocalStorageWithCache {
            loc_stor: LocalStorage::new(remote_dir_without_local),
        }
    }
}

impl StorageBackend for LocalStorageWithCache {
    async fn get_bytes_in_range(
        &self,
        filename: &str,
        _range_start: u32,
        _range_end: u32,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        self.loc_stor
            .get_bytes_in_range(&filename, _range_start, _range_end)
            .await
    }
}
