use std::fs;
use std::io::{self};
use std::path::Path;

use super::StorageBackend;

#[derive(Clone)]
pub struct LocalStorage;

#[allow(dead_code)]
#[allow(unused_variables)]
impl LocalStorage {
    pub fn new() -> Self {
        LocalStorage
    }
}

impl StorageBackend for LocalStorage {
    async fn list(&self, path: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let dir = Path::new(path);
        if !dir.exists() {
            return Err(format!("Directory does not exist: {}", path).into());
        }
        if !dir.is_dir() {
            return Err(format!("Path is not a directory: {}", path).into());
        }

        let mut files = Vec::new();
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let file_name = entry.file_name().into_string().unwrap_or_default();
            files.push(file_name);
        }
        Ok(files)
    }

    async fn upload(
        &self,
        local_path: &str,
        remote_path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let source = Path::new(local_path);
        let destination = Path::new(remote_path);

        // Check if source exists and is a file
        if !source.exists() {
            return Err(format!("Source file does not exist: {}", local_path).into());
        }
        if !source.is_file() {
            return Err(format!("Source is not a file: {}", local_path).into());
        }

        let final_dest = if destination.is_dir() {
            // Append the filename to the destination directory
            destination.join(source.file_name().ok_or("Invalid file name")?)
        } else {
            destination.to_path_buf()
        };

        fs::copy(source, final_dest)?;
        Ok(())
    }

    async fn download(
        &self,
        remote_path: &str,
        local_path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let source = Path::new(remote_path);
        let destination = Path::new(local_path);

        // Check if source exists and is a file
        if !source.exists() {
            return Err(format!("Source file does not exist: {}", remote_path).into());
        }
        if !source.is_file() {
            return Err(format!("Source is not a file: {}", remote_path).into());
        }

        let final_dest = if destination.is_dir() {
            // Append the filename to the destination directory
            destination.join(source.file_name().ok_or("Invalid file name")?)
        } else {
            destination.to_path_buf()
        };

        fs::copy(source, final_dest)?;
        Ok(())
    }

    async fn byte_range_download(
        &self,
        _remote_path: &str,
        _local_path: &str,
        _num_threads: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Err(Box::new(io::Error::new(
            io::ErrorKind::Unsupported,
            "Byte-range download is not supported for local storage",
        )))
    }

    async fn get_bytes_in_range(
        &self,
        _url: &str,
        _range_start: u32,
        _range_end: u32,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        Err(Box::new(io::Error::new(
            io::ErrorKind::Unsupported,
            "Not implemented",
        )))
    }

    async fn does_file_exist(&self, path: &str) -> bool {
        let p = Path::new(path);
        p.exists() && p.is_file()
    }

    async fn delete(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let file = Path::new(path);
        if !file.exists() {
            return Err(format!("File does not exist: {}", path).into());
        }
        fs::remove_file(file)?;
        Ok(())
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
    pub fn new() -> Self {
        LocalStorageWithCache {
            loc_stor: LocalStorage::new(),
        }
    }
}

impl StorageBackend for LocalStorageWithCache {
    async fn list(&self, path: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        self.loc_stor.list(path).await
    }

    async fn upload(
        &self,
        local_path: &str,
        remote_path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.loc_stor.upload(local_path, remote_path).await
    }

    async fn download(
        &self,
        remote_path: &str,
        local_path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let remote_path = remote_path.trim_start_matches("local:");

        self.loc_stor.download(remote_path, local_path).await
    }

    async fn byte_range_download(
        &self,
        _remote_path: &str,
        _local_path: &str,
        _num_threads: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Err(Box::new(io::Error::new(
            io::ErrorKind::Unsupported,
            "Byte-range download is not supported for local storage",
        )))
    }

    async fn get_bytes_in_range(
        &self,
        _url: &str,
        _range_start: u32,
        _range_end: u32,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        Err(Box::new(io::Error::new(
            io::ErrorKind::Unsupported,
            "Not implemented",
        )))
    }

    async fn does_file_exist(&self, path: &str) -> bool {
        self.loc_stor.does_file_exist(path).await
    }

    async fn delete(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.loc_stor.delete(path);
        Ok(())
    }
}
