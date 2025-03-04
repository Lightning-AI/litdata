// use super::StorageBackend;
// use aws_config::BehaviorVersion;
// use aws_sdk_s3::primitives::ByteStream;
// use aws_sdk_s3::Client;
// use pyo3::prelude::*;
// use std::sync::Arc;
// use tokio::fs::File;
// use tokio::fs::OpenOptions;
// use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};
// use tokio::sync::Mutex;
// use tokio::task::JoinSet;

// #[pyclass]
// pub struct S3Storage {
//     s3client: Client,
// }

// #[pymethods]
// impl S3Storage {
//     #[new]
//     pub fn new() -> Self {
//         let rt = tokio::runtime::Runtime::new().unwrap();
//         // Load AWS config, setting a default region if not configured
//         let config = rt.block_on(async {
//             let mut config_loader = aws_config::defaults(BehaviorVersion::latest());

//             if std::env::var("AWS_REGION").is_err() {
//                 config_loader = config_loader.region(aws_sdk_s3::config::Region::new("us-east-1"));
//             }

//             config_loader.load().await
//         });

//         let s3 = aws_sdk_s3::Client::new(&config);
//         S3Storage { s3client: s3 }
//     }

//     pub fn list(&self, path: &str) -> PyResult<Vec<String>> {
//         let rt = tokio::runtime::Runtime::new().unwrap();
//         rt.block_on(StorageBackend::list(self, path))
//             .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
//     }

//     pub fn does_file_exist(&self, path: &str) -> PyResult<bool> {
//         let rt = tokio::runtime::Runtime::new().unwrap();
//         Ok(rt.block_on(StorageBackend::does_file_exist(self, path)))
//     }

//     pub fn upload(&self, local_path: &str, remote_path: &str) -> PyResult<()> {
//         let rt = tokio::runtime::Runtime::new().unwrap();
//         rt.block_on(StorageBackend::upload(self, local_path, remote_path))
//             .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
//     }

//     pub fn delete(&self, path: &str) -> PyResult<()> {
//         let rt = tokio::runtime::Runtime::new().unwrap();
//         rt.block_on(StorageBackend::delete(self, path))
//             .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
//     }

//     pub fn download(&self, path: &str, local_path: &str) -> PyResult<()> {
//         let rt = tokio::runtime::Runtime::new().unwrap();
//         rt.block_on(StorageBackend::download(self, path, local_path))
//             .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
//     }

//     pub fn byte_range_download(
//         &self,
//         remote_path: &str,
//         local_path: &str,
//         num_threads: usize,
//     ) -> PyResult<()> {
//         let rt = tokio::runtime::Runtime::new().unwrap();
//         rt.block_on(StorageBackend::byte_range_download(
//             self,
//             remote_path,
//             local_path,
//             num_threads,
//         ))
//         .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
//     }
// }

// impl StorageBackend for S3Storage {
//     async fn list(&self, path: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
//         // Ensure path starts with "s3://"
//         if !path.starts_with("s3://") {
//             return Err("Invalid S3 path. Must start with 's3://'".into());
//         }

//         // Extract bucket name and prefix
//         let parts: Vec<&str> = path.trim_start_matches("s3://").splitn(2, '/').collect();
//         if parts.len() < 1 {
//             return Err("Invalid S3 path. Must contain a bucket name.".into());
//         }
//         let bucket_name = parts.get(0).unwrap_or(&"No bucket name found.");
//         let prefix = parts.get(1).unwrap_or(&""); // Prefix is optional

//         // Make the request
//         let response = self
//             .s3client
//             .list_objects_v2()
//             .bucket(bucket_name.to_string())
//             .prefix(prefix.to_string()) // Use extracted prefix
//             .send()
//             .await?;

//         // Extract object keys safely
//         let keys = response
//             .contents()
//             .iter()
//             .map(|c| c.key.clone().unwrap_or("".to_string()))
//             .collect();

//         Ok(keys)
//     }

//     async fn does_file_exist(&self, path: &str) -> bool {
//         let parts: Vec<&str> = path.trim_start_matches("s3://").splitn(2, '/').collect();
//         if parts.len() < 2 {
//             return false; // Invalid path format
//         }

//         let bucket_name = parts[0];
//         let object_key = parts[1]; // Ensure only object key is passed

//         let response = self
//             .s3client
//             .head_object()
//             .bucket(bucket_name)
//             .key(object_key)
//             .send()
//             .await;

//         response.is_ok()
//     }

//     async fn upload(
//         &self,
//         local_path: &str,
//         remote_path: &str,
//     ) -> Result<(), Box<dyn std::error::Error>> {
//         // Read the file into a byte buffer
//         let mut file = File::open(local_path).await?;
//         let mut buffer = Vec::new();
//         file.read_to_end(&mut buffer).await?;

//         // Extract bucket and key from the remote_path
//         let parts: Vec<&str> = remote_path
//             .trim_start_matches("s3://")
//             .splitn(2, '/')
//             .collect();
//         if parts.len() < 2 {
//             return Err("Invalid S3 path format. Expected 's3://bucket-name/key'".into());
//         }
//         let bucket_name = parts[0];
//         let key = parts[1];

//         // Upload to S3
//         self.s3client
//             .put_object()
//             .bucket(bucket_name)
//             .key(key)
//             .body(ByteStream::from(buffer))
//             .send()
//             .await?;

//         Ok(())
//     }

//     async fn delete(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
//         // Extract bucket and key from the S3 path
//         let parts: Vec<&str> = path.trim_start_matches("s3://").splitn(2, '/').collect();
//         if parts.len() < 2 {
//             return Err("Invalid S3 path format. Expected 's3://bucket-name/key'".into());
//         }
//         let bucket_name = parts[0];
//         let key = parts[1];

//         // Send delete request
//         self.s3client
//             .delete_object()
//             .bucket(bucket_name)
//             .key(key)
//             .send()
//             .await?;

//         Ok(())
//     }

//     async fn download(
//         &self,
//         path: &str,
//         local_path: &str,
//     ) -> Result<(), Box<dyn std::error::Error>> {
//         // Extract bucket and key from the S3 path
//         let parts: Vec<&str> = path.trim_start_matches("s3://").splitn(2, '/').collect();
//         if parts.len() < 2 {
//             return Err("Invalid S3 path format. Expected 's3://bucket-name/key'".into());
//         }
//         let bucket_name = parts[0];
//         let key = parts[1];

//         // println!("in download fn with: bucket_name: {bucket_name} & key: {key}");
//         // Fetch the object from S3
//         let response = self
//             .s3client
//             .get_object()
//             .bucket(bucket_name)
//             .key(key)
//             .send()
//             .await?;

//         // Read the response body
//         let mut stream = response.body.into_async_read();
//         let mut file = File::create(local_path).await?;
//         tokio::io::copy(&mut stream, &mut file).await?;

//         Ok(())
//     }

//     async fn byte_range_download(
//         &self,
//         remote_path: &str,
//         local_path: &str,
//         num_threads: usize,
//     ) -> Result<(), Box<dyn std::error::Error>> {
//         // Extract bucket & key
//         let parts: Vec<&str> = remote_path
//             .trim_start_matches("s3://")
//             .splitn(2, '/')
//             .collect();
//         if parts.len() < 2 {
//             return Err("Invalid S3 path format. Expected 's3://bucket-name/key'".into());
//         }
//         let bucket_name = parts[0];
//         let key = parts[1];
//         // println!("in byte-range download fn with: bucket_name: {bucket_name} & key: {key}");
//         // Step 1: Get file size
//         let head_response = self
//             .s3client
//             .head_object()
//             .bucket(bucket_name)
//             .key(key)
//             .send()
//             .await?;
//         let file_size = head_response.content_length().unwrap_or(0) as u64;

//         // Step 2: Create an empty file with correct size
//         let file = OpenOptions::new()
//             .create(true)
//             .write(true)
//             .truncate(true)
//             .open(local_path)
//             .await?;
//         file.set_len(file_size).await?; // Preallocate space

//         // Step 3: Spawn async tasks for each byte range
//         let chunk_size = file_size / num_threads as u64;
//         let file = Arc::new(Mutex::new(file)); // Shared mutable file
//         let mut tasks = JoinSet::new();

//         for i in 0..num_threads {
//             let start = i as u64 * chunk_size;
//             let end = if i == num_threads - 1 {
//                 file_size - 1
//             } else {
//                 (start + chunk_size) - 1
//             };
//             let s3client = self.s3client.clone();
//             let bucket_name = bucket_name.to_string();
//             let key = key.to_string();
//             let file = Arc::clone(&file);

//             tasks.spawn(async move {
//                 let range_header = format!("bytes={}-{}", start, end);
//                 let response = s3client
//                     .get_object()
//                     .bucket(bucket_name)
//                     .key(key)
//                     .range(range_header)
//                     .send()
//                     .await?;

//                 let body = response.body.collect().await?;
//                 let mut file = file.lock().await;
//                 file.seek(std::io::SeekFrom::Start(start)).await?;
//                 file.write_all(&body.into_bytes()).await?;
//                 Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
//             });
//         }

//         // Wait for all tasks to finish
//         while let Some(result) = tasks.join_next().await {
//             if let Err(e) = result {
//                 return Err(e.to_string().into()); // Convert String error back to Box<dyn Error>
//             }
//         }

//         Ok(())
//     }
// }
