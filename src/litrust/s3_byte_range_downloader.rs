use aws_sdk_s3::Client;
use pyo3::prelude::*;
use std::sync::Arc;
use tokio::fs::File;
use tokio::io::{AsyncSeekExt, AsyncWriteExt};
use tokio::sync::Mutex;
use tokio::task::JoinSet;

use aws_config::BehaviorVersion;

#[pyclass]
pub struct S3ByteRangeDownloader {
    s3client: Client,
}

#[pymethods]
impl S3ByteRangeDownloader {
    #[new]
    pub fn new() -> Self {
        let rt = tokio::runtime::Runtime::new().unwrap();
        // Load AWS config, setting a default region if not configured
        let config = rt.block_on(async {
            let mut config_loader = aws_config::defaults(BehaviorVersion::latest());

            if std::env::var("AWS_REGION").is_err() {
                config_loader = config_loader.region(aws_sdk_s3::config::Region::new("us-east-1"));
            }

            config_loader.load().await
        });

        let s3 = aws_sdk_s3::Client::new(&config);

        S3ByteRangeDownloader { s3client: s3 }
    }

    fn byte_range_download(
        &self,
        remote_path: &str,
        local_path: &str,
        num_of_bytes: u32,
        num_workers: u32,
    ) -> PyResult<()> {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(self.chunk_byte_range_download(
            remote_path,
            local_path,
            num_of_bytes,
            num_workers,
        ))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
}

impl S3ByteRangeDownloader {
    async fn chunk_byte_range_download(
        &self,
        remote_path: &str,
        local_path: &str,
        num_of_bytes: u32,
        num_workers: u32,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let (bucket_name, key) = get_bucket_name_and_key(remote_path);

        let file = File::create(local_path).await?;
        file.set_len(num_of_bytes as u64).await?;
        let file = Arc::new(Mutex::new(file));
        let mut tasks = JoinSet::new();

        let chunk_size = (num_of_bytes / num_workers) as u64;

        for i in 0..num_workers {
            let start = i as u64 * chunk_size;
            let end = if i == num_workers - 1 {
                (num_of_bytes - 1) as u64
            } else {
                (start + chunk_size) - 1
            };
            let bucket_name = bucket_name.to_string();
            let key = key.to_string().clone();
            let file = Arc::clone(&file);
            let s3client = self.s3client.clone();

            tasks.spawn(async move {
                let range_header = format!("bytes={}-{}", start, end);
                let response = s3client
                    .get_object()
                    .bucket(bucket_name)
                    .key(key)
                    .range(range_header)
                    .send()
                    .await?;

                let body: aws_sdk_s3::primitives::AggregatedBytes = response.body.collect().await?;
                let mut file = file.lock().await;
                file.seek(tokio::io::SeekFrom::Start(start)).await?;
                file.write_all(&body.into_bytes()).await?;
                Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
            });
        }

        // Wait for all tasks to finish
        while let Some(result) = tasks.join_next().await {
            if let Err(e) = result {
                return Err(e.to_string().into()); // Convert String error back to Box<dyn Error>
            }
        }

        Ok(())
    }
}

pub fn get_bucket_name_and_key(url: &str) -> (String, String) {
    if url.starts_with("s3://") == false {
        panic!("Invalid S3 URL: {}", url);
    }
    let url = url.trim_start_matches("s3://");
    let parts = url.splitn(2, '/').collect::<Vec<&str>>();
    (parts[0].to_string(), parts[1].to_string())
}
