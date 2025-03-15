use crate::rust_impl::utils::get_bucket_name_and_key;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use super::StorageBackend;
use aws_config::BehaviorVersion;
use aws_sdk_s3::Client;
use std::time::Duration;

#[derive(Clone, Debug)]
#[pyclass]
pub struct S3Storage {
    s3client: Client,
    remote_dir: String,
}

#[pymethods]
impl S3Storage {
    #[new]
    pub fn new(remote_dir: String) -> Self {
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

        let mut remote_dir = remote_dir;
        if remote_dir.ends_with("/") == false {
            remote_dir = format!("{}/", remote_dir);
        }

        S3Storage {
            s3client: s3,
            remote_dir: remote_dir,
        }
    }

    fn byte_range_download(
        &self,
        remote_path: &str,
        local_path: &str,
        num_workers: usize,
    ) -> PyResult<()> {
        _ = (remote_path, local_path, num_workers);
        Err(PyErr::new::<PyRuntimeError, &str>(
            "S3 byte_range_download -> Not implemented",
        ))
    }
}

impl StorageBackend for S3Storage {
    async fn get_bytes_in_range(
        &self,
        filename: &str,
        range_start: u32,
        range_end: u32,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let (bucket_name, key) =
            get_bucket_name_and_key(&format!("{}{}", self.remote_dir, filename));
        let range_header = format!("bytes={}-{}", range_start, range_end - 1);

        const MAX_RETRIES: u32 = 3;
        const INITIAL_BACKOFF_MS: u64 = 100;

        let mut attempt = 0;
        loop {
            match self
                .s3client
                .get_object()
                .bucket(&bucket_name)
                .key(&key)
                .range(&range_header)
                .send()
                .await
            {
                Ok(response) => {
                    let body = response.body.collect().await?;
                    return Ok(body.into_bytes().to_vec());
                }
                Err(e) => {
                    attempt += 1;
                    if attempt >= MAX_RETRIES {
                        return Err(e.into());
                    }
                    // Exponential backoff
                    let delay = INITIAL_BACKOFF_MS * (2_u64.pow(attempt - 1));
                    tokio::time::sleep(Duration::from_millis(delay)).await;
                    continue;
                }
            }
        }
    }
}
