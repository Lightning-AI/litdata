use crate::rust_impl::utils::get_bucket_name_and_key;

use super::StorageBackend;
use aws_config::BehaviorVersion;
use aws_sdk_s3::Client;

#[derive(Clone, Debug)]
pub struct S3Storage {
    s3client: Client,
    remote_dir: String,
}

impl S3Storage {
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
}

impl StorageBackend for S3Storage {
    async fn get_bytes_in_range(
        &self,
        filename: &str,
        _range_start: u32,
        _range_end: u32,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let _url = &format!("{}{}", self.remote_dir, filename); // remote_dir ends with "/"
        // println!("going to download {_url}");
        let (bucket_name, key) = get_bucket_name_and_key(_url);
        let range_header = format!("bytes={}-{}", _range_start, _range_end-1);

        let response = self
            .s3client
            .get_object()
            .bucket(bucket_name)
            .key(key)
            .range(range_header)
            .send()
            .await;
<<<<<<< HEAD
=======

>>>>>>> c02e8ac (update)
        if let Err(e) = response {
            panic!("failed to download data: {:?}", e);
        }
        let response = response.unwrap();
<<<<<<< HEAD
=======

>>>>>>> c02e8ac (update)
        let body = response.body.collect().await?;
        let bytes = body.into_bytes().to_vec();
        Ok(bytes)
    }
}
