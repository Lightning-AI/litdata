use std::collections::HashMap;
use tokio::task::JoinSet;

use crate::rust_impl::fs::{StorageBackend, StorageBackendType};

use super::utils::get_storage_backend;

///
#[allow(dead_code)]
pub struct StreamingDataProvider {
    epoch: usize,

    remote_dir: String,
    chunks: Vec<HashMap<String, String>>,

    index_to_bytes: HashMap<usize, Vec<u8>>,

    chunk_index_odd_epoch: Vec<usize>, // contains the index of the chunks for the odd epochs (1, 3, 5, ...)
    chunk_index_even_epoch: Vec<usize>, // contains the index of the chunks for the even epochs (0, 2, 4, ...)
    sample_index_odd_epoch: Vec<Vec<usize>>, // contains the index of the samples for the odd epochs (1, 3, 5, ...)
    sample_index_even_epoch: Vec<Vec<usize>>, // contains the index of the samples for the even epochs (0, 2, 4, ...)

    chunk_index_offset: HashMap<usize, Vec<u32>>, // since offsets were stored as np.uint32 in the chunks file
    chunk_index_begin: HashMap<usize, u32>, // for the current order of streaming data, at which index the chunk starts

    pointer_x: usize, // current pointer (x) in sample_index
    pointer_y: usize, // current pointer (y) in sample_index

    pre_item_download_count: usize, // how many items to download in advance

    storage_provider: StorageBackendType,
}

#[allow(dead_code)]
#[allow(unused_variables)]
impl StreamingDataProvider {
    pub fn new(remote_dir: String) -> Self {
        return StreamingDataProvider {
            epoch: 0,
            remote_dir: String::from(&remote_dir),
            chunks: Vec::new(),
            index_to_bytes: HashMap::new(),
            chunk_index_odd_epoch: Vec::new(),
            chunk_index_even_epoch: Vec::new(),
            sample_index_odd_epoch: Vec::new(),
            sample_index_even_epoch: Vec::new(),
            chunk_index_offset: HashMap::new(),
            chunk_index_begin: HashMap::new(),
            pointer_x: 0,
            pointer_y: 0,
            pre_item_download_count: 60,
            storage_provider: get_storage_backend(&remote_dir),
        };
    }

    pub fn on_start(&self) {
        // fetch `pre_item_download_count`
        let rt = tokio::runtime::Runtime::new().unwrap();
        // let mut tasks = JoinSet::new();

        // first we need to download offset array for all chunk indexes in parallel
        // and store them in `chunk_index_offset` map
        for chunk_index in self.chunk_index_odd_epoch.iter() {}

        // for i in 0..self.pre_item_download_count {
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
    }

    pub fn set_epoch(&mut self, epoch: usize) {
        self.epoch = epoch;
    }

    pub fn set_chunk_and_sample_index(
        &mut self,
        epoch: usize,
        chunk_index: usize,
        sample_index: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // set chunk_index and sample_index in {odd/even} depending on epoch.
        Ok(())
    }

    pub fn get_bytes_for_chunk_index_and_index(
        &self,
        chunk_index: usize,
        sample_index: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        panic!("Not implemented");
    }

    pub async fn get_chunk_index_offset(
        &mut self,
        odd_epoch: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // go through all chunk indexes and if they don't already exist in the chunk_index_offset hashamp, download them.
        // if they already exist, skip them.
        let chunk_index_vec = if odd_epoch {
            &self.chunk_index_odd_epoch
        } else {
            &self.chunk_index_even_epoch
        };

        let mut tasks = JoinSet::new();

        for chunk_index in chunk_index_vec.iter() {
            if self.chunk_index_offset.contains_key(chunk_index) {
                continue;
            }

            let storage_provider = self.storage_provider.clone();
            let remote_dir = self.remote_dir.clone();
            let chunks = self.chunks.clone();
            let chunk_index = chunk_index.clone();

            tasks.spawn(async move {
                let range_start = 4; // first 4 bytes of chunk store number of samples in the chunk
                let range_end = chunks[chunk_index]
                    .get("chunk_size")
                    .unwrap()
                    .parse::<u32>()
                    .unwrap();

                let offset_array_bytes = match storage_provider {
                    StorageBackendType::Local(ref local_storage) => {
                        local_storage
                            .get_bytes_in_range(&remote_dir, range_start, range_end)
                            .await
                    }
                    StorageBackendType::LocalWithCache(ref local_with_cache) => {
                        local_with_cache
                            .get_bytes_in_range(&remote_dir, range_start, range_end)
                            .await
                    }
                    _ => {
                        panic!("Not implemented");
                    }
                };
                if let Err(e) = offset_array_bytes {
                    panic!("failed to download offset array: {}", e);
                }
                let offset_array_bytes = offset_array_bytes.unwrap();

                // Convert bytes to u32
                let u32_vec: Vec<u32> = offset_array_bytes
                    .chunks(4) // Group bytes into chunks of 4
                    .map(|chunk| {
                        let mut buf = [0u8; 4];
                        buf.copy_from_slice(chunk);
                        u32::from_le_bytes(buf) // Convert 4 bytes into a u32 (little-endian)
                    })
                    .collect();

                return (chunk_index, u32_vec);
            });
        }

        // Wait for all tasks to finish
        while let Some(result) = tasks.join_next().await {
            if let Err(e) = result {
                panic!("failed to download offset array: {}", e);
            }
            let (chunk_index, offset_array) = result.unwrap();
            self.chunk_index_offset.insert(chunk_index, offset_array);
        }

        Ok(())
    }

    pub fn get_chunk_index_offset_array(&self, chunk_index: u32) {}
}
