use pyo3::prelude::*;
use std::collections::HashMap;
use tokio::task::JoinSet;

use crate::rust_impl::fs::StorageBackendType;

use super::utils::get_storage_backend;

#[pyclass]
pub struct StreamingDataProvider {
    // downloading_epoch is the epoch of the items being downloaded
    downloading_epoch: u32,
    // streaming_epoch is the epoch of the items being streamed
    streaming_epoch: u32,

    remote_dir: String,
    chunks: Vec<HashMap<String, String>>,

    chunk_index_odd_epoch: Vec<u32>, // contains the index of the chunks for the odd epochs (1, 3, 5, ...)
    chunk_index_even_epoch: Vec<u32>, // contains the index of the chunks for the even epochs (0, 2, 4, ...)
    sample_index_odd_epoch: Vec<Vec<u32>>, // contains the index of the samples for the odd epochs (1, 3, 5, ...)
    sample_index_even_epoch: Vec<Vec<u32>>, // contains the index of the samples for the even epochs (0, 2, 4, ...)

    chunk_index_offset: HashMap<u32, Vec<u32>>, // offset array for each chunk_index

    pointer_x: usize, // current pointer (x) in sample_index
    pointer_y: usize, // current pointer (y) in sample_index

    on_start_pre_item_download_count: u32, // how many items to download in advance on start
    get_next_k_item_count: u32, // how many items to download in advance on get_next_k_item

    storage_provider: StorageBackendType,
}

#[allow(dead_code)]
#[allow(unused_variables)]
impl StreamingDataProvider {
    pub async fn pre_fetch_items_on_start(
        &mut self,
        limit: u32,
    ) -> Result<Vec<(u32, u32, u32, Vec<u8>)>, Box<dyn std::error::Error>> {
        let mut tasks = JoinSet::new();

        for idx in 0..limit {
            if self.downloading_epoch - self.streaming_epoch > 1 {
                // downloading_epoch should not predownload items for more than 1 epoch
                break;
            }
            // don't move curr_chunk_index and curr_sample_index out of the loop
            // because within the loop we might update `self.cache` which will invalidate the references
            let curr_chunk_index = if self.downloading_epoch % 2 == 0 {
                &self.chunk_index_odd_epoch
            } else {
                &self.chunk_index_even_epoch
            };

            let curr_sample_index = if self.downloading_epoch % 2 == 0 {
                &self.sample_index_odd_epoch
            } else {
                &self.sample_index_even_epoch
            };

            let chunk_index = curr_chunk_index[self.pointer_x];
            let sample_index = curr_sample_index[self.pointer_x][self.pointer_y];

            let byte_offset_start = self.chunk_index_offset[&chunk_index][sample_index as usize];
            let byte_offset_end = self.chunk_index_offset[&chunk_index][sample_index as usize + 1];

            let filename = self.chunks[chunk_index as usize].get("filename").unwrap().clone();

            let storage_provider = self.storage_provider.clone();

            let remote_dir = self.remote_dir.clone();
            let current_downloading_epoch = self.downloading_epoch.clone();
            let current_chunk_index = chunk_index.clone();
            let current_sample_index = sample_index.clone();

            tasks.spawn(async move {
                let data = storage_provider
                    .get_bytes_in_range(&filename, byte_offset_start, byte_offset_end)
                    .await;

                if let Err(e) = data {
                    panic!("failed to download data: {}", e);
                }

                let data = data.unwrap();

                return (
                    current_downloading_epoch,
                    current_chunk_index,
                    current_sample_index,
                    data,
                );
            });

            self.pointer_y = (self.pointer_y + 1) % (curr_sample_index[self.pointer_x].len());
            if self.pointer_y == 0 {
                self.pointer_x += 1;

                if self.pointer_x >= curr_chunk_index.len() {
                    self.pointer_x = 0;
                    self.downloading_epoch += 1;
                }
            }
        }

        let mut downloaded_items = Vec::new();
        while let Some(result) = tasks.join_next().await {
            let (epoch, chunk_index, sample_index, data) = result.unwrap();
            downloaded_items.push((epoch, chunk_index, sample_index, data));
        }

        Ok(downloaded_items)
    }

    /// go through chunks and download offset array for each chunk
    pub async fn get_chunk_offset(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut tasks: JoinSet<(usize, Vec<u32>)> = JoinSet::new();

        for chunk_index in 0..self.chunks.len() {
            let storage_provider = self.storage_provider.clone();
            let remote_dir = self.remote_dir.clone();
            let chunks = self.chunks.clone();
            let filename = self.chunks[chunk_index as usize].get("filename").unwrap().clone();

            tasks.spawn(async move {
                let range_start = 4; // first 4 bytes of chunk store number of samples in the chunk
                let items_in_chunk = chunks[chunk_index as usize]
                    .get("chunk_size")
                    .unwrap()
                    .parse::<u32>()
                    .unwrap();

                let range_end = (1 + items_in_chunk) * 4;

                let offset_array_bytes = storage_provider
                    .get_bytes_in_range(&filename, range_start, range_end)
                    .await;
                if let Err(e) = offset_array_bytes {
                    panic!("failed to download offset array: {}", e);
                }
                let offset_array_bytes = offset_array_bytes.unwrap();

                // Convert bytes to u32
                let offset_array: Vec<u32> = offset_array_bytes
                    .chunks(4) // Group bytes into chunks of 4
                    .map(|chunk| {
                        let mut buf = [0u8; 4];
                        buf.copy_from_slice(chunk);
                        u32::from_le_bytes(buf) // Convert 4 bytes into a u32 (little-endian)
                    })
                    .collect();

                return (chunk_index, offset_array);
            });
        }

        // Wait for all tasks to finish
        while let Some(result) = tasks.join_next().await {
            if let Err(e) = result {
                panic!("failed to download offset array: {}", e);
            }
            let (chunk_index, offset_array) = result.unwrap();
            self.chunk_index_offset
                .insert(chunk_index as u32, offset_array);
        }

        Ok(())
    }

    /// get bytes for a given chunk index and sample index
    ///
    /// Chunk binary format:
    ///
    ///         +------------+---------------+-------------+
    ///         | num_items  | offset_array  | item_data   |
    ///         +------------+---------------+-------------+
    ///         | uint32     | uint32[N+1]   | bytes       |
    ///         | 4 bytes    | 4*(N+1) bytes | variable    |
    ///         +------------+---------------+-------------+
    ///
    /// To load sample at index `i`:
    /// We need to get offset array from `offset_array[i]` to `offset_array[i+1]`
    /// and read bytes from `offset_start` to `offset_end` to get the item bytes.
    pub async fn get_bytes_for_chunk_index_and_sample_index(
        &self,
        chunk_index: u32,
        sample_index: u32,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let byte_offset_start = self.chunk_index_offset[&chunk_index][sample_index as usize];
        let byte_offset_end = self.chunk_index_offset[&chunk_index][sample_index as usize + 1];

        let filename = self.chunks[chunk_index as usize].get("filename").unwrap();

        self.storage_provider
            .get_bytes_in_range(filename, byte_offset_start, byte_offset_end)
            .await
    }
}

#[pymethods]
impl StreamingDataProvider {
    #[new]
    pub fn new(
        epoch: u32,
        remote_dir: String,
        chunks: Vec<HashMap<String, String>>,
        on_start_pre_item_download_count: u32,
        get_next_k_item_count: u32,
    ) -> Self {
        let mut provider = StreamingDataProvider {
            downloading_epoch: epoch,
            streaming_epoch: epoch,
            remote_dir: String::from(&remote_dir),
            chunks: chunks,
            chunk_index_odd_epoch: Vec::new(),
            chunk_index_even_epoch: Vec::new(),
            sample_index_odd_epoch: Vec::new(),
            sample_index_even_epoch: Vec::new(),
            chunk_index_offset: HashMap::new(),
            pointer_x: 0,
            pointer_y: 0,
            on_start_pre_item_download_count: on_start_pre_item_download_count,
            get_next_k_item_count: get_next_k_item_count,
            storage_provider: get_storage_backend(&remote_dir),
        };

        provider.on_start();

        provider
    }

    /// on_start
    ///     -> download offset array for all chunk indexes in parallel
    ///     -> download `pre_item_download_count` items in advance
    ///     -> return a list of downloaded items (epoch, chunk_index, sample_index, data)
    pub fn on_start(&mut self) -> Vec<(u32, u32, u32, Vec<u8>)> {
        // first we need to download offset array for all chunk indexes in parallel
        // and store them in `chunk_index_offset` map
        let rt = tokio::runtime::Runtime::new().unwrap();

        // get offset arrays for odd and even epochs
        rt.block_on(self.get_chunk_offset()).unwrap();

        // fetch `pre_item_download_count`
        let downloaded_items =
            rt.block_on(self.pre_fetch_items_on_start(self.on_start_pre_item_download_count));

        if let Err(e) = downloaded_items {
            panic!("failed to download items on start: {}", e);
        }

        let downloaded_items = downloaded_items.unwrap();

        return downloaded_items;
    }

    pub fn get_next_k_item(&mut self) -> Vec<(u32, u32, u32, Vec<u8>)> {
        let rt = tokio::runtime::Runtime::new().unwrap();

        let downloaded_items =
            rt.block_on(self.pre_fetch_items_on_start(self.get_next_k_item_count));

        if let Err(e) = downloaded_items {
            panic!("failed to download items on get_next_k_item: {}", e);
        }

        let downloaded_items = downloaded_items.unwrap();

        return downloaded_items;
    }

    pub fn set_epoch(&mut self, epoch: u32) {
        // update the epoch of the items being streamed
        self.streaming_epoch = epoch;
    }

    pub fn set_chunk_and_sample_index(
        &mut self,
        epoch: u32,
        chunk_index: Vec<u32>,
        sample_index: Vec<Vec<u32>>,
    ) {
        // set chunk_index and sample_index in {odd/even} depending on epoch.
        if epoch % 2 == 0 {
            self.chunk_index_odd_epoch = chunk_index;
            self.sample_index_odd_epoch = sample_index;
        } else {
            self.chunk_index_even_epoch = chunk_index;
            self.sample_index_even_epoch = sample_index;
        }
    }

    pub fn set_chunk(&mut self, epoch: u32, chunk_index: Vec<u32>) {
        // set chunk_index and sample_index in {odd/even} depending on epoch.
        if epoch % 2 == 0 {
            self.chunk_index_odd_epoch = chunk_index;
        } else {
            self.chunk_index_even_epoch = chunk_index;
        }
    }

    pub fn set_sample_index(&mut self, epoch: u32, sample_index: Vec<Vec<u32>>) {
        if epoch % 2 == 0 {
            self.sample_index_odd_epoch = sample_index;
        } else {
            self.sample_index_even_epoch = sample_index;
        }
    }
}
