# -*- coding: utf-8 -*-
# @Author: huangyangyu
# @Author: The-Hierophant
# @Date:   2024-08-16 00:01:59
# @Last Modified by:   The-Hierophant
# @Last Modified time: 2024-08-16 00:11:35
import os
import argparse
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import smart_open
from datasets import load_dataset, Dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from multiprocessing import Process, Manager, Queue
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import time


DOWNLOAD_WORKERS_PER_PROCESS = 4

def download_task(s3_client, row, idx, download_queue, write_queue, progress_queue):
    blob_id, src_encoding = row["blob_id"], row["src_encoding"]
    s3_url = f"s3://softwareheritage/content/{blob_id}"
    start_time = time.time()
    try:
        with smart_open.open(s3_url, "rb", compression=".gz", transport_params={"client": s3_client}) as fin:
            content = fin.read()
            decoded_content = content.decode(src_encoding)
            write_queue.put((blob_id, decoded_content, src_encoding))
            elapsed_time = time.time() - start_time
            progress_queue.put((len(content), elapsed_time))
    except Exception as e:
        print(f"Failed to download {blob_id}: {e}")
    download_queue.task_done()

def monitor_progress(progress_queue: Queue, pbar: tqdm):
    total_size = 0
    smoothed_speed = 0
    
    start_time = time.time()
    count_idx = 0
    while True:
        item = progress_queue.get()
        count_idx += 1
        size, elapsed_time = item
        total_size += size

        if item is None:
            break

        # instant_speed = size / (elapsed_time * 1024 * 1024)  # Convert to MB/s
        smoothed_speed = total_size / ((time.time() - start_time) * 1024 * 1024)
        
        pbar.set_postfix(speed=f"{smoothed_speed:.2f} MB/s")
        pbar.update(1)
        if count_idx > 2000:
            count_idx = 0
            total_size = 0
            start_time = time.time()

def write_to_parquet(write_queue: Queue, output_parquet_path):
    buffer, file_counter = [], 0
    while True:
        item = write_queue.get()
        if item is None:
            break
        blob_id, decoded_content, src_encoding = item
        buffer.append({
            "blob_id": blob_id,
            "content": decoded_content,
            "encoding": src_encoding
        })
        if len(buffer) >= 1000000:  # Adjust buffer size to 1M records
            save_parquet(buffer, output_parquet_path, file_counter)
            buffer.clear()
            file_counter += 1
    if buffer:
        save_parquet(buffer, output_parquet_path, file_counter)  # Save remaining records

def save_parquet(buffer, path, rank):
    df = pd.DataFrame(buffer)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, os.path.join(path, f'part-{rank}.parquet'), compression='SNAPPY')

def process_chunk(num_chunks, chunk_idx, ds: Dataset, download_queue, write_queue, progress_queue):
    chunk = ds.shard(num_chunks, chunk_idx)
    if chunk_idx == 0:
        print("shard complete")
    del ds
    s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED, 
                                                 max_pool_connections=DOWNLOAD_WORKERS_PER_PROCESS * 2,
                                                 proxies_config=None))
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS_PER_PROCESS) as executor:
        futures = []
        for idx, row in enumerate(chunk):
            futures.append(executor.submit(download_task, s3_client, row, idx, download_queue, write_queue, progress_queue))
        for future in as_completed(futures):
            future.result()

def main(data_repo, language, hug_access_token, download_folder, max_workers):
    manager = Manager()
    download_queue = manager.Queue()
    write_queue = manager.Queue()
    progress_queue = manager.Queue()

    ds = load_dataset(data_repo, language, split="train", streaming=False, token=hug_access_token)
    total_files = len(ds)


    num_chunks = max_workers
    # chunks = [ds.shard(num_chunks, chunk_idx, keep_in_memory=True) for chunk_idx in range(num_chunks)]
    # print("shard complete")

    # Start writing and monitoring processes
    write_proc = Process(target=write_to_parquet, args=(write_queue, download_folder))
    monitor_proc = Process(target=monitor_progress, args=(progress_queue, tqdm(total=total_files, desc="Progress", unit="file")))
    write_proc.start()
    monitor_proc.start()
    # Process chunks in parallel using multiple processes
    processes = []
    for chunk_idx in range(num_chunks):
        p = Process(target=process_chunk, args=(num_chunks, chunk_idx, ds, download_queue, write_queue, progress_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # stop signal
    for _ in range(max_workers * DOWNLOAD_WORKERS_PER_PROCESS):  # signal all of them to stop
        download_queue.put(None)

    download_queue.join()

    # stop signal
    write_queue.put(None)
    write_proc.join()

    # stop signal
    progress_queue.put(None)
    monitor_proc.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The Stack V2 Download Script")
    parser.add_argument("--data_repo", type=str, default="bigcode/the-stack-v2", help="The data repo name.")
    parser.add_argument("--language", type=str, default="Python", help="The programming language name, None is the whole dataset.")
    parser.add_argument("--hug_access_token", type=str, default="your_huggingface_access_token", help="The access token of huggingface account, which could be acquired from https://huggingface.co/settings/tokens.")
    parser.add_argument("--download_folder", type=str, default=".", help="The folder path to download the data.")
    parser.add_argument("--max_workers", type=int, default=5, help="The number of concurrent download workers.")
    args = parser.parse_args()
    main(args.data_repo, args.language, args.hug_access_token, args.download_folder, args.max_workers)

