
import time
import torch
from jepa import CrossInstanceDataset
import tqdm
from torch.utils.data import DataLoader

def benchmark(dataset_name="cifar10"):
    print(f"Benchmarking with dataset={dataset_name}")
    try:
        ds = CrossInstanceDataset("train", V_global=2, V_local=3, V_mixed=4, dataset=dataset_name)
        print("Keys in first item:", ds.ds[0].keys())
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    print(f"Dataset length: {len(ds)}")
    
    loader = DataLoader(ds, batch_size=256, num_workers=4, prefetch_factor=2)
    
    start = time.time()
    count = 0
    limit = 50 # batches
    
    for _ in tqdm.tqdm(loader, total=limit):
        count += 1
        if count >= limit:
            break
            
    end = time.time()
    duration = end - start
    print(f"Processed {count} batches in {duration:.2f} seconds.")
    print(f"Speed: {count/duration:.2f} batches/sec")
    print(f"Throughput: {count * 256 / duration:.2f} samples/sec")

if __name__ == "__main__":
    benchmark("cifar10")
