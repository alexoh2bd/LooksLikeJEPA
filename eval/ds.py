
import torch
# import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import v2
from datasets import load_dataset

import random
import numpy as np


def collate_views(batch):
    """
    Custom collate function that pre-stacks views by resolution.
    
    This is more memory efficient than the default collate because:
    1. Views are stacked on CPU before GPU transfer (single large transfer vs many small ones)
    2. Views of the same resolution are batched together for efficient processing
    
    Args:
        batch: List of (views, label) tuples from dataset
        
    Returns:
        stacked_views: List of tensors, each (B, C, H, W) for a resolution group
        labels: Tensor of labels (B,)
    """
    # Collect all views grouped by resolution
    views_by_size = {}
    labels = []
    
    for views, label in batch:
        labels.append(label)
        for v in views:
            size = v.shape[-1]  # Use width as resolution key
            if size not in views_by_size:
                views_by_size[size] = []
            views_by_size[size].append(v)
    
    stacked_views = []
    for size in sorted(views_by_size.keys(), reverse=True):
        # Stack all views of this size: (num_views_of_this_size, C, H, W)
        stacked = torch.stack(views_by_size[size])
        # Reshape to (B, num_views_per_sample, C, H, W) then back to list of (B, C, H, W)
        batch_size = len(batch)
        num_views_per_sample = len(views_by_size[size]) // batch_size
        # Reshape: (B * num_views, C, H, W) -> (B, num_views, C, H, W)
        stacked = stacked.reshape(batch_size, num_views_per_sample, *stacked.shape[1:])
        # Add each view position as separate tensor for compatibility with existing code
        for v_idx in range(num_views_per_sample):
            stacked_views.append(stacked[:, v_idx])
    
    labels = torch.tensor(labels, dtype=torch.long)
    
    return stacked_views, labels


class HFDataset(Dataset):
    def __init__(self, split, V_global=2, V_local=4, device="cuda", global_img_size=224, local_img_size=96, dataset="inet100"):
        self.V_global = V_global
        self.V_local = V_local
        self.split = split
        self.global_img_size = global_img_size
        self.local_img_size = local_img_size
        
        self._get_ds(dataset)
        
        # 2. Define Transforms
        # Global Views: 224x224
        self.global_transform = v2.Compose([
            v2.RandomResizedCrop(self.global_img_size, scale=(0.08, 1.0)),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
        ])
        
        # Local Views: 96x96
        self.local_transform = v2.Compose([
            v2.RandomResizedCrop(self.local_img_size, scale=(0.05, 0.4)),
            v2.ToImage(),
        ])

        # Test transform
        self.test_transform = v2.Compose([
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.bfloat16, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    def _get_ds(self, dataset):
        if dataset == "cifar10":
            self.ds = load_dataset("cifar10", split=self.split)
        elif dataset == "inet100":
            self.inet_dir = "/home/users/aho13/jepa_tests/data/cache/datasets--clane9--imagenet-100/snapshots/0519dc2f402a3a18c6e57f7913db059215eee25b/data/"
            filenames = {
                "train": self.inet_dir + "train-*.parquet",
                "val": self.inet_dir + "validation*.parquet",
            }
            self.ds = load_dataset("parquet", data_files=filenames, split=self.split)
        elif dataset=="imagenet-1k":
            self.inet_dir = "/home/users/aho13/jepa_tests/data/hub/datasets--ILSVRC--imagenet-1k/snapshots/49e2ee26f3810fb5a7536bbf732a7b07389a47b5/data"
            
            filenames = {
                "train": self.inet_dir + "train*.parquet",
                "val": self.inet_dir + "validation*.parquet",
                "test": self.inet_dir + "test*.parquet",
            }
        else:
            raise ValueError(f"Dataset {self.dataset} not supported")

    def _load_image(self, entry):
        """Helper to handle safe image extraction from row entry."""
        if "image" in entry:
            return entry["image"].convert("RGB")
        elif "img" in entry:
            return entry["img"].convert("RGB")
        else:
            raise ValueError("Image not found in entry")

    def __getitem__(self, i):
        entry = self.ds[i]
        img = self._load_image(entry)
        label = entry["label"]
        
        if self.split == 'train':
            views = []
            
            # Global Views
            if self.V_global > 0:
                views += [self.global_transform(img) for _ in range(self.V_global)]
            
            # Local Views
            if self.V_local > 0:
                views += [self.local_transform(img) for _ in range(self.V_local)]
            
            return views, label
        else:
            # Validation/Test
            return [self.test_transform(img)], label

    def __len__(self):
        return len(self.ds)


class CrossInstanceDataset(HFDataset):
    def __init__(
        self,
        split,
        V_global=2,
        V_local=3,
        V_mixed=1,
        global_img_size=224,
        local_img_size=96,
        part_upload=False,
        dataset="inet100",
    ):
        # Initialize parent (Handles loading DS, transforms, V_global/V_local)
        super().__init__(
            split=split,
            V_global=V_global,
            V_local=V_local,
            global_img_size=global_img_size,
            local_img_size=local_img_size,
            dataset=dataset
        )
        
        self.V_mixed = V_mixed
        self.label_to_indices = None

        # Build fast index mapping if training with mixed views
        if self.split == "train" and self.V_mixed > 0:
            self._build_label_index()

    def _build_label_index(self):
        """Build numpy arrays for O(1) random sampling per class."""
        labels = self.ds["label"]  # zero-copy Arrow column
        num_classes = max(labels) + 1

        # Use numpy arrays instead of lists for faster random indexing
        label_lists = [[] for _ in range(num_classes)]
        for idx, lbl in enumerate(labels):
            label_lists[lbl].append(idx)
        
        # Convert to numpy arrays for fast fancy indexing
        self.label_to_indices = [np.array(lst, dtype=np.int64) for lst in label_lists]
        # Pre-compute class sizes for bounds checking
        self.class_sizes = np.array([len(lst) for lst in self.label_to_indices], dtype=np.int64)

    def __getitem__(self, i):
        # 1. Get standard views (Global + Local) from parent
        views, label = super().__getitem__(i)

        # 2. If valid/test or no mixed views requested, return immediately
        if self.split != "train" or self.V_mixed == 0:
            return views, label

        # 3. Generate Mixed Views (Same Class, Different Instance)
        # Fast numpy random sampling - much faster than random.sample()
        class_indices = self.label_to_indices[label]
        class_size = len(class_indices)
        
        # Use numpy randint + fancy indexing (faster than random.sample)
        rand_positions = np.random.randint(0, class_size, size=self.V_mixed)
        mixed_indices = class_indices[rand_positions].tolist()
        
        # Batch fetch entries (single I/O call)
        mixed_entries = self.ds[mixed_indices]
        
        # Handle batch output from datasets library (returns dict of lists)
        if "image" in mixed_entries:
            mixed_imgs = mixed_entries["image"]
        elif "img" in mixed_entries:
            mixed_imgs = mixed_entries["img"]
        else:
            raise ValueError("Image key not found in mixed entries")

        # Transform and append - keep loop but avoid .convert() overhead when possible
        for m_img in mixed_imgs:
            rgb_img = m_img if m_img.mode == "RGB" else m_img.convert("RGB")
            views.append(self.local_transform(rgb_img))

        return views, label
