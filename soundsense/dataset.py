import numpy as np
import torch
from torch.utils.data import TensorDataset


def split_dataset(features, manifest, split):
    mask = manifest.split.to_numpy() == split
    x = torch.from_numpy(np.ascontiguousarray(features[mask, None]))
    y = torch.tensor(manifest.loc[mask, 'class_index'].to_numpy(), dtype=torch.long)
    return TensorDataset(x, y)
