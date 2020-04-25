r"""
Some utils used in the data loading functions
"""
from typing import List, Dict
import os
import torch
from torch.utils.data import DataLoader
import numpy as np


def to_dict(masks: List, bboxes: List, labels: List, img_info: List,
            idx: int, area: List) -> Dict[str, torch.Tensor]:
    """
    Converts the inputs to pytorch.tensor type and puts them into a dict
    """
    target = {}
    target['boxes'] = torch.as_tensor(bboxes, dtype=torch.float32)
    target['masks'] = torch.as_tensor(masks, dtype=torch.uint8)
    target['img_info'] = torch.as_tensor(img_info, dtype=torch.float32)
    target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
    target['image_id'] = torch.tensor([idx])
    target['area'] = torch.as_tensor(area, dtype=torch.float32)
    return target

def extract_boxes(masks: List[np.array]) -> np.array:
    """Finds the bounding boxes of each masks

    Returns
    -------
    bboxes: np.array
        a numpy array with (xmin, ymin, width, height)i
    """
    bboxes = np.zeros((len(masks), 4))
    # import pdb; pdb.set_trace()
    for idx, img in enumerate(masks):
        pos = np.where(img)
        xmin = pos[1].min()
        ymin = pos[0].min()
        xmax = pos[1].max()
        ymax = pos[0].max()
        bboxes[idx, :] = [xmin, ymin, xmax, ymax]
    return bboxes

def check_area(img: np.array) -> bool:
    """Checks if img has a positive area"""
    pos = np.where(img)
    xmin = pos[1].min()
    ymin = pos[0].min()
    xmax = pos[1].max()
    ymax = pos[0].max()
    return xmin != xmax and ymin != ymax


def sample_stats(dataset, n_samples, fun='mean'):
    """Returns the sample value of function fun accross n_samples in the
    dataset"""
    means = torch.zeros((n_samples, 3))
    for i in range(n_samples):
        idx = np.random.randint(0, len(dataset))
        img = dataset[idx][0]
        if fun == 'mean':
            means[i, :] = img.mean(dim=(1, 2))
        elif fun == 'std':
            means[i, :] = img.std(dim=(1, 2))
    return means


def _split(a: List, n: int):
    """
    Splitting the list a into n equally sizes (if possible) sublists. Returns
    a generator. Credit to: stackoverflow.com/questions/312443
    """
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def split_work(to_split: List, n_workers: int) -> List[List]:
    """
    Interacts with the split function and returns a list
    """
    splits = _split(to_split, n_workers)
    return [i for i in splits]


def find_missing_files(search_dir: str, required_indices: List[int]) -> List[int]:
    """
    Searches the search_dir for a filename pattern and returns all indices
    which are not in search_dir but in required_indices

    Parameters
    ----------
    search_dir: str
        The directory to search through

    required_indices: List[int]
        Within search dir, every file should be pickle according to its dbs
        indices. The file contains these indices

    Returns
    ------
    List[int]
        The indices which are not on the hdd
    """
    existing = []
    for file in os.listdir(search_dir):
        if os.path.isfile(os.path.join(search_dir, file)):
            file = file.split('.')[0]
            try:
                existing.append(int(file))
            except ValueError:
                continue
    remaining = [i for i in required_indices if i not in existing]
    return remaining


def shuffle_targets(targets):
    """
    Shuffles the targets
    """
    import pdb; pdb.set_trace()
    indices = np.arange(len(targets['labels']))
    np.random.shuffle(indices)
    targets['labels'] = [targets['labels'][i] for i in indices]
    targets['boxes'] = [targets['boxes'][i] for i in indices]
    targets['area'] = [targets['area'][i] for i in indices]
    if 'masks' in targets.keys():
        targets['masks'] = [targets['masks'][i] for i in indices]
    return targets
