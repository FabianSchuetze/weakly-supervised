r"""
Some utils used in the data loading functions
"""
from typing import List, Dict
import torch
from torch.utils.data import DataLoader, Subset
import easydict
import numpy as np
import os


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


def collate_fn(batch):
    return tuple(zip(*batch))


def train_test(databases: List,
               splits: List[int], config: easydict) -> List[DataLoader]:
    """Returns the train and test set

    Parameters
    ----------
    database: transferlearning.data
        Class used to load train and test examples

    path: str
        Path to loocate the raw files on the hdd

    Returns
    -------
    Tuple[DataLoader, DataLoader]:
        Train and Val Databases
    """
    # import pdb; pdb.set_trace()
    assert len(databases) == len(splits), "Requires the same length"
    indices = torch.randperm(len(databases[0])).tolist()
    thresh = np.cumsum([0, *splits])
    assert thresh[-1] <= len(databases[0]), "to many examples selected"
    datasets = []
    for  db, lower, upper in zip(databases, thresh[:-1], thresh[1:]):
        subset = Subset(db, indices[lower:upper])
        shuffle = upper < thresh[-1]
        num_workers = config.num_workers if upper < thresh[-1] else 0
        datasets.append(DataLoader(subset, batch_size=config.batch_size,
                                   shuffle=shuffle,
                                   num_workers=num_workers,
                                   collate_fn=collate_fn))
    return datasets


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def split_work(to_split: List, n_workers: int) -> List[List]:
    splits = split(to_split, n_workers)
    return [i for i in splits]


def find_missing_files(search_dir, required_files) -> List[int]:
    """
    Fins the existing files and returns the elements that need to be
    cached
    """
    # possible = [i for i in range(0, len(self._index))]
    existing = []
    for file in os.listdir(search_dir):
        if os.path.isfile(os.path.join(search_dir, file)):
            file = file.split('.')[0]
            try:
                existing.append(int(file))
            except ValueError:
                continue
    remaining = [i for i in required_files if i not in existing]
    return remaining
