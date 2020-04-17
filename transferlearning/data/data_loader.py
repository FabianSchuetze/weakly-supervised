r"""
Returns the dataloaders for train and testing
"""
from typing import List
import easydict
from torch.utils.data import DataLoader, Subset
import torch
import numpy as np
from .data_utils import collate_fn


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
                                   shuffle=shuffle, num_workers=num_workers,
                                   collate_fn=collate_fn))
    return datasets
