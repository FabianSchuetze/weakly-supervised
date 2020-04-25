r"""
Returns the correct databases used for train and testing
"""
from typing import List
import torch
from torch.utils.data import DataLoader, Subset
import easydict
from transferlearning.data import VaihingenDataBase, PennFudanDataset, CocoDB,\
        PascalVOCDB
import transferlearning


def collate_fn(batch):
    return tuple(zip(*batch))


def transform(training: bool):
    """
    Transforms raw data for train and test. Passed to the Database clases
    """
    transforms = []
    transforms.append(transferlearning.ToTensor()) # convert PIL image to tensor
    if training:
        transforms.append(transferlearning.RandomHorizontalFlip(0.5))
    return transferlearning.Compose(transforms)


def get_dbs(config):
    """
    Returns the correct databases for training
    """
    if config.dataset == 'Vaihingen':
        db = VaihingenDataBase(config.root_folder,
                               transforms=transform(True), train=True)
        db_test = VaihingenDataBase(config.root_folder,
                                    transforms=transform(False), train=False)
    if config.dataset == 'Pascal':
        db = PascalVOCDB(config.root_folder, config.year,
                         transforms=transform(True), train=True)
        db_test = PascalVOCDB(config.root_folder, config.year,
                              transforms=transform(False), train=False)
    return [db, db_test]


def generate_train_splits(config: easydict, size: int) -> List[List[int]]:
    """
    Generates train test splits
    """
    all_indices = torch.randperm(size).tolist()
    cutoff = int(size * config.train_test_split)
    train, test = all_indices[:cutoff], all_indices[cutoff:]
    if config.weakly_supervised:
        cutoff = int(len(train) * config.transfer_split)
        supervised, transfer = train[:cutoff], train[cutoff:]
        return [supervised, transfer, test]
    return [train, test]


def train_test(databases: List, config: easydict) -> List[DataLoader]:
    """Returns the train and val set

    Parameters
    ----------
    database: List[transferlearning.data]
        Class used to load train and val examples

    config: easydict
        Parameters

    Returns
    -------
    List[DataLoader]:
        Databases used for training and validation
    """
    indices = generate_train_splits(config, len(databases[0]))
    test_it = len(indices) - 1
    datasets = []
    for it, index in enumerate(indices):
        db = databases[1] if it == test_it else databases[0]
        subset = Subset(db, index)
        shuffle = it < test_it
        num_workers = config.num_workers if it < test_it  else 0
        datasets.append(DataLoader(subset, batch_size=config.batch_size,
                                   shuffle=shuffle,
                                   num_workers=num_workers,
                                   collate_fn=collate_fn))
    return datasets
