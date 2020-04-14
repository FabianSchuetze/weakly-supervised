#!/usr/bin/python
r"""
Main file used to initialize and train a model.
"""
from typing import Tuple
import torch
from torch.utils.data.dataloader import DataLoader
from transferlearning.data import VaihingenDataBase
from transferlearning.data import PennFudanDataset
from transferlearning import Supervised, Processing
from transferlearning import eval_masks, print_evaluation, eval_metrics
import transferlearning


def collate_fn(batch):
    return tuple(zip(*batch))


def get_transform(training: bool):
    """
    Transforms raw data for train and test. Passed to the Database clases
    """
    transforms = []
    transforms.append(transferlearning.ToTensor()) # convert PIL image to tensor
    if training:
        transforms.append(transferlearning.RandomHorizontalFlip(0.5))
    return transferlearning.Compose(transforms)


def train_test(database: transferlearning.data, path: str)\
        -> Tuple[DataLoader, DataLoader]:
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
    dataset = database(path, get_transform(training=True))
    dataset_test = database(path, get_transform(training=False))
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:500])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[500:550])
    # dataset = torch.utils.data.Subset(dataset, indices[:-50])
    # dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0,
        collate_fn=collate_fn)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collate_fn)
    return data_loader, data_loader_test


if __name__ == "__main__":
    DEVICE = torch.device('cpu')
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    DATA, DATA_TEST = train_test(VaihingenDataBase, 'data/vaihingen')
    # DATA, DATA_TEST = train_test(PennFudanDataset, 'data/PennFudanPed')
    N_GROUPS = 5
    MEAN_DATA = [0.485, 0.456, 0.406]
    STDV_DATA = [0.229, 0.224, 0.225]
    PROCESSING = Processing(200, 200, MEAN_DATA, STDV_DATA)
    MODEL = Supervised(N_GROUPS, PROCESSING, weakly_supervised=True)
    MODEL.to(DEVICE)
    PARAMS = [p for p in MODEL.parameters() if p.requires_grad]
    OPT = torch.optim.SGD(PARAMS, lr=0.005, momentum=0.9, weight_decay=0.0005)
    LR_SCHEDULER = torch.optim.lr_scheduler.StepLR(OPT, step_size=3, gamma=0.1)
    # import pdb; pdb.set_trace()
    for epoch in range(2):
        transferlearning.train(DATA, OPT, MODEL, DEVICE, epoch, 20)
        LR_SCHEDULER.step()
        pred, gt, imgs = transferlearning.evaluate(MODEL, DATA_TEST, DEVICE)
        res = eval_metrics(pred, gt, ['box', 'segm'])
        print_evaluation(res)
