#!/usr/bin/python
r"""
Main file used to initialize and train a model.
"""
# from typing import Tuple
import torch
from transferlearning.data import VaihingenDataBase, PennFudanDataset, CocoDB
from transferlearning import Supervised, Processing
from transferlearning import print_evaluation, eval_metrics
from transferlearning.config import conf
from transferlearning.data.data_loader import train_test
import transferlearning


def get_transform(training: bool):
    """
    Transforms raw data for train and test. Passed to the Database clases
    """
    transforms = []
    transforms.append(transferlearning.ToTensor()) # convert PIL image to tensor
    if training:
        transforms.append(transferlearning.RandomHorizontalFlip(0.5))
    return transferlearning.Compose(transforms)


if __name__ == "__main__":
    DEVICE = torch.device('cpu')
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    DB = VaihingenDataBase('data/vaihingen', get_transform(training=True))
    DB_BOX = VaihingenDataBase('data/vaihingen', get_transform(training=True))
    DB_TEST = VaihingenDataBase('data/vaihingen', get_transform(training=False))
    CONFIG = conf("Vaihingen")
    DATASETS = train_test([DB, DB_BOX, DB_TEST], [100, 100, 100], CONFIG)
    PROCESSING = Processing(CONFIG.min_size, CONFIG.max_size, CONFIG.mean,
                            CONFIG.std)
    MODEL = Supervised(CONFIG.num_classes, PROCESSING, weakly_supervised=True)
    MODEL.to(DEVICE)
    PARAMS = [p for p in MODEL.parameters() if p.requires_grad]
    OPT = torch.optim.SGD(PARAMS, lr=0.005, momentum=0.9, weight_decay=0.0005)
    LR_SCHEDULER = torch.optim.lr_scheduler.StepLR(OPT, step_size=3, gamma=0.1)
    for epoch in range(10):
        transferlearning.train(DATASETS[0], OPT, MODEL, DEVICE, epoch, 50)
        # transferlearning.train_transfer(DATASETS[0], DATASETS[1],
                                        # OPT, MODEL, DEVICE, epoch, 50)
        LR_SCHEDULER.step()
        pred, gt, imgs = transferlearning.evaluate(MODEL, DATASETS[2], DEVICE)
        res = eval_metrics(pred, gt, ['box', 'segm'])
        print_evaluation(res)
