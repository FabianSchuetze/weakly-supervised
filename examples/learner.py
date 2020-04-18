#!/usr/bin/python
r"""
Main file used to initialize and train a model.
"""
# from typing import Tuple
import argparse
import torch
import torch.optim as optim
import time
import easydict
from torch.utils.tensorboard import SummaryWriter
from transferlearning.data import VaihingenDataBase, PennFudanDataset, CocoDB,\
        train_test
from transferlearning import Supervised, Processing
from transferlearning import print_evaluation, eval_metrics
from transferlearning.config import conf
from transferlearning import logging
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

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Command Line Arguments')
    parser.add_argument('--dataset', dest='dataset', help='training dataset',
                        default='Vaihingen', type=str)
    parser.add_argument('--weakly_supervised', dest='weakly_supervised',
                        help='Whether to train weakly supervised or not',
                        default='True', type=bool)
    args = parser.parse_args()
    return args


def print_config(conf_dict: easydict) ->None:
    """Pretty prints the config"""
    print("\nThe config at " + str(time.time()) + " is:")
    for key in conf_dict:
        print(key + ': ' + str(conf_dict[key]))
    print('To change the parameters, please edit\n'
          './transferlearing/config/agnostic_config.py for general paras\n'\
          'and ./transferlearning/config/dataset_config.py for'\
          'data-preprocessing paras')


if __name__ == "__main__":
    CLARGS = parse_args()
    DEVICE = torch.device('cpu')
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    DB = VaihingenDataBase('data/vaihingen', get_transform(training=True))
    DB_BOX = VaihingenDataBase('data/vaihingen', get_transform(training=True))
    DB_TEST = VaihingenDataBase('data/vaihingen', get_transform(training=False))
    CONFIG = conf(CLARGS.dataset)
    print_config(CONFIG)
    DATASETS = train_test([DB, DB_BOX, DB_TEST], [50, 10, 50], CONFIG)
    PROCESSING = Processing(CONFIG.min_size, CONFIG.max_size, CONFIG.mean,
                            CONFIG.std)
    MODEL = Supervised(CONFIG.num_classes, PROCESSING, weakly_supervised=True)
    MODEL.to(DEVICE)
    PARAMS = [p for p in MODEL.parameters() if p.requires_grad]
    OPT = optim.SGD(PARAMS, lr=CONFIG.learning_rate, momentum=CONFIG.momentum,
                    weight_decay=CONFIG.weight_decay)
    SCHEDULER = optim.lr_scheduler.StepLR(OPT,
                                          step_size=CONFIG.decay_step_size,
                                          gamma=CONFIG.gamma)
    WRITER = SummaryWriter()
    # WRITER = None
    logging.log_architecture(WRITER, MODEL, DATASETS, OPT, CLARGS.dataset)
    WRITER_ITER = 0
    if CLARGS.weakly_supervised:
        train = transferlearning.train_transfer
    else:
        train = transferlearning.train_supervised
    # import pdb; pdb.set_trace()
    for epoch in range(10):
        WRITER_ITER = train(DATASETS, OPT, MODEL, DEVICE, epoch,
                            CONFIG.display_iter, WRITER, WRITER_ITER)
        pred, gt, imgs = transferlearning.evaluate(MODEL, DATASETS[2], DEVICE,
                                                   epoch, CONFIG.display_iter)
        SCHEDULER.step()
        res = eval_metrics(pred, gt, ['box', 'segm'])
        print_evaluation(res)
        logging.log_accuracies(WRITER, res, epoch)
    WRITER.close()
