#!/usr/bin/python
r"""
Main file used to initialize and train a model.
"""
# from typing import Tuple
import argparse
import torch
import torch.optim as optim
import datetime
import easydict
from torch.utils.tensorboard import SummaryWriter
from transferlearning.data import VaihingenDataBase, PennFudanDataset, CocoDB,\
        train_test, PascalVOCDB
from transferlearning import Supervised, Processing
from transferlearning import print_evaluation, eval_metrics
from transferlearning.config import conf
from transferlearning import logging
import transferlearning

# import resource
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (3096, rlimit[1]))



def transform(training: bool):
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
                        action='store_true')
    parser.add_argument('--supervised', dest='weakly_supervised',
                        action='store_false')
    parser.add_argument('--only_boxes', dest='only_boxes',
                        action='store_true')
    parser.set_defaults(weakly_supervised=True)
    parser.set_defaults(only_boxes=False)
    args = parser.parse_args()
    return args


def print_config(conf_dict: easydict) ->None:
    """Pretty prints the config"""
    time = datetime.datetime.now()
    print("\nThe config at " + time.strftime("%Y-%m-%d") + " is:")
    for key in conf_dict:
        print(key + ': ' + str(conf_dict[key]))
    print('To change the parameters, please edit\n'
          './transferlearing/config/agnostic_config.py for general paras\n'\
          'and ./transferlearning/config/dataset_config.py for'\
          'data-preprocessing paras')


def get_db(config):
    """
    Returns the correct databases for training
    """
    if config.dataset == 'Vaihingen':
        db = VaihingenDataBase(config.root_folder, transform(True))
        db_box = VaihingenDataBase(config.root_folder, transform(True))
        db_test = VaihingenDataBase(config.root_folder, transform(False))
    elif config.dataset == 'Pascal':
        # Pascal only has boxes, so copy the database
        db = PascalVOCDB(config.root_folder, config.year,
                         transforms=transform(True))
        db_box = PascalVOCDB(config.root_folder, config.year,
                         transforms=transform(True))
        db_test = PascalVOCDB(config.root_folder, config.year,
                              transforms=transform(False))
    else:
        print("Implement properly")
    return db, db_box, db_test


if __name__ == "__main__":
    CLARGS = parse_args()
    CONFIG = conf(CLARGS.dataset, CLARGS)
    DEVICE = torch.device('cpu')
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    DBS = get_db(CONFIG)
    # DB = VaihingenDataBase('data/vaihingen', get_transform(training=True))
    # DB_BOX = VaihingenDataBase('data/vaihingen', get_transform(training=True))
    # DB_TEST = VaihingenDataBase('data/vaihingen', get_transform(training=False))
    print_config(CONFIG)
    DATASETS = train_test(DBS, [40, 1000, 500], CONFIG)
    PROCESSING = Processing(CONFIG.min_size, CONFIG.max_size, CONFIG.mean,
                            CONFIG.std)
    MODEL = Supervised(CONFIG.num_classes, PROCESSING,
                       weakly_supervised=CONFIG.weakly_supervised,
                       only_boxes=CONFIG.only_boxes)
    MODEL.to(DEVICE)
    PARAMS = [p for p in MODEL.parameters() if p.requires_grad]
    OPT = optim.SGD(PARAMS, lr=CONFIG.learning_rate, momentum=CONFIG.momentum,
                    weight_decay=CONFIG.weight_decay)
    SCHEDULER = optim.lr_scheduler.StepLR(OPT,
                                          step_size=CONFIG.decay_step_size,
                                          gamma=CONFIG.gamma)
    WRITER = SummaryWriter()
    logging.log_architecture(WRITER, MODEL, DATASETS, OPT, CONFIG.dataset)
    transferlearning.train(DATASETS, OPT, MODEL, DEVICE, CONFIG, WRITER, SCHEDULER)
    pred, gt, _ = transferlearning.evaluate(MODEL, DATASETS[2], DEVICE,
                                            CONFIG.max_epochs + 1,
                                            CONFIG.display_iter)
    res = eval_metrics(pred, gt, CONFIG.loss_types)
    print_evaluation(res)
    logging.log_accuracies(WRITER, res, CONFIG.max_epochs + 1)
    WRITER.close()
