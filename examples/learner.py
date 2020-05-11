#!/usr/bin/python
r"""
Main file used to initialize and train a model.
"""
import os
import argparse
import datetime
import easydict
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from transferlearning.data import get_dbs, train_test
from transferlearning import Supervised, Processing, print_evaluation, \
    eval_metrics, logging, load
from transferlearning.config import conf
import transferlearning
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def _plot_boxes(img, boxes):
    boxes = boxes.tolist()
    fig, axis = plt.subplots()
    axis.imshow(np.array(img).tranpose(1, 2, 0))
    for rec in boxes:
        width, height = rec[2] - rec[0], rec[3] - rec[1]
        patch = patches.Rectangle((rec[0], rec[1]), width, height,
                                  linewidth=1, edgecolor='r',
                                  facecolor='none')
        axis.add_patch(patch)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Command Line Arguments')
    parser.add_argument('--dataset', dest='dataset', help='training dataset',
                        default='Vaihingen', type=str)
    parser.add_argument('--load_path', dest='load_path',
                        help='path to restore model', type=str)
    parser.add_argument('--weakly_supervised', dest='weakly_supervised',
                        action='store_true')
    parser.add_argument('--supervised', dest='weakly_supervised',
                        action='store_false')
    parser.add_argument('--only_boxes', dest='only_boxes',
                        action='store_true')
    parser.add_argument('--restore', dest='restore',
                        action='store_true')
    parser.set_defaults(weakly_supervised=True)
    parser.set_defaults(only_boxes=False)
    parser.set_defaults(restore=False)
    args = parser.parse_args()
    return args


def print_config(conf_dict: easydict) -> None:
    """Pretty prints the config"""
    time = datetime.datetime.now()
    print("\nThe config at " + time.strftime("%Y-%m-%d") + " is:")
    for key in conf_dict:
        print(key + ': ' + str(conf_dict[key]))
    print('To change the parameters, please edit\n'
          './transferlearing/config/config_file.py for paras')


if __name__ == "__main__":
    CLARGS = parse_args()
    CONFIG = conf(CLARGS.dataset, CLARGS)
    DEVICE = torch.device('cpu')
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    if CLARGS.only_boxes:
        CONFIG.weakly_supervised = False
    DBS = get_dbs(CONFIG)
    print_config(CONFIG)
    DATASETS = train_test(DBS, CONFIG)
    if CONFIG.output_dir:
        NOW = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        CONFIG.output_dir = CONFIG.output_dir + '/' + NOW
        os.makedirs(CONFIG.output_dir)
    PROCESSING = Processing(CONFIG.min_size, CONFIG.max_size, CONFIG.mean,
                            CONFIG.std)
    MODEL = Supervised(CONFIG.num_classes, PROCESSING,
                       weakly_supervised=CONFIG.weakly_supervised,
                       only_boxes=CLARGS.only_boxes)
    PARAMS = [p for p in MODEL.parameters() if p.requires_grad]
    OPT = optim.SGD(PARAMS, lr=CONFIG.learning_rate, momentum=CONFIG.momentum,
                    weight_decay=CONFIG.weight_decay)
    SCHEDULER = optim.lr_scheduler.ReduceLROnPlateau(
        OPT,
        mode='max',
        patience=CONFIG.patience,
        verbose=True,
        factor=CONFIG.gamma)
    if CLARGS.restore:
        EPOCH = load(MODEL, OPT, SCHEDULER, CLARGS.load_path)
    else:
        EPOCH = 0
    MODEL._heads.score_thresh = 0.5
    MODEL.to(DEVICE)
    WRITER = SummaryWriter()
    logging.log_architecture(WRITER, MODEL, DATASETS, OPT, CLARGS.dataset)
    EPOCH = transferlearning.train(DATASETS, OPT, MODEL, DEVICE, CONFIG, EPOCH,
                                   WRITER, SCHEDULER)
    pred, gt, _ = transferlearning.evaluate(MODEL, DATASETS[-1], DEVICE,
                                            EPOCH + 1, CONFIG.display_iter,
                                            200)
    res = eval_metrics(pred, gt, CONFIG.loss_types)
    print_evaluation(res)
    logging.log_accuracies(WRITER, res, EPOCH + 1)
    WRITER.close()
    print_evaluation(res)
    logging.log_accuracies(WRITER, res, EPOCH + 1)
    # transferlearning.save(EPOCH + 1, MODEL, OPT, SCHEDULER, CONFIG)
    # transferlearning.visualize_predictions(pred, DBS[1], gt, CONFIG.output_dir)
