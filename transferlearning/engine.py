r"""
Methods to train and evaluate the model
"""
import sys
import math
import time
from typing import Dict, Optional, List
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from transferlearning import logging


@torch.no_grad()
def evaluate(model: torch.nn.Module, data: DataLoader,
             device: torch.device, epoch: int, print_freq: int) -> None:
    """Evaluates the model

    Parameters
    ----------
    model: troch.nn.Module
        The model used to predict outputs

    data_loader: DataLoader
        The class from which samples are drawn

    device: torch.device
        device indication whether to run on cpu or gpu

    Returns
    -------
    Tuple[List, List, List]
        Different Lists containing the predictions, targets, and input data
        used for evaluation
    """
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    all_targets, all_preds, all_images = [], [], []
    logger = get_logging(training=False)
    header = 'Epoch Val: [{}]'.format(epoch)
    for images, targets in logger.log_every(data, print_freq, header):
        all_images.append(images[0])
        all_targets.append(targets[0])
        images = list(i.to(device) for i in images)
        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)
        model_time = time.time() - model_time
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        all_preds.append(outputs[0])
        logger.update(model_time=model_time)
    torch.set_num_threads(n_threads)
    return all_preds, all_targets, all_images


def check_losses(losses: float, loss_dict: Dict, target) -> None:
    """
    Checks if the losses are finite. If not exist the program, after
    printing a report to stdout
    """
    if not math.isfinite(losses):
        print("\nA loss is not finite.")
        for key in loss_dict:
            rounded = np.round(loss_dict[key].item(), 2)
            print(key + ': ' + str(rounded))
        print("The error occured with sample id: %s"\
              %(str(target[0]['image_id'])))
        sys.exit(1)


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)\
        -> torch.optim.lr_scheduler.LambdaLR:
    """Determines the learning rate"""

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def learning_rate_scheduler(optimizer: torch.optim, epoch: int, n_steps: int)\
        -> Optional[torch.optim.lr_scheduler.LambdaLR]:
    """
    The lr scheduler in the first epoch used for warum of the lr. The schedule
    is linearly increasing

    Parameters
    ----------
    optimizer: torch.optim
        The gradient descent optimizer to use

    eoch: int
        The number of the current epoch.

    n_steps: int
        How many steps are done.

    Returns
    -------
    Optional[torch.optim.lr_scheduler.LambdaLR]
        Possibly a lr scheduler
    """
    if epoch != 0:
        return None
    warmup_factor = 1. / 1000
    warmup_iters = min(1000, n_steps - 1)
    lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    return lr_scheduler


def get_logging(training=True):
    """
    Instantiates a logging class. The class both writes information to
    stdout and also iterates over the dataset

    Parameters
    ----------
    training: bool
        If true, losses are smoothed
    """
    metric_logger = logging.MetricLogger(delimiter="  ")
    if training:
        metric_logger.add_meter(
            'lr', logging.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    return metric_logger


def train_supervised(datasets: List[DataLoader], optimizer: torch.optim,
                     model: torch.nn.Module, device: torch.device,
                     epoch: int, print_freq: int,
                     writer: Optional[SummaryWriter] = None,
                     writer_iter: Optional[int] = None) -> int:
    """Trains the model

    Parameters
    ----------
    datasets: List[dataloader]
        the dataloader used for training

    optimizer: troch.optim
        The optimizer used during training for gradient descent

    model: subclass of troch.nn.Module
        The nn

    device: troch.device
        Location for where to put the data

    epoch: int
        Current epoch number

    print_freq: int
        Determines after how many training iterations notificaitons are printed
        to stdout

    writer: Optional[SummaryWriter]
        Tensorboard usmmary writter to save experiments

    writer_iter: Optional[int]
        Species the gradient steps (location) for the writer
    """
    data = datasets[0]
    model.train()
    model.to(device)
    logger = get_logging(training=True)
    header = 'Epoch: [{}]'.format(epoch)
    lr_scheduler = learning_rate_scheduler(optimizer, epoch, len(data))
    for images, targets in logger.log_every(data, print_freq, header):
        optimizer.zero_grad()
        images = list(i.to(device) for i in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        if writer:
            logging.log_losses(writer, loss_dict, print_freq, writer_iter)
        check_losses(losses, loss_dict, targets)
        losses.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        logger.update(loss=losses, **loss_dict)
        logger.update(lr=optimizer.param_groups[0]["lr"])
        writer_iter += 1
    return writer_iter

def train_transfer(datasets: List[DataLoader], optimizer: torch.optim,
                   model: torch.nn.Module, device: torch.device,
                   epoch: int, print_freq: int,
                   writer: Optional[SummaryWriter] = None,
                   writer_iter: Optional[int] = None) -> int:
    """Implements the simple stage-wise training of 'Segment Every Thing'.
    Arguments are the same as in the supervised function"""
    data_mask = datasets[0]
    data_box = datasets[1]
    for para in model.parameters():
        para.requires_grad = True
    # import pdb; pdb.set_trace()
    model._heads.train_mask = False  # Doesn't train the mask head
    writer_iter = train_supervised([data_box], optimizer, model, device,
                                   epoch, print_freq, writer, writer_iter)
    model._heads.train_mask = True
    writer_iter = train_supervised([data_mask], optimizer, model, device, epoch,
                                   print_freq, writer, writer_iter)
    return writer_iter
