r"""
Methods to train and evaluate the model
"""
import sys
import math
from typing import Dict, Optional
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import transferlearning.logging as logging


@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader: DataLoader,
             device: torch.device) -> None:
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
    cpu_device = torch.device("cpu")
    model.eval()
    dataset = iter(data_loader)
    all_targets, all_preds, all_images = [], [], []
    for _ in range(len(dataset)):
        images, targets = next(dataset)
        all_images.append(images[0])
        all_targets.append(targets[0])
        images = list(i.to(device) for i in images)
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        all_preds.append(outputs[0])
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


def get_logging():
    """
    Instantiates a logging class. The class both writes information to
    stdout and also iterates over the dataset
    """
    metric_logger = logging.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', logging.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    return metric_logger


def train(data: DataLoader, optimizer: torch.optim, model: torch.nn.Module,
          device: torch.device, epoch: int, print_freq: int) -> None:
    """Trains the model

    Parameters
    ----------
    data: dataloader
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
    """
    model.train()
    model.to(device)
    logger = get_logging()
    header = 'Epoch: [{}]'.format(epoch)
    lr_scheduler = learning_rate_scheduler(optimizer, epoch, len(data))
    for images, targets in logger.log_every(data, print_freq, header):
        optimizer.zero_grad()
        images = list(i.to(device) for i in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        check_losses(losses, loss_dict, targets)
        losses.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        logger.update(loss=losses, **loss_dict)
        logger.update(lr=optimizer.param_groups[0]["lr"])

def train_transfer(data_box: DataLoader, data_mask, optimizer, model,
                   device, epoch, print_freq) -> None:
    """Implements the simple stage-wise training of XXXX"""
    import pdb; pdb.set_trace()
    model._heads.train_mask = False
    train(data_box, optimizer, model, device, epoch, print_freq)
    model._heads.train_mask = True
    train(data_mask, optimizer, model, device, epoch, print_freq)
