r"""
All methods to train the model
"""
import sys
import math
from typing import Dict
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import transferlearning.coco_eval_utils as coco_eval_utils

from transferlearning.coco_eval import CocoEvaluator
from transferlearning.coco_utils import get_coco_api_from_dataset


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
    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = ['bbox', 'segm']
    coco_evaluator = CocoEvaluator(coco, iou_types)
    dataset = iter(data_loader)
    all_targets, all_preds, all_images = [], [], []
    # import pdb; pdb.set_trace()
    for _ in range(len(dataset)):
        images, targets = next(dataset)
        all_images.append(images[0])
        all_targets.append(targets[0])
        images = list(i.to(device) for i in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        all_preds.append(outputs[0])
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        coco_evaluator.update(res)
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    return all_preds, all_targets, all_images


def check_losses(losses: float, loss_dict: Dict, target):
    """Checks if the losses are in the valid range"""
    if not math.isfinite(losses):
        print("\nA loss is not finite.")
        for key in loss_dict:
            rounded = np.round(loss_dict[key].item(), 2)
            print(key + ': ' + str(rounded))
        print("The error occured with sample id: %s"\
              %(str(target['image_id'])))
        sys.exit(1)


def learning_rate_scheduler(optimizer, epoch, dataset):
    """The lr schedule"""
    if epoch != 0:
        return None
    warmup_factor = 1. / 1000
    warmup_iters = min(1000, len(dataset) - 1)
    lr_scheduler = coco_eval_utils.warmup_lr_scheduler(optimizer,
                                                       warmup_iters,
                                                       warmup_factor)
    return lr_scheduler


def get_logging():
    """Logger and iterator"""
    metric_logger = coco_eval_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', coco_eval_utils.SmoothedValue(
            window_size=1, fmt='{value:.6f}'))
    return metric_logger


def train(data: DataLoader, optimizer: torch.optim, model,
          device: torch.device, epoch: int, print_freq: int) -> None:
    """Trains the model"""
    model.train()
    model.to(device)
    logger = get_logging()
    header = 'Epoch: [{}]'.format(epoch)
    lr_scheduler = learning_rate_scheduler(optimizer, epoch, data)
    for images, targets in logger.log_every(data, print_freq, header):
        optimizer.zero_grad()
        images = list(i.to(device) for i in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # import pdb; pdb.set_trace()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        check_losses(losses, loss_dict, targets)
        losses.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        logger.update(loss=losses, **loss_dict)
        logger.update(lr=optimizer.param_groups[0]["lr"])
