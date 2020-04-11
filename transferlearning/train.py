r"""
All methods to train the model
"""
import sys
import math
from typing import Dict
import numpy as np
import transferlearning.coco_eval_utils as coco_eval_utils


def check_losses(losses: float, loss_dict: Dict, target):
    """Checks if the losses are in the valid range"""
    if not math.isfinite(losses):
        import pdb; pdb.set_trace()
        print("\nA loss is not finite.")
        for key in loss_dict:
            rounded = np.round(loss_dict[key].item(), 2)
            print(key + ': ' + str(rounded))
        print("The error occured with sample id: %s"\
              %(str(target['image_id'])))
        sys.exit(1)


def engine(data_loader, optimizer, model, device, epoch, print_freq):
    """Trains the model"""
    model.train()
    model.to(device)
    metric_logger = coco_eval_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', coco_eval_utils.SmoothedValue(
            window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = coco_eval_utils.warmup_lr_scheduler(optimizer,
                                                           warmup_iters,
                                                           warmup_factor)
    # import pdb; pdb.set_trace()
    for img, target in metric_logger.log_every(
            data_loader, print_freq, header):
        target['boxes'] = target['boxes'].squeeze(0)
        target['labels'] = target['labels'].squeeze(0)
        target['masks'] = target['masks'].squeeze(0)
        img = img.to(device)
        for key in target:
            target[key] = target[key].to(device)
        # import pdb; pdb.set_trace()
        loss_dict = model(img, target)
        losses = sum(loss for loss in loss_dict.values())
        check_losses(losses, loss_dict, target)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses, **loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    return metric_logger
