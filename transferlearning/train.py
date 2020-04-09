r"""
All methods to train the model
"""
import sys
import math
import transferlearning.coco_eval_utils as coco_eval_utils


def engine(data_loader, optimizer, model, device, epoch, print_freq):
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
    for img, target in metric_logger.log_every(
            data_loader, print_freq, header):
        target['boxes'] = target['boxes'].squeeze(0)
        target['labels'] = target['labels'].squeeze(0)
        img = img.to(device)
        for key in target:
            target[key] = target[key].to(device)
        loss_dict = model(img, [target])
        losses = sum(loss for loss in loss_dict.values())
        if not math.isfinite(losses):
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses, **loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    return metric_logger
