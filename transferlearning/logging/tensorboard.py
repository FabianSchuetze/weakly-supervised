r"""
Several tensorboard convenience functions
"""
from typing import Dict, List
from torch.utils.tensorboard.writer import SummaryWriter


def log_losses(writer: SummaryWriter, losses: Dict, print_freq: int,
               iteration: int):
    """Writes elements in losses to writer at certain iterations"""
    if (iteration % print_freq) == 0:
        for key in losses:
            writer.add_scalar('Loss/train/' + key, losses[key], iteration)


def log_accuracies(writer: SummaryWriter, metrics, cur_epoch: int):
    """
    Prints some val accurcies to tensorboard
    """
    summaries = ['ap/iou=0.50:0.95/area=all/max_dets=100',
                 'ar/iou=0.50:0.95/area=all/max_dets=100']
    # import pdb; pdb.set_trace()
    for key in metrics:
        for summary in summaries:
            tag = 'accuracy/' + key + '/' + summary
            numbers = metrics[key][summary]
            writer.add_scalar(tag, numbers.mean(), cur_epoch)


def log_architecture(writer: SummaryWriter, model, datasets: List, optimizer, data_name):
    """
    Writes a summary of the experiment
    """
    writer.add_text('architecture/', str(model))
    writer.add_text('dataset', data_name)
    writer.add_text('number datasets/:', str(len(datasets)))
    writer.add_text('optimizer/', str(optimizer))
