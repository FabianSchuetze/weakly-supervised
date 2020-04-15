r"""Contains the config for the runs"""
from easydict import EasyDict
import numpy as np

cfg = EasyDict()

cfg.learning_rate = 0.001

cfg.momentum = 0.9

cfg.weight_decay = 0.0005

cfg.display_iter = 20

cfg.double_bias = True

cfg.decay_lr = 0.1

cfg.decay_step_lr = 10

cfg.fg_iou_thresh = 0.5

cfg.bg_iou_thresh = 0.5

cfg.nms_thresh = 0.5

cfg.detections_per_img = 100
