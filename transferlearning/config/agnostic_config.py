from easydict import EasyDict
import numpy as np

__C = EasyDict()

conf = __C

__C.stride = 16

__C.learning_rate = 0.001

__C.momentum = 0.9

__C.weight_decay = 0.0005

__C.display_iter = 10

__C.double_bias = True

__C.decay_lr = 0.1

__C.decay_step_lr = 10
