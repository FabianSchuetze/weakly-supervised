from easydict import EasyDict
import numpy as np

__C = EasyDict()

conf = __C

__C.learning_rate = 0.005

__C.momentum = 0.9

__C.weight_decay = 0.0005

__C.display_iter = 10

__C.double_bias = True

__C.gamma = 0.1

__C.decay_step_size = 3

__C.num_workers = 4
