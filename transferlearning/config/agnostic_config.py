from easydict import EasyDict
import numpy as np

__C = EasyDict()

conf = __C

__C.learning_rate = 0.005

__C.momentum = 0.9

__C.weight_decay = 0.0005

__C.display_iter = 50

__C.gamma = 0.1

__C.decay_step_size = 10

__C.num_workers = 8

__C.max_epochs = 10

__C.val_iters = 50

__C.train_test_split = 0.9
