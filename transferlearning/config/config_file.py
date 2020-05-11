from easydict import EasyDict
import numpy as np

__C = EasyDict()

conf = __C

__C.learning_rate = 0.005

__C.momentum = 0.9

__C.weight_decay = 0.0005

__C.display_iter = 20

__C.gamma = 0.1

__C.decay_step_size = 10

__C.num_workers = 8

__C.max_epochs = 30

__C.val_iters = 150

__C.train_test_split = 0.9

__C.pickle = True

__C.PennFudanDataset = EasyDict()

__C.PennFudanDataset.mean = [0.485, 0.456, 0.406]
__C.PennFudanDataset.std = [0.229, 0.224, 0.225]
__C.PennFudanDataset.num_classes = 2
__C.PennFudanDataset.min_size = 600
__C.PennFudanDataset.max_size = 1200
__C.PennFudanDataset.batch_size = 1
__C.PennFudanDataset.loss_types = ['box', 'segm']
__C.PennFudanDataset.output_dir = 'outputs/pennfudan'

__C.Vaihingen = EasyDict()

__C.Vaihingen.mean = [0.472, 0.317, 0.316]
__C.Vaihingen.std = [0.192, 0.128, 0.125]
__C.Vaihingen.num_classes = 5
__C.Vaihingen.min_size = 200
__C.Vaihingen.max_size = 200
__C.Vaihingen.batch_size = 2
__C.Vaihingen.loss_types = ['box', 'segm']
__C.Vaihingen.root_folder = 'data/vaihingen'
__C.Vaihingen.transfer_split = 0.5
__C.Vaihingen.output_dir = 'outputs/vaihingen'
__C.Vaihingen.learning_rate = 0.005


__C.Coco = EasyDict()

__C.Coco.mean = [0.485, 0.456, 0.406]
__C.Coco.std = [0.229, 0.224, 0.225]
__C.Coco.num_classes = 91
__C.Coco.min_size = 800
__C.Coco.max_size = 1333
__C.Coco.batch_size = 1
__C.Coco.loss_types = ['box', 'segm']
__C.Coco.output_dir = 'outputs/coco'

__C.Faces = EasyDict()

__C.Faces.mean = [0.21, 0.35, 0.0]
__C.Faces.std = [1., 1., 1.]
__C.Faces.num_classes = 4
__C.Faces.min_size = 256
__C.Faces.max_size = 500
__C.Faces.batch_size = 2
__C.Faces.loss_types = ['box']
__C.Faces.database_path =\
    '/home/fabian/CrossCalibration/TCLObjectDetectionDatabase/tcl3_data.xml'
__C.Faces.output_dir = 'outputs/faces'

__C.Pascal = EasyDict()

__C.Pascal.mean = [0.445, 0.456, 0.3862]
## __C.Pascal.std = [0.2405, 0.2315, 0.2303]
__C.Pascal.std = [1., 1., 1.]
__C.Pascal.num_classes = 20
__C.Pascal.min_size = 600
__C.Pascal.max_size = 1000
__C.Pascal.batch_size = 2
__C.Pascal.loss_types = ['box']
__C.Pascal.root_folder = 'data/VOC2007'
__C.Pascal.year = '2007'
__C.Pascal.learning_rate = 0.001
__C.Pascal.output_dir = 'outputs/pascal'
