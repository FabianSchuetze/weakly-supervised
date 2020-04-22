r"""
The config for different datasets
"""
from easydict import EasyDict

__C = EasyDict()

conf = __C

__C.PennFudanDataset = EasyDict()

__C.PennFudanDataset.mean = [0.485, 0.456, 0.406]
__C.PennFudanDataset.std = [0.229, 0.224, 0.225]
__C.PennFudanDataset.num_classes = 2
__C.PennFudanDataset.min_size = 600
__C.PennFudanDataset.max_size = 1200
__C.PennFudanDataset.batch_size = 1

__C.Vaihingen = EasyDict()

__C.Vaihingen.mean = [0.472, 0.317, 0.316]
__C.Vaihingen.std = [0.192, 0.128, 0.125]
__C.Vaihingen.num_classes = 5
__C.Vaihingen.min_size = 200
__C.Vaihingen.max_size = 200
__C.Vaihingen.batch_size = 2

__C.Coco = EasyDict()

__C.Coco.mean = [0.485, 0.456, 0.406]
__C.Coco.std = [0.229, 0.224, 0.225]
__C.Coco.num_classes = 91
__C.Coco.min_size = 800
__C.Coco.max_size = 1333
__C.Coco.batch_size = 1


__C.Pascal = EasyDict()

__C.Pascal.mean = [0.445, 0.456, 0.3862]
__C.Pascal.std = [0.2405, 0.2315, 0.2303]
__C.Pascal.num_classes = 20
__C.Pascal.min_size = 250
__C.Pascal.max_size = 500
__C.Pascal.batch_size = 2
