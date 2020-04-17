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

__C.Vaihingen.mean = [0.485, 0.456, 0.406]
__C.Vaihingen.std = [0.229, 0.224, 0.225]
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
