from easydict import EasyDict

__C = EasyDict()

conf = __C

__C.PennFudanDataset = EasyDict()

__C.PennFudanDataset.mean = [0.485, 0.456, 0.406]
__C.PennFudanDataset.std = [0.229, 0.224, 0.225]
__C.PennFudanDataset.num_classes = 2

__C.Vaihingen = EasyDict()

__C.Vaihingen.mean = [0.485, 0.456, 0.406]
__C.Vaihingen.std = [0.229, 0.224, 0.225]
__C.Vaihingen.num_classes = 5

__C.Coco = EasyDict()

__C.Coco.mean = [0.485, 0.456, 0.406]
__C.Coco.std = [0.229, 0.224, 0.225]
__C.Coco.num_classes = 91
