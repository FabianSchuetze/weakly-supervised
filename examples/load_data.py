r"""
Examples how to load the different databases
"""
import numpy as np
from transferlearning.data import VaihingenDataBase, PennFudanDataset,\
        CocoDB, PascalVOCDB
import transferlearning

def get_transform(training):
    """The transform pipeline"""
    transforms = []
    transforms.append(transferlearning.ToTensor()) # convert PIL image to tensor
    if training:
        transforms.append(transferlearning.RandomHorizontalFlip(0.5))
    return transferlearning.Compose(transforms)

if __name__ == "__main__":
    # dataset = VaihingenDataBase('data/vaihingen', get_transform(training=True))
    dataset = PascalVOCDB('data/VOC2007', '2007',
            transforms=get_transform(training=False))
    # dataset = PennFudanDataset('data/PennFudanPed',
                                # get_transform(training=False))
    # dataset = CocoDB('data/coco', 'train2014', get_transform(training=False))
