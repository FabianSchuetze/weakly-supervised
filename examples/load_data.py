r"""
Examples how to load the different databases
"""
import numpy as np
from transferlearning.data import VaihingenDataBase, PennFudanDataset,\
        CocoDB, PascalVOCDB
import transferlearning
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def get_transform(training):
    """The transform pipeline"""
    transforms = []
    transforms.append(transferlearning.ToTensor()) # convert PIL image to tensor
    if training:
        transforms.append(transferlearning.RandomHorizontalFlip(0.5))
    return transferlearning.Compose(transforms)

def visualize_boxes(image, boxes, labels):
    """
    Plots the boxes
    """
    import pdb; pdb.set_trace()
    fig, ax = plt.subplots()
    image = np.array(image).transpose(1, 2, 0)
    ax.imshow(image)
    for label, box in zip(labels.tolist(), boxes.tolist()):
        width = box[2] - box[0]
        height = box[3] - box[1]
        rect = patches.Rectangle((box[0], box[1]), width, height,
                                 linewidth=1, edgecolor='r',
                                 facecolor='none')
        ax.annotate(label,(box[0], box[1]))
        ax.add_patch(rect)
    import pdb; pdb.set_trace()
    return fig

if __name__ == "__main__":
    # dataset = VaihingenDataBase('data/vaihingen', train=True,
                                # transforms=get_transform(training=True))
    dataset = PascalVOCDB('data/VOC2007', '2007', train=True,
            transforms=get_transform(training=False))
    # dataset = PennFudanDataset('data/PennFudanPed',
                                # get_transform(training=False))
    # dataset = CocoDB('data/coco', 'train2014', get_transform(training=False))


