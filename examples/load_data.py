from transferlearning.data import VaihingenDataBase
from transferlearning.data import PennFudanDataset
from transferlearning.data import CocoDB
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
    dataset = PennFudanDataset('data/PennFudanPed',
                                get_transform(training=False))
    dataset = CocoDB('data/coco' ,'train2014')
