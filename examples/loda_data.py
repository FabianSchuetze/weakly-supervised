from transferlearning.data import VaihingenDataBase
from transferlearning.data import PennFudanDataset
import transferlearning

def get_transform(training):
    """The transform pipeline"""
    transforms = []
    transforms.append(transferlearning.ToTensor()) # convert PIL image to tensor
    if training:
        transforms.append(transferlearning.RandomHorizontalFlip(0.5))
    return transferlearning.Compose(transforms)

if __name__ == "__main__":
    # dataset = VaihingenDataBase('data', get_transform(training=True))
    dataset = PennFudanDataset('data/PennFudanPed',
                                get_transform(training=False))
