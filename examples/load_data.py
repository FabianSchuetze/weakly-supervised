from transferlearning.data import VaihingenDataBase
from transferlearning.data import PennFudanDataset
from transferlearning.data import CocoDB
import transferlearning
import numpy as np

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
    # dataset = CocoDB('data/coco', 'train2014', get_transform(training=False))
    # all_labels = []
    # for i in range(5000):
        # idx = np.random.randint(len(dataset))
        # img, target = dataset[idx]
        # all_labels.extend(target['labels'].tolist())
        # if (i % 200) == 0:
            # # import pdb; pdb.set_trace()
            # print("at iter %i " %(i))
            # unq = np.unique(all_labels)
            # all_labels = unq.tolist()
            # print(all_labels)
            # print('\n\n')

