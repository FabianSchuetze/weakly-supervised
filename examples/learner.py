#!/usr/bin/python
r"""
Showing some of the functionality of the coco dataset
"""
import torch
from transferlearning.data import VaihingenDataBase
from transferlearning.data import PennFudanDataset
from transferlearning import Supervised, Processing
import transferlearning
# from transferlearning.engine import train, evaluate
# from transferlearning.transforms import ToTensor, RandomHorizontalFlip,\
        # Compose
# from transferlearning.processing import Processing

def collate_fn(batch):
    return tuple(zip(*batch))

def train_test(database, path):
    """Returns the train and test set"""
    dataset = database(path, get_transform(training=True))
    dataset_test = database(path, get_transform(training=False))
    indices = torch.randperm(len(dataset)).tolist()
    # dataset = torch.utils.data.Subset(dataset, indices[:500])
    # dataset_test = torch.utils.data.Subset(dataset_test, indices[500:550])
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=4,
        collate_fn=collate_fn)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=collate_fn)
    return data_loader, data_loader_test, indices[500:550]

def get_transform(training):
    """The transform pipeline"""
    transforms = []
    transforms.append(transferlearning.ToTensor()) # convert PIL image to tensor
    if training:
        transforms.append(transferlearning.RandomHorizontalFlip(0.5))
    return transferlearning.Compose(transforms)

if __name__ == "__main__":
    N_GROUPS = 2
    DEVICE = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    # DEVICE = torch.device('cpu')
    # DATA, DATA_TEST, indices = train_test(VaihingenDataBase, 'data')
    DATA, DATA_TEST, indices = train_test(PennFudanDataset, 'data/PennFudanPed')
    PROCESSING = Processing(200, 200, [0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    MODEL = Supervised(N_GROUPS, PROCESSING)
    MODEL.to(DEVICE)
    PARAMS = [p for p in MODEL.parameters() if p.requires_grad]
    OPT = torch.optim.SGD(PARAMS, lr=0.005, momentum=0.9, weight_decay=0.0005)
    LR_SCHEDULER = torch.optim.lr_scheduler.StepLR(OPT, step_size=3, gamma=0.1)
    # import pdb; pdb.set_trace()
    for epoch in range(5):
        transferlearning.train(DATA, OPT, MODEL, DEVICE, epoch, 10)
        LR_SCHEDULER.step()
        transferlearning.evaluate(MODEL, DATA_TEST, DEVICE, indices)
