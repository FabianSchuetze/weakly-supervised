r"""
Showing some of the functionality of the coco dataset
"""
import torch
from transferlearning.data.vaihingen import VaihingenDataBase
from transferlearning.data.penndata import PennFudanDataset
from transferlearning.supervised import Supervised
from transferlearning.train import engine
from transferlearning.evaluate import evaluate
from transferlearning.transforms import ToTensor, RandomHorizontalFlip,\
        Compose
from transferlearning.processing import Processing

def train_test(database, path):
    """Returns the train and test set"""
    dataset = database(path, get_transform(train=True))
    dataset_test = database(path, get_transform(train=False))
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:500])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[500:1000])
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=4)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4)
    return data_loader, data_loader_test

def get_transform(train):
    """The transform pipeline"""
    transforms = []
    transforms.append(ToTensor()) # convert PIL image to tensor
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)

if __name__ == "__main__":
    N_GROUPS = 5
    DEVICE = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    # DEVICE = torch.device('cpu')
    DATA, DATA_TEST = train_test(VaihingenDataBase, 'data')
    # DATA, DATA_TEST = train_test(PennFudanDataset, 'data/PennFudanPed')
    PROCESSING = Processing(200, 200, [0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    MODEL = Supervised(N_GROUPS, PROCESSING)
    MODEL.to(DEVICE)
    PARAMS = [p for p in MODEL.parameters() if p.requires_grad]
    OPTIMIZER = torch.optim.SGD(PARAMS, lr=0.005, momentum=0.9,
                                weight_decay=0.0005)
    LR_SCHEDULER = torch.optim.lr_scheduler.StepLR(OPTIMIZER,
                                                   step_size=3,
                                                   gamma=0.1)
    for epoch in range(2):
        engine(DATA, OPTIMIZER, MODEL, DEVICE, epoch, 10)
        LR_SCHEDULER.step()
        evaluate(MODEL, DATA_TEST, DEVICE)
