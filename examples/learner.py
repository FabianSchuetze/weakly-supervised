r"""
Showing some of the functionality of the coco dataset
"""
import torch
# from transferlearning.data.generate_data import VaihingenDataBase
from transferlearning.data.penndata import PennFudanDataset
from transferlearning.supervised import Supervised
from transferlearning.train import engine
from transferlearning.evaluate import evaluate
from transferlearning.transforms import ToTensor, RandomHorizontalFlip,\
        Compose

# @torch.no_grad()
# def evaluate(base, proposal, model, data_loader, device):
# cpu_device = torch.device("cpu")
# model.eval()
# base.eval()
# proposal.eval()
# base.to(device)
# proposal.to(device)
# model.to(device)
# coco = get_coco_api_from_dataset(data_loader.dataset)
# iou_types = ['bbox', 'segm']
# coco_evaluator = CocoEvaluator(coco, iou_types)
# dataset = iter(data_loader)
# import pdb; pdb.set_trace()
# for i in range(len(dataset)):
# img, target, img_shapes = get_data(dataset, device)
# targets = [target]
# img_list = ImageList(img, img_shapes)
# base_features = base(img)
# boxes = proposal(img_list, base_features)[0]
# outputs = model(base_features, boxes, img_shapes)[0]
# outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
# res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
# coco_evaluator.update(res)

# coco_evaluator.synchronize_between_processes()
# coco_evaluator.accumulate()
# coco_evaluator.summarize()

def train_test_set():
    """Returns the train and test set"""
    dataset = PennFudanDataset('data/PennFudanPed', get_transform(train=True))
    dataset_test = PennFudanDataset('data/PennFudanPed', get_transform(train=False))
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
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
    N_GROUPS = 2
    DEVICE = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    # DEVICE = torch.device('cpu')
    DATA_LOADER, DATA_LOADER_TEST = train_test_set()
    MODEL = Supervised(2)
    MODEL.to(DEVICE)
    PARAMS = [p for p in MODEL.parameters() if p.requires_grad]
    OPTIMIZER = torch.optim.SGD(PARAMS, lr=0.005, momentum=0.9,
                                weight_decay=0.0005)
    LR_SCHEDULER = torch.optim.lr_scheduler.StepLR(OPTIMIZER,
                                                   step_size=3,
                                                   gamma=0.1)
    for epoch in range(2):
        engine(DATA_LOADER, OPTIMIZER, MODEL, DEVICE, epoch, 10)
        LR_SCHEDULER.step()
        evaluate(MODEL, DATA_LOADER_TEST, DEVICE)
