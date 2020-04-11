r"""Has everyhing to evaluate the model"""
import torch
from transferlearning.coco_utils import get_coco_api_from_dataset
import transferlearning.coco_eval_utils as coco_eval_utils
from transferlearning.coco_eval import CocoEvaluator

def get_data(input_iter, device):
    img, target = next(input_iter)
    target['boxes'] = target['boxes'].squeeze(0)
    target['labels'] = target['labels'].squeeze(0)
    target['masks'] = target['masks'].squeeze(0)
    img = img.to(device)
    for key in target:
        target[key] = target[key].to(device)
    img_shapes = [(img.shape[2], img.shape[3])]
    return img, target, img_shapes

@torch.no_grad()
def evaluate(model, data_loader, device, indices) ->None:
    """Evaluates hte model"""
    # n_threads = torch.get_num_threads()
    # torch.set_num_threads(1)
    # import pdb; pdb.set_trace()
    cpu_device = torch.device("cpu")
    model.eval()
    coco = get_coco_api_from_dataset(data_loader.dataset, indices)
    iou_types = ['bbox', 'segm']
    coco_evaluator = CocoEvaluator(coco, iou_types)
    # import pdb; pdb.set_trace()
    dataset = iter(data_loader)
    for _ in range(len(dataset)):
        img, target, _ = get_data(dataset, device)
        img = img.to(device)
        targets = [target]
        torch.cuda.synchronize()
        outputs = model(img)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    # torch.set_num_threads(n_threads)
