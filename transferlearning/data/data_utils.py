r"""
Some utils used in the data loading functions
"""
from typing import List, Dict
import torch
import numpy as np


def to_dict(masks: List, bboxes: List, labels: List, img_info: List,
            idx: int, area: List) -> Dict[str, torch.Tensor]:
    """
    Converts the inputs to pytorch.tensor type and puts them into a dict
    """
    target = {}
    target['boxes'] = torch.as_tensor(bboxes, dtype=torch.float32)
    target['masks'] = torch.as_tensor(masks, dtype=torch.uint8)
    target['img_info'] = torch.as_tensor(img_info, dtype=torch.float32)
    target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
    target['image_id'] = torch.tensor([idx])
    target['area'] = torch.as_tensor(area, dtype=torch.float32)
    return target

def extract_boxes(masks: List[np.array]) -> np.array:
    """Finds the bounding boxes of each masks

    Returns
    -------
    bboxes: np.array
        a numpy array with (xmin, ymin, width, height)i
    """
    bboxes = np.zeros((len(masks), 4))
    # import pdb; pdb.set_trace()
    for idx, img in enumerate(masks):
        pos = np.where(img)
        xmin = pos[1].min()
        ymin = pos[0].min()
        xmax = pos[1].max()
        ymax = pos[0].max()
        bboxes[idx, :] = [xmin, ymin, xmax, ymax]
    return bboxes

def check_area(img: np.array) -> bool:
    """Checks if img has a positive area"""
    pos = np.where(img)
    xmin = pos[1].min()
    ymin = pos[0].min()
    xmax = pos[1].max()
    ymax = pos[0].max()
    return xmin != xmax and ymin != ymax


def compute_stats(dataset, n_samples):
    means = torch.zeros((n_samples, 3))
    for i in range(n_samples):
        idx = np.random.randint(0, len(dataset))
        img = dataset[idx][0]
        means[i, :] = img.mean(dim=(1, 2))
    return means

