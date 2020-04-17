r"""
The Coco dataset, ineriting from the torchvision class
"""
from typing import Optional, Dict, Tuple, List
from torchvision.datasets import CocoDetection
from pycocotools import mask
from torch import Tensor
from PIL import Image
import numpy as np
from transferlearning.transforms import Compose
from .data_utils import to_dict


class CocoDB:
    """The CoCo DB"""

    def __init__(self, root: str, name: str,
                 transforms: Optional[Compose] = None):
        data_path = root + '/' + name
        anns = root + '/annotations/instances_' + name + '.json'
        self._orig_coco = CocoDetection(data_path, anns)
        self._transforms = transforms

    def _convert_to_box(self, box: List[float]):
        width, height = box[2], box[3]
        return [box[0], box[1], box[0] + width, box[1] + height]

    def _convert_targets(self, img, originals):
        masks, labels, areas, boxes = [], [], [], []
        height, width = img.height, img.width
        for original in originals:
            if not original['iscrowd']:
                rle = mask.frPyObjects(original['segmentation'], height, width)
                masks.append(mask.decode(rle)[:, :, 0])
                labels.append(original['category_id'])
                areas.append(original['area'])
                boxes.append(self._convert_to_box(original['bbox']))
            else:
                return [], [], [], []
        return boxes, masks, labels, areas

    def _inadmissible_example(self, masks: List[float], labels: List[float]):
        """If samples are occupied fully by one class the sample is deemed
        inadmissible
        """
        return not labels

    def __getitem__(self, idx: int) ->Tuple[Image.Image, Dict[str, Tensor]]:
        img, orig_targets = self._orig_coco[idx]
        boxes, masks, labels, areas = self._convert_targets(img, orig_targets)
        if self._inadmissible_example(masks, labels):
            return self.__getitem__(np.random.randint(0, self.__len__()))
        img_info = [img.height, img.width]
        target = to_dict(masks, boxes, labels, img_info, idx, areas)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

    def __len__(self):
        return len(self._orig_coco)

    def n_classes(self) -> int:
        """The number of classes"""
        return 91
