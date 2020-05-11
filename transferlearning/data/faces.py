r"""
Loader for the faces
"""
from typing import Optional, Dict, List, Tuple
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import torch
from transferlearning.transforms import Compose
from .data_utils import shuffle_targets

class FacesDB:
    """The Pascal VOC DB"""

    def __init__(self, database: str, train: bool,
                 transforms: Optional[Compose] = None):
        self._database = database
        self._images = self._load_images()
        self._train = train
        self._conversion = {'glabella': 1, 'left_eye':2, 'right_eye':3,
                            'nose_tip': 4}
        self._transforms = transforms

    def _load_images(self):
        tree = ET.parse(self._database)
        return tree.findall('images/image')

    def _inadmissible_example(self, labels: List[int]):
        """If samples are occupied fully by one class the sample is deemed
        inadmissible
        """
        return not labels

    def _to_dict(self, target: Dict, idx: int) ->Dict[str, torch.Tensor]:
        torch_target = {}
        torch_target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32)
        torch_target['img_info'] = torch.as_tensor(target['img_info'], dtype=torch.float32)
        torch_target['labels'] = torch.as_tensor(target['labels'], dtype=torch.int64)
        torch_target['area'] = torch.as_tensor(target['area'], dtype=torch.float32)
        torch_target['image_id'] = torch.tensor([idx])
        torch_target['is_flipped'] = torch.tensor([0]) ## Not flipped by default
        return torch_target

    def _convert_to_box(self, box: ET.Element) -> List[int]:
        """
        Generates the bouding boxes
        """
        xmin = int(box.get('left'))
        ymin = int(box.get('top'))
        xmax = int(box.get('left')) + int(box.get('width'))
        ymax = int(box.get('top')) + int(box.get('height'))
        return [xmin, ymin, xmax, ymax]

    def _append_label(self, box: ET.Element) -> int:
        """
        Gets the corresponding label to the box
        """
        label = box.find('label').text
        return self._conversion[label]

    def _load_sample(self, idx) -> Tuple[Image.Image, List, List, List]:
        sample = self._images[idx]
        img = Image.open(sample.get('file'))
        boxes, labels, areas = [], [], []
        for tag in sample.findall('box'):
            box = self._convert_to_box(tag)
            boxes.append(box)
            labels.append(self._append_label(tag))
            areas.append((box[3] - box[1]) * (box[2] - box[0]))
        return img, boxes, labels, areas

    def __getitem__(self, idx) -> Tuple[torch.tensor, Dict[str, torch.tensor]]:
        """returns the image with index idx"""
        img, boxes, labels, areas = self._load_sample(idx)
        if self._inadmissible_example(labels):
            print("Idx %i is not admissible " %(idx))
            return self.__getitem__(np.random.randint(0, self.__len__()))
        target = {}
        target['labels'] = labels
        target['boxes'] = boxes
        target['area'] = areas
        target['img_info'] = [img.height, img.width]
        target = shuffle_targets(target) if self._train else target
        target = self._to_dict(target, idx)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

    def __len__(self):
        return len(self._images)
