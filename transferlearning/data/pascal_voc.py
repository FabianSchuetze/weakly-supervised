r"""
Loader for the pascal voc class.

Subclasses  form torchvision.datasets.VOCDetection
"""
from typing import Optional, Dict, List, Tuple
import numpy as np
from torchvision.datasets import VOCDetection
import torch
from transferlearning.transforms import Compose

class PascalVOCDB:
    """The Pascal VOC DB"""

    def __init__(self, root: str, year: str, train: bool,
                 image_set: str = 'trainval',
                 transforms: Optional[Compose] = None):
        self._orig_pascal = VOCDetection(root, year, image_set)
        self._train = train
        self._transforms = transforms
        self._class_to_ind = self._class_conversion()

    def _class_conversion(self) -> Dict[str, int]:
        """
        Converts the Pascal VOC classes into the ints

        Returns
        -------
        class_to_ind: Dict[str, int]
            The conversion table
        """
        classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                   'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                   'train', 'tvmonitor')
        class_to_ind = dict(zip(classes, range(len(classes))))
        return class_to_ind

    def _get_box(self, box: Dict[str, str]) -> List[int]:
        """
        Extracts the bounding boxes
        """
        xmin = int(box['xmin'])
        xmax = int(box['xmax'])
        ymin = int(box['ymin'])
        ymax = int(box['ymax'])
        return [xmin, ymin, xmax, ymax]

    def _convert_targets(self, originals: Dict[str, str])\
            -> Tuple[List[List[int]], List[int], List[int]]:
        """
        Converst the targets from the Pascal VOC format into a lists

        Parameters
        ----------
        originals: Dict[str, str]
            The original annotations in the Pascal VOC format

        Returns
        -------
        boxes: List[int]
            The bounding boxes

        labels: List[int]
            The targets
        """
        labels, boxes, areas = [], [], []
        targets = originals['annotation']['object']
        targets = [targets] if isinstance(targets, dict) else targets
        for target in targets:
            labels.append(self._class_to_ind[target['name']])
            box = self._get_box(target['bndbox'])
            boxes.append(box)
            areas.append((box[3] - box[1]) * (box[2] - box[0]))
        return boxes, labels, areas

    def _inadmissible_example(self, labels: List[int]):
        """If samples are occupied fully by one class the sample is deemed
        inadmissible
        """
        return not labels

    def _to_dict(self, boxes: List[List[int]], labels: List[int],
                 area: List[int], img_info: List[float])\
                         ->Dict[str, torch.Tensor]:
        target = {}
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['img_info'] = torch.as_tensor(img_info, dtype=torch.float32)
        target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
        target['area'] = torch.as_tensor(area, dtype=torch.float32)
        return target

    def __getitem__(self, idx):
        """returns the image with index idx"""
        img, target = self._orig_pascal[idx]
        boxes, labels, areas = self._convert_targets(target)
        if self._inadmissible_example(labels):
            return self.__getitem__(np.random.randint(0, self.__len__()))
        img_info = [img.height, img.width]
        # if self._train:
            # shuffle(boxes, labels, areas)
        target = self._to_dict(boxes, labels, areas, img_info)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

    def __len__(self):
        return len(self._orig_pascal)
