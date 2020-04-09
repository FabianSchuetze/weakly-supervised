r"""
Uses the generate crops of size 200x200 and labels for the images
"""

import os
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image
from scipy import ndimage
import torch
from transferlearning.visualization import plot_target


class VaihingenDataBase:
    """Gerneates the image database"""

    def __init__(self, path):
        """
        XXX
        """
        self._path = path
        self._index = self._generate_indices()


    def _generate_indices(self):
        """
        Generates the list with the crops
        """
        index = []
        img_path = self._path + '/images/original_images/'
        for file in os.listdir(img_path):
            img = Image.open(img_path + file)
            n_rows, n_cols = img.height // 200, img.width // 200
            rows = np.arange(n_rows) * 200
            cols = np.arange(n_cols) * 200
            rows, cols = np.meshgrid(rows, cols)
            tmp_index = [(file, i) for i in zip(rows.ravel(), cols.ravel())]
            index.extend(tmp_index)
        return index


    def _generate_crop(self, idx):
        """
        Generates the crops for picture name
        """
        info = self._index[idx]
        left, top = info[1]
        img = Image.open(self._path + '/images/original_images/' + info[0])
        target = Image.open(self._path + '/annotations/' + info[0])
        region = (top, left, top+200, left+200)
        new_img = img.crop(region)
        new_target = target.crop(region)
        return new_img, new_target

    def _find_next_object(self, blob):
        min_x, min_y = np.inf, np.inf
        for key in blob:
            tmp_min_x, tmp_min_y = blob[key][1][0], blob[key][2][0]
            if (tmp_min_x <= min_x and tmp_min_y < min_y):
                next_key = key
                min_x = tmp_min_x
                min_y = tmp_min_y
        return next_key

    def _identify_targets(self, blob: Dict[int, Tuple]):
        masks, labels = [], []
        while blob.keys():
            next_object = self._find_next_object(blob)
            regions = blob[next_object][0]
            mask = (regions == 1).astype(np.uint8)
            masks.append(mask)
            labels.append(next_object)
            regions -= 1
            new_regions, n_regions = ndimage.label(regions > 0)
            if n_regions:
                next_where = np.where(new_regions)
                blob[next_object] = (new_regions, next_where[0], next_where[1])
            else:
                blob.pop(next_object)
        return masks, labels

    def _extract_boxes(self, masks: List[np.array]) -> np.array:
        """Finds the bounding boxes of each masks

        Returns
        -------
        bboxes: np.array
            a numpy array with (xmin, ymin, width, height)i
        """
        bboxes = np.zeros((len(masks), 4))
        for idx, img in enumerate(masks):
            pos = np.where(img)
            xmin = pos[1].min()
            ymin = pos[0].min()
            xmax = pos[1].max()
            ymax = pos[0].max()
            bboxes[idx, :] = [xmin, ymin, xmax, ymax]
        return bboxes

    def _generate_targets(self, pixel_annotations: Image):
        """
        Generates the taret
        """
        fun = lambda x: x[0] + 2*x[1] + 3*x[2]
        summaries = np.apply_along_axis(fun, 2, np.array(pixel_annotations))
        blob = {}
        for obj in np.unique(summaries):
            regions = ndimage.label(summaries == obj)[0]
            where = np.where(regions)
            blob[obj] = (regions, where[0], where[1])
        masks, labels = self._identify_targets(blob)
        return masks, labels

    def _to_tensor_dict(self, masks, bboxes, labels, img_info, idx) -> Dict:
        """
        Converts the inputs to pytorch.tensor type and puts them into a dict
        """
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        img_info = torch.as_tensor(img_info, dtype=torch.float32)
        img_id = torch.tensor([idx])
        target = {}
        target['boxes'] = bboxes
        target['masks'] = masks
        target['img_info'] = img_info
        target['labels'] = labels
        target['image_id'] = img_id
        return target

    def __getitem__(self, idx: int) ->Tuple:
        """
        Returns a img and target pair from the database.

        Parameters
        ----------
        idx: int
            Which pair to take from the database

        Returns
        -------
        img: Image
            The training image

        target: Dict[string, Torch]
            All required training targets
        """
        img, target = self._generate_crop(idx)
        masks, labels = self._generate_targets(target)
        if not labels:
            return self.__getitem__(np.random.randint(0, len(self._index)))
        bboxes = self._extract_boxes(masks)
        img_info = np.array([img.height, img.width])
        target = self._to_tensor_dict(masks, bboxes, labels, img_info, idx)
        return img, target

    def __len__(self):
        return len(self._index)


if __name__ == "__main__":
    DB = VaihingenDataBase('data')
    IMG, TARGET = DB[10]
