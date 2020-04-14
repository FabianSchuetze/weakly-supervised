r"""
Uses the generate crops of size 200x200 and labels for the images
"""
import os
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image
from scipy import ndimage
import torch
from transferlearning.transforms import Compose


class VaihingenDataBase:
    """
    Gerneates the image database for the Vaihingen dataset. The dataset
    contains 33 high-resulution images. From this dataset, I crop images of
    200x200 pixel sizes which constitute the samples of the training set.
    In total that leads to about 120*33 samples

    Parameters
    ----------
    path: str
        The root to the datafolder. It is assumed that the images then reside
        at path + '/images/' and the anntations at path + '/annotations'/

    transforms: Optional[transferlearning.transforms.Compose]
        Transformation operating on troch.tensors
    """

    def __init__(self, path: str, transforms: Optional[Compose] = None):
        self._path = path
        self._index = self._generate_indices()
        self._transforms = transforms

    def _generate_indices(self):
        """
        Generates the list with the crops. Each image from the original file is
        read and cut into pieces of 200x200 pixel size. The image named and the
        croped location provide the indices for the samples.
        """
        index = []
        img_path = self._path + '/images/'
        for file in os.listdir(img_path):
            img = Image.open(img_path + file)
            n_rows, n_cols = img.height // 200, img.width // 200
            rows = np.arange(n_rows) * 200
            cols = np.arange(n_cols) * 200
            rows, cols = np.meshgrid(rows, cols)
            tmp_index = [(file, i) for i in zip(rows.ravel(), cols.ravel())]
            index.extend(tmp_index)
        return index

    def _generate_crop(self, idx: int) ->Tuple[Image.Image, Image.Image]:
        """
        Returns a image and a traget annoation from the index

        Parameters
        ----------
        idx: int
            The sample index

        Returns
        -------
        new_img: PIL.Image.Image
            The 200x200 rgb image from the files

        new_target: PIL.Image.Image
            The 200x200 ground-truth annotation
        """
        # import pdb; pdb.set_trace()
        info = self._index[idx]
        left, top = info[1]
        img = Image.open(self._path + '/images/' + info[0])
        target = Image.open(self._path + '/annotations/' + info[0])
        region = (top, left, top+200, left+200)
        new_img = img.crop(region)
        new_target = target.crop(region)
        return new_img, new_target

    def _find_next_object(self, blob):
        """
        Finds the object that is located to the most top-right among all
        objects in blob
        """
        min_x, min_y = np.inf, np.inf
        for key in blob:
            tmp_min_x, tmp_min_y = blob[key][1][0], blob[key][2][0]
            if (tmp_min_x <= min_x and tmp_min_y < min_y):
                next_key = key
                min_x = tmp_min_x
                min_y = tmp_min_y
        return next_key

    def _check_area(self, img: np.array) ->bool:
        """Checks if img has a positive area"""
        pos = np.where(img)
        xmin = pos[1].min()
        ymin = pos[0].min()
        xmax = pos[1].max()
        ymax = pos[0].max()
        return xmin != xmax and ymin != ymax

    def _identify_targets(self, blob: Dict[int, Tuple]):
        """
        Generates the masks and labels encoded in `blob`. The masks and labels
        are returns such that they are ordered from top-left to bottom-right.

        Parameters
        ----------
        blob: Dict[int, Tuple[np.array, np.array, np.array]]
            For each target label, the blob stores the mask areas and the
            x,y locations for the masks.

        Returns
        -------
        masks: List[np.array]
            Indicates the region which is masked. Ordered from top right to
            bottom left

        labels: List[int]
            The labels for each masked region. Same ordering as masks
        """
        masks, labels = [], []
        while blob.keys():
            next_object = self._find_next_object(blob)
            regions = blob[next_object][0]
            mask = (regions == 1).astype(np.uint8)
            large_enough = self._check_area(mask)
            if large_enough:
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
        # import pdb; pdb.set_trace()
        for idx, img in enumerate(masks):
            pos = np.where(img)
            xmin = pos[1].min()
            ymin = pos[0].min()
            xmax = pos[1].max()
            ymax = pos[0].max()
            bboxes[idx, :] = [xmin, ymin, xmax, ymax]
        return bboxes

    def _convert_to_label(self, color: np.array) -> int:
        """
        Converts the input image to a label. The annotations are encoded in
        a color image. Each pixel is then convert into one int.
        """
        label = 0  # default label, background
        if color[0] == 0 and color[1] == 0 and color[2] == 255:
            label = 1
        elif color[0] == 0 and color[1] == 255 and color[2] == 255:
            label = 2
        elif color[0] == 0 and color[1] == 255 and color[2] == 0: # tree
            label = 3
        elif color[0] == 255 and color[1] == 255 and color[2] == 0:
            label = 4
        return label

    def _generate_targets(self, annotations: Image.Image)\
            -> Tuple[List[np.array], List[int]]:
        """
        Generates the masks and labels for each object class.

        Parameters
        ----------
        annotations: Image.Image
            The ground truth annotations

        Returns
        -------
        masks: List[np.array]
            Indicates the region which is masked. Ordered from top right to
            bottom left

        labels: List[int]
            The labels for each masked region. Same ordering as masks
        """
        summaries = np.apply_along_axis(self._convert_to_label, 2,
                                        np.array(annotations))
        blob = {}
        for obj in np.unique(summaries):
            regions = ndimage.label(summaries == obj)[0]
            where = np.where(regions)
            blob[obj] = (regions, where[0], where[1])
        masks, labels = self._identify_targets(blob)
        return masks, labels

    def _to_dict(self, masks, bboxes, labels, img_info, idx, area)\
            -> Dict[str, torch.Tensor]:
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
        # target['iscrowd'] = torch.as_tensor(iscrowd, dtype=torch.int32)
        return target

    def _inadmissible_example(self, masks, labels):
        """If samples are occupied fully by one class the sample is deemed
        inadmissible
        """
        wrong_object = len(masks) == 1 and masks[0].sum() == 200 * 200
        no_labels = not labels
        return wrong_object or no_labels

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
        if self._inadmissible_example(masks, labels):
            return self.__getitem__(np.random.randint(0, len(self._index)))
        bboxes = self._extract_boxes(masks)
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        # iscrowd = torch.zeros((bboxes.shape[0],), dtype=torch.int64)
        img_info = np.array([img.height, img.width])
        target = self._to_dict(masks, bboxes, labels, img_info, idx, area)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

    def __len__(self):
        return len(self._index)
