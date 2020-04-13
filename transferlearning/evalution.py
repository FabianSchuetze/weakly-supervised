r"""
Generates the evaluation functions by inheriting from chainer cv"""

from typing import List, Dict
from chainercv.evaluations import eval_detection_voc
import torch
import numpy as np

def _convert_pred_box(pred_box: torch.Tensor):
    pred_box = np.array(pred_box)
    return pred_box[:, [1, 0, 3, 2]]


def eval_detection(predictions: List[Dict], gts: List[Dict]):
# , pred_labels, pred_scores,
                   # gt_bboxes, gt_labels):
    """Does some stuff"""
    # import pdb; pdb.set_trace()
    cv_pred_boxes = []
    cv_pred_labels = []
    cv_pred_scores = []
    cv_gt_boxes = []
    cv_gt_labels = []
    for prediction, gt in zip(predictions, gts):
        cv_pred_boxes.append(_convert_pred_box(prediction['boxes']))
        cv_pred_labels.append(np.array(prediction['labels'], dtype=np.int32))
        cv_pred_scores.append(np.array(prediction['scores']))
        cv_gt_boxes.append(_convert_pred_box(gt['boxes']))
        cv_gt_labels.append(np.array(gt['labels'], dtype=np.int32))
    # import pdb; pdb.set_trace()
    res = eval_detection_voc(cv_pred_boxes, cv_pred_labels, cv_pred_scores,
                             cv_gt_boxes, cv_gt_labels)
    return res
