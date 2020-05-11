r"""
The evaluation functions by inheriting from chainer cv
"""
from typing import List, Dict
from chainercv.evaluations import eval_detection_coco, \
    eval_instance_segmentation_coco
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def _convert_box(pred_box: torch.Tensor):
    """Converts box to a form that can be used by cocoeval"""
    box = np.array(pred_box)
    return box[:, [1, 0, 3, 2]]


def _convert_mask(mask: torch.Tensor, threshold=0.5):
    """Converts mask to a form that can be used by cocoeval"""
    mask = np.array(mask.squeeze(1)) if mask.dim() == 4 else np.array(mask)
    return mask > threshold


def eval_boxes(predictions: List[Dict], gts: List[Dict]) -> Dict:
    """Returns the coco evaluation metric for box detection.

    Parameters
    ----------
    predictions: List[Dict]
        The predictions. Length of the list indicates the number of samples.
        Each element in the list are the predictions. Keys must be 'boxes',
        'scores', and 'labels'.

    gts: List[Dict]
        The gts. Length of the list indicates the number of samples.
        Each element in the list are the predictions. Keys must be 'boxes',
        'scores', and 'labels'.

    Returns
    -------
    eval: Dict:
        The results according to the coco metric. At IoU=0.5: VOC metric.
    """
    pred_boxes, pred_labels, pred_scores = [], [], []
    gt_boxes, gt_labels, gt_areas = [], [], []
    for prediction, gt in zip(predictions, gts):
        pred_boxes.append(_convert_box(prediction['boxes']))
        pred_labels.append(np.array(prediction['labels'], dtype=np.int32))
        pred_scores.append(np.array(prediction['scores']))
        gt_boxes.append(_convert_box(gt['boxes']))
        gt_labels.append(np.array(gt['labels'], dtype=np.int32))
        gt_areas.append(np.array(gt['area'], dtype=np.float32))
    res = eval_detection_coco(pred_boxes, pred_labels, pred_scores,
                              gt_boxes, gt_labels, gt_areas)
    return res


def eval_masks(predictions: List[Dict], gts: List[Dict]) -> Dict:
    """Returns the coco evaluation metric for instance segmentation.

    Parameters
    ----------
    predictions: List[Dict]
        The predictions. Length of the list indicates the number of samples.
        Each element in the list are the predictions. Keys must be 'boxes',
        'scores', and 'labels'.

    gts: List[Dict]
        The gts. Length of the list indicates the number of samples.
        Each element in the list are the predictions. Keys must be 'boxes',
        'scores', and 'labels'.

    Returns
    -------
    eval: Dict:
        The results according to the coco metric
    """
    # import pdb; pdb.set_trace()
    pred_masks, pred_labels, pred_scores = [], [], []
    gt_masks, gt_labels, gt_area = [], [], []
    for prediction, gt in zip(predictions, gts):
        pred_masks.append(_convert_mask(prediction['masks']))
        pred_labels.append(np.array(prediction['labels'], dtype=np.int32))
        pred_scores.append(np.array(prediction['scores']))
        gt_masks.append(_convert_mask(gt['masks']))
        gt_labels.append(np.array(gt['labels'], dtype=np.int32))
        gt_area.append(np.array(gt['area'], dtype=np.float32))
    res = eval_instance_segmentation_coco(pred_masks, pred_labels, pred_scores,
                                          gt_masks, gt_labels, gt_area)
    return res


def eval_metrics(predictions: List[Dict], gts: List[Dict], metrics: List[str],
                 ) -> Dict:
    """
    Returns the metrics specifies in metrics. The metrics can be 'semg' for
    instance segmentation and 'box' for bounding box evaluation

    Parameters
    ----------
    predictions: List[Dict]
        The predictions. Length of the list indicates the number of samples.
        Each element in the list are the predictions. Keys must be 'boxes',
        'scores', and 'labels'.

    gts: List[Dict]
        The gts. Length of the list indicates the number of samples.
        Each element in the list are the predictions. Keys must be 'boxes',
        'scores', and 'labels'.

    Returns
    -------
    eval: Dict:
        The results according to the coco metric. At IOU=0.5: VOC metrics

    """
    # import pdb; pdb.set_trace()
    results = {}
    if 'segm' in metrics:
        tmp = eval_masks(predictions, gts)
        results['segm'] = tmp
    if 'box' in metrics:
        tmp = eval_boxes(predictions, gts)
        results['box'] = tmp
    return results


def print_evaluation(metrics) -> None:
    """
    Prints the evaluation received from above
    """
    for key in metrics:
        print("The metrics for %s are:" % (key))
        out = metrics[key]['coco_eval'].__str__()
        print(out)
    print("")


def _add_patch(rec, axis, color):
    width, height = rec[2] - rec[0], rec[3] - rec[1]
    patch = patches.Rectangle((rec[0], rec[1]), width, height, linewidth=1,
                              edgecolor=color, facecolor='none')
    axis.add_patch(patch)


def _add_score(rec, axis, score):
    axis.annotate(int(score*100), (rec[0], rec[1]), fontsize=12)


# TODO: Find good way to add label names
def _visualize_box(imgs: List[torch.Tensor], boxes: List[torch.Tensor],
                   gt_boxes: List[torch.Tensor], ids: List[int],
                   output_path: str, scores: List[torch.Tensor]) -> None:
    """
    Returns the list of picutres as the result
    """
    for img, box, gt_box, img_id, score in zip(imgs, boxes, gt_boxes, ids,
                                               scores):
        fig, axis = plt.subplots()
        axis.imshow(np.array(img).transpose(1, 2, 0))
        for rec, prob in zip(np.array(box), np.array(score)):
            _add_patch(rec, axis, color='r')
            _add_score(rec, axis, prob)
        for rec in np.array(gt_box):
            _add_patch(rec, axis, color='g')
        fig.savefig(output_path + '/' +  str(img_id))


def visualize_predictions(predictions: List[Dict[str, torch.Tensor]],
                          database, gts: List[Dict[str, torch.Tensor]],
                          save_path: str, samples: int = 10) -> None:
    """
    Saves bounding box plots for `samples` images in `predictions` and saves it
    to harddisk at `config.output_dir`.

    Parameters
    ----------
    predictions: List[Dir[str, torch.Tensor]]
        The outputs of the model

    database:
        The database used to generate the samples

    gts: List[Dir[str, torch.Tensor]]
        The gts database

    save_path: str
        The location where to store the pictures

    samples: int
        How many sampes to take from the output
    """
    boxes, scores, images, image_ids, gt_boxes = [], [], [], [], []
    for idx in range(samples):
        img_id = gts[idx]['image_id'].item()
        img, target = database[img_id]
        if gts[idx]['is_flipped'].item():
            print("Id %i is flipped" %(img_id))
            img = img.flip(-1)
        assert img_id == target['image_id'].item(), "different images"
        boxes.append(predictions[idx]['boxes'])
        gt_boxes.append(gts[idx]['boxes'])
        scores.append(predictions[idx]['scores'])
        images.append(img)
        image_ids.append(img_id)
    return _visualize_box(images, boxes, gt_boxes, image_ids, save_path,
                          scores)
