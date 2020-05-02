r"""
The evaluation functions by inheriting from chainer cv
"""
from typing import List, Dict
from chainercv.evaluations import eval_detection_coco, \
    eval_instance_segmentation_coco, eval_detection_voc,\
    eval_instance_segmentation_voc
# from chainercv.visualizations import vis_bbox
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# import matplotlib
# matplotlib.use('GTK3Agg')


def _convert_box(pred_box: torch.Tensor):
    """Converts box to a form that can be used by cocoeval"""
    box = np.array(pred_box)
    return box[:, [1, 0, 3, 2]]


def _convert_mask(mask: torch.Tensor, threshold=0.5):
    """Converts mask to a form that can be used by cocoeval"""
    mask = np.array(mask.squeeze(1)) if mask.dim() == 4 else np.array(mask)
    return mask > threshold


def eval_boxes(predictions: List[Dict], gts: List[Dict], competition: str) -> Dict:
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
        The results according to the coco metric
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
    if competition == 'Pascal':
        res = eval_detection_coco(pred_boxes, pred_labels, pred_scores,
                                  gt_boxes, gt_labels, gt_areas)
    elif competition == 'VOC':
        res = eval_detection_voc(pred_boxes, pred_labels, pred_scores,
                                 gt_boxes, gt_labels)
    return res


def eval_masks(predictions: List[Dict], gts: List[Dict], competition: str) -> Dict:
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
    if competition == 'Pascal':
        res = eval_instance_segmentation_coco(pred_masks, pred_labels, pred_scores,
                                              gt_masks, gt_labels, gt_area)
    elif competition == 'VOC':
        res = eval_instance_segmentation_voc(pred_masks, pred_labels,
                                             pred_scores, gt_masks,
                                             gt_labels)
    return res


def eval_metrics(predictions: List[Dict], gts: List[Dict], metrics: List[str],
                 competition: str = 'Pascal')\
        -> Dict:
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
        The results according to the coco metric

    """
    # import pdb; pdb.set_trace()
    results = {}
    if 'segm' in metrics:
        tmp = eval_masks(predictions, gts, competition)
        results['segm'] = tmp
    if 'box' in metrics:
        tmp = eval_boxes(predictions, gts, competition)
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


# TODO: Finda good way to add label names
def _visualize_box(images: List[torch.Tensor], boxes: List[torch.Tensor],
                   image_ids: List[int], output_path: str,
                   scores: List[torch.Tensor]) -> None:
    """
    Returns the list of picutres as the result
    """
    for image, box, img_id, score in zip(images, boxes, image_ids, scores):
        fig, axis = plt.subplots()
        image = np.array(image).transpose(1, 2, 0)
        box = np.array(box)
        # score = np.array(score) // Score not used at the moment
        axis.imshow(image)
        for rec in box:
            width, height = rec[2] - rec[0], rec[3] - rec[1]
            patch = patches.Rectangle((rec[0], rec[1]), width, height,
                                      linewidth=1, edgecolor='r',
                                      facecolor='none')
            axis.add_patch(patch)
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
    import pdb; pdb.set_trace()
    boxes, scores, images, image_ids = [], [], [], []
    indices = np.random.choice(np.arange(len(gts)), samples, replace=False)
    for idx in indices:
        img_id = gts[idx]['image_id'].item()
        img, target = database[img_id]
        assert img_id == target['image_id'].item(), "different images"
        boxes.append(predictions[idx]['boxes'])
        scores.append(predictions[idx]['scores'])
        images.append(img)
        image_ids.append(img_id)
    return _visualize_box(images, boxes, image_ids, save_path,
                          scores)
