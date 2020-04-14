r"""
Contains the Neural Network model
"""
from typing import List, Dict, Optional
import torch
import torchvision.models as models
from .masking import SupervisedMask, TransferFunction, WeaklySupervised
from .my_roi_heads import RoIHeads


class TwoHeaded(torch.nn.Module):
    """With two outputs"""

    def __init__(self, input_dimension, out_dim):
        super(TwoHeaded, self).__init__()
        self._in_dim = input_dimension
        self._out_dim = out_dim
        self._cls_score, self._bbox_pred = self._init_layers()

    def _init_layers(self):
        cls = torch.nn.Linear(self._in_dim, self._out_dim)
        bbox = torch.nn.Linear(self._in_dim, self._out_dim * 4)
        return cls, bbox

    def forward(self, data):
        unscaled_probs = self._cls_score(data)
        bbox_pred = self._bbox_pred(data)
        return unscaled_probs, bbox_pred


class Supervised(torch.nn.Module):
    """Teh supervised training"""

    def __init__(self, n_dim: int, processing, weakly_supervised: bool = False):
        """
        Instance Segmenation based on mask r-cnn

        Parameters
        ----------
        n_dim: int
            The number of classes to label

        processing:
            A class used for pre- and postprocessing.

        weakly_supervised: bool
            Indicates whether to use weakly supervised training
        """
        super(Supervised, self).__init__()
        self._weakly_supervised = weakly_supervised
        self._backbone, self._rpn = self._get_backbone()
        self._heads = self._get_heads(n_dim)
        self._processing = processing

    def _get_backbone(self):
        """the backbone of the model, used to downsample the input images to a
        features representation"""
        model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        return model.backbone, model.rpn

    # TODO: Make the bbox head class agnostic, -> benchmark
    def _get_heads(self, out_dim: int):
        """Defines the box, classificaiton and mask heads of the network"""
        model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        # == custom == #
        box_pred = TwoHeaded(1024, out_dim)
        in_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels
        mask_predictor = SupervisedMask(in_channels, 256, out_dim)

        # import pdb; pdb.set_trace()
        if self._weakly_supervised:
            transfer = TransferFunction(1024 * 4, 256, out_dim)
            weakly_supervised = WeaklySupervised(in_channels, 256, out_dim)

        roi_heads = RoIHeads(
            box_roi_pool=model.roi_heads.box_roi_pool,
            box_head=model.roi_heads.box_head,
            box_predictor=box_pred,
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.25,
            bbox_reg_weights=None,
            score_thresh=0.05,
            nms_thresh=0.5,
            detections_per_img=100,
            mask_roi_pool=model.roi_heads.mask_roi_pool,
            mask_head=model.roi_heads.mask_head,
            mask_predictor=mask_predictor)
            # transfer=transfer,
            # weakly_supervised=weakly_supervised)
        return roi_heads

    def forward(self, images: List[torch.Tensor],
                targets: Optional[List[Dict]] = [None]):
        """The forward propagation"""
        if self.training:
            self._backbone.train()
            self._rpn.train()
            self._heads.train()
        else:
            self._backbone.eval()
            self._rpn.eval()
            self._heads.eval()
        # import pdb; pdb.set_trace()
        orig_sizes = [(i.shape[1], i.shape[2]) for i in images]
        images, targets = self._processing(images, targets)
        targets = targets if targets[0] else None
        base = self._backbone(images.tensors)
        rois, loss_dict = self._rpn(images, base, targets)
        res, loss_head = self._heads(base, rois, images.image_sizes, targets)
        loss_dict.update(loss_head)
        if self.training:
            return loss_dict
        res = self._processing.postprocess(res, images.image_sizes, orig_sizes)
        return res
