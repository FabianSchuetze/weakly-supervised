r"""Contains the supervised model"""
import torch
import torchvision.models as models
from torchvision.models.detection.image_list import ImageList


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

    def __init__(self, n_dim):
        """
        The fully supervised model
        """
        super(Supervised, self).__init__()
        self._backbone, self._rpn = self._get_backbone()
        self._heads = self._get_heads(n_dim)

    def _get_backbone(self):
        """the backbone of the model
        """
        model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        return model.backbone, model.rpn

    # def _get_roi_align(self):
        # model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        # return model.roi_heads.box_roi_pool

    def _get_heads(self, out_dim):
        """The RoI heads"""
        model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        ## == BOX HEAD == ##
        roi_align = model.roi_heads.box_roi_pool
        box_head = model.roi_heads.box_head
        box_pred = TwoHeaded(1024, out_dim)

        ## = MASK HEAD == ##
        mask_roi_pool = model.roi_heads.mask_roi_pool
        mask_head = model.roi_heads.mask_head
        mask_predictor = models.detection.mask_rcnn.MaskRCNNPredictor(
            model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, out_dim)
        fg_iou_thresh = 0.5
        bg_iou_thresh = 0.5
        batch_size_per_image = 512
        positive_fraction = 0.25
        box_reg_weights = None
        score_thresh = 0.05
        nms_thresh = 0.5
        detections_per_img = 100
        roi_heads = model.detection.RoIHeads(
            roi_align,
            box_head,
            box_pred,
            fg_iou_thresh,
            bg_iou_thresh,
            batch_size_per_image,
            positive_fraction,
            box_reg_weights,
            score_thresh,
            nms_thresh,
            detections_per_img,
            mask_roi_pool=mask_roi_pool,
            mask_head=mask_head,
            mask_predictor=mask_predictor)
        return roi_heads

    # TODO: A pre- and postprocessing
    # TODO: Distinguish between training and testing
    def forward(self, img, target):
        # preparation
        if self.training:
            self._base.train()
            self._rpn.train()
            self._heads.train()
        base_features = self._base(img)
        img_shapes = [(img.shape[2], img.shape[3])]
        img_list = ImageList(img, img_shapes)
        rois, loss_dict = self._rpn(img_list, base_features, [target])
        loss_dict_head = self._roi(
            base_features, rois, img_shapes, [target])[1]
        loss_dict.update(loss_dict_head)
        return loss_dict
