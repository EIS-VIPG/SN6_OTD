# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .two_stage_output_backbone import TwoStageDetector_Output_Backbone


@DETECTORS.register_module()
class FasterRCNN_Output_Backbone(TwoStageDetector_Output_Backbone):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(FasterRCNN_Output_Backbone, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
