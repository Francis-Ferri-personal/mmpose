from mmpose.registry import MODELS
from .base_backbone import BaseBackbone


@MODELS.register_module()
class PigNet(BaseBackbone):
    pass