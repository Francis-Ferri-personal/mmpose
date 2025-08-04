import math
from typing import Dict, Optional, Sequence, Tuple, Union


import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer, DepthwiseSeparableConvModule
from mmengine.model import BaseModule, ModuleDict
from mmengine.structures import InstanceData, PixelData
from mmcv.ops import DeformConv2dPack
from torch import Tensor


from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.typing import (ConfigType, Features, OptConfigType,
                                 OptSampleList, Predictions)
from ..base_head import BaseHead
from .utils import AdaptiveRotatedConv2d, RountingFunction

MODELS.register_module('DepthwiseSeparableConvModule', module=DepthwiseSeparableConvModule)
# Reference: https://www.youtube.com/watch?v=6TtOuVJ9GBQ
MODELS.register_module("DeformConv", module=DeformConv2dPack)

MODELS.register_module("AdaptiveRotatedConv2d", module=AdaptiveRotatedConv2d)


def smooth_heatmaps(heatmaps: Tensor, blur_kernel_size: int) -> Tensor:
    """Smooth the heatmaps by blurring and averaging.

    Args:
        heatmaps (Tensor): The heatmaps to smooth.
        blur_kernel_size (int): The kernel size for blurring the heatmaps.

    Returns:
        Tensor: The smoothed heatmaps.
    """
    smoothed_heatmaps = torch.nn.functional.avg_pool2d(
        heatmaps, blur_kernel_size, 1, (blur_kernel_size - 1) // 2)
    smoothed_heatmaps = (heatmaps + smoothed_heatmaps) / 2.0
    return smoothed_heatmaps


class TruncSigmoid(nn.Sigmoid):
    """A sigmoid activation function that truncates the output to the given
    range.

    Args:
        min (float, optional): The minimum value to clamp the output to.
            Defaults to 0.0
        max (float, optional): The maximum value to clamp the output to.
            Defaults to 1.0
    """

    def __init__(self, min: float = 0.0, max: float = 1.0):
        super(TruncSigmoid, self).__init__()
        self.min = min
        self.max = max

    def forward(self, input: Tensor) -> Tensor:
        """Computes the truncated sigmoid activation of the input tensor."""
        output = torch.sigmoid(input)
        output = output.clamp(min=self.min, max=self.max)
        return output
    

def get_conv_operation(
        conv_type: str, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3
    ) -> nn.Module:
    layer_config = dict(
                type='Conv2d',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size)
    
    layer_config["padding"] = layer_config["kernel_size"] // 2

    if conv_type == "AdaptiveRotatedConv2d":
        kernel_number = 4
        # NOTE: This only works for 3x3 kernels
        layer_config["kernel_size"] = 3
        layer_config["padding"] = layer_config["kernel_size"] // 2
        return AdaptiveRotatedConv2d(
                in_channels=layer_config["in_channels"],
                out_channels=layer_config["out_channels"],
                kernel_size=layer_config["kernel_size"], 
                padding=layer_config["padding"],
                kernel_number=kernel_number,
                rounting_func=RountingFunction(
                    in_channels=layer_config["in_channels"],
                    kernel_number=kernel_number
                ),
            )
    
    elif conv_type == "1x1Conv":
        layer_config["kernel_size"] = 1
        layer_config["padding"] = layer_config["kernel_size"] // 2
    
    elif conv_type == 'DepthwiseSeparableConvModule':
        layer_config["type"] = 'DepthwiseSeparableConvModule'
        # For kernel_size=5, the padding must be 2.
        # If you use dilation=2 and kernel_size=3: padding=2

    elif conv_type == 'DilatedConv':
        # padding = dilation × (k - 1) // 2
        layer_config["dilation"] = 4
        layer_config["padding"] = layer_config["dilation"] * (layer_config["kernel_size"] - 1) // 2 # Keep spatial dims

    elif conv_type == 'DeformConv':
        layer_config["type"] = "DeformConv"
    
    
    return build_conv_layer(layer_config)


# TODO: Rename it as OIIA for Optimized or UIIA
class CustomIIAModule(BaseModule):
    """Modification of CIDHead's IIA module

    Args:
        clamp_delta (float, optional): A small value that prevents the sigmoid
            activation from becoming saturated. Defaults to 1e-4. clamp_delta prevents sigmoid outputs from reaching 0 or 1 exactly, avoiding vanishing gradients due to saturation
        

    """

    #FIXME : Since our individuals (pigs) are homogeneous, not heterogeneous, the IIA module has a more difficult task representing the feature vectors of our pigs. We could even avoid using the individual features and just the position. Try to increase contrastive loss NOTE: They ARe using CONTRASTIVE LOSS JUST BECAUSE OF THIS. They say that if two people are similar, it helps.

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        clamp_delta: float = 1e-4,
        init_cfg: OptConfigType = None,
    ):
        super().__init__(init_cfg=init_cfg)

        conv_type = init_cfg[-1]["layer"][0]

        # TODO: Trata adaptable convolutions en donde junta las instancias, es decir los instance embeddings. Tal vez debas mover este
        """
            2. Dynamic Convolutions
            Convolutional kernels are dynamically generated from embeddings or input conditions (as you mentioned).
            It allows you to adapt filters according to each instance or context.
        """

        # TODO: Trata Deformable convolutions
        # HACK: TODO: Try to implement this  Adaptive Rotated convolutions: Adaptive Rotated Convolution for Rotated Object Detection
        # Default
        self.keypoint_root_conv = get_conv_operation(
            conv_type= conv_type,
            in_channels=in_channels,
            out_channels=out_channels,
            # kernel_size=1)
            kernel_size=3)
        self.sigmoid = TruncSigmoid(min=clamp_delta, max=1 - clamp_delta)

    def forward(self, feats: Tensor):
        # print("HERE:", feats.shape)
        heatmaps = self.keypoint_root_conv(feats)
        # print("HERE:", heatmaps.shape)
        heatmaps = self.sigmoid(heatmaps)
        return heatmaps
    
    # TODO: Why only an index and not a gaussian (circle around the center of the instance). Maybe it is to optimize computations.
    def _sample_feats(self, feats: Tensor, indices: Tensor) -> Tensor:
        """Extract feature vectors at the specified indices from the input
        feature map.

        Args:
            feats (Tensor): Input feature map.
            indices (Tensor): Indices of the feature vectors to extract.

        Returns:
            Tensor: Extracted feature vectors.
        """
        assert indices.dtype == torch.long
        if indices.shape[1] == 3:
            b, w, h = [ind.squeeze(-1) for ind in indices.split(1, -1)]
            instance_feats = feats[b, :, h, w]
        elif indices.shape[1] == 2:
            w, h = [ind.squeeze(-1) for ind in indices.split(1, -1)]
            instance_feats = feats[:, :, h, w]
            instance_feats = instance_feats.permute(0, 2, 1)
            instance_feats = instance_feats.reshape(-1,
                                                    instance_feats.shape[-1])

        else:
            raise ValueError(f'`indices` should have 2 or 3 channels, '
                             f'but got f{indices.shape[1]}')
        return instance_feats
    

    def _hierarchical_pool(self, heatmaps: Tensor) -> Tensor:
        """Conduct max pooling on the input heatmaps with different kernel size
        according to the input size.

        Use average of height and width (not area) to estimate spatial scale and choose pooling size accordingly

        - If the heat map is large, using a large kernel helps capture broader regions of maximum activation.

        - If the map is small, a smaller kernel prevents over-smoothing the information.

        Args:
            heatmaps (Tensor): Input heatmaps.

        Returns:
            Tensor: Result of hierarchical pooling.
        """
        map_size = (heatmaps.shape[-1] + heatmaps.shape[-2]) / 2.0
        if map_size > 300:
            maxm = torch.nn.functional.max_pool2d(heatmaps, 7, 1, 3)
        elif map_size > 200:
            maxm = torch.nn.functional.max_pool2d(heatmaps, 5, 1, 2)
        else:
            maxm = torch.nn.functional.max_pool2d(heatmaps, 3, 1, 1)
        return maxm
    

    def forward_train(self, feats: Tensor, instance_coords: Tensor,
                      instance_imgids: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass during training.

        Args:
            feats (Tensor): Input feature tensor.
            instance_coords (Tensor): Coordinates of the instance roots.
            instance_imgids (Tensor): Sample indices of each instances
                in the batch.

        Returns:
            Tuple[Tensor, Tensor]: Extracted feature vectors and heatmaps
                for the instances.
        """
        heatmaps = self.forward(feats)
        indices = torch.cat((instance_imgids[:, None], instance_coords), dim=1)
        instance_feats = self._sample_feats(feats, indices)

        return instance_feats, heatmaps
    

    def forward_test(
        self, feats: Tensor, test_cfg: Dict
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """Forward pass during testing.

        Args:
            feats (Tensor): Input feature tensor.
            test_cfg (Dict): Testing configuration, including:
                - blur_kernel_size (int, optional): Kernel size for blurring
                    the heatmaps. Defaults to 3.
                - max_instances (int, optional): Maximum number of instances
                    to extract. Defaults to 30.
                - score_threshold (float, optional): Minimum score for
                    extracting an instance. Defaults to 0.01.
                - flip_test (bool, optional): Whether to compute the average
                    of the heatmaps across the batch dimension.
                    Defaults to False.

        Returns:
            A tuple of Tensor including extracted feature vectors,
            coordinates, and scores of the instances. Any of these can be
            empty Tensor if no instances are extracted.
        """
        blur_kernel_size = test_cfg.get('blur_kernel_size', 3)
        max_instances = test_cfg.get('max_instances', 30)
        score_threshold = test_cfg.get('score_threshold', 0.01)
        H, W = feats.shape[-2:]

        # compute heatmaps
        # Extract the last channel (coupled heatmap) to locate instances during inference.
        # That last channel is used to decide where there are instances in the image.
        # NOTE: It seems one feature map per sampel in batch. but if that is the case, we are wasting resources in the convolutions.
        # TODO: The output channel of the convolution should be one check it because they arre using only one channe for instance it seems.
        heatmaps = self.forward(feats).narrow(1, -1, 1)
        # TODO: print(heatmaps.shape) # it should be (B, 1, H, W)
        
        # If flip_test is enabled, average the original and flipped heatmaps for robustness
        if test_cfg.get('flip_test', False):
            heatmaps = heatmaps.mean(dim=0, keepdims=True)
        
        # Then apply Gaussian smoothing to reduce noise and enhance keypoint peaks
        smoothed_heatmaps = smooth_heatmaps(heatmaps, blur_kernel_size)

        # decode heatmaps
        # This step highlights local maxima (peaks in the heatmap) that are possible centers of individuals or keypoints.
        maximums = self._hierarchical_pool(smoothed_heatmaps)

        # Keep only the original heatmap values that are true local maxima
        # (i.e., values equal to their pooled maximum in the neighborhood).
        # TODO: NOTE: What if in addition to the local maximum that is the center of the instance we include the maximum or minimum points that divide two instances as a feature map. I mean this could be an additional feature map to pay attention to, but the relation will be inverse we want to avoid theses in the feature map not to include them.
        maximums = torch.eq(maximums, smoothed_heatmaps).float()
        maximums = (smoothed_heatmaps * maximums).reshape(-1) # flattens the vector i.e. [0.1, 0.0, 0.9, 0.0, ..., 0.0]
        scores, pos_ind = maximums.topk(max_instances, dim=0)
        select_ind = (scores > (score_threshold)).nonzero().squeeze(1)
        scores, pos_ind = scores[select_ind], pos_ind[select_ind]

        # sample feature vectors from feature map
        # Transform to coordinates again
        instance_coords = torch.stack((pos_ind % W, pos_ind // W), dim=1)
        # Get features at that position
        # TODO: We should try the gaussian thing that I propose to have a region around the center of the instance.
        # TODO: Adaptive convolution could also be useful. Although I think the whole attention part is done later in the other module.
        instance_feats = self._sample_feats(feats, instance_coords)

        return instance_feats, instance_coords, scores


class ChannelAttention(nn.Module):
    """Channel-wise attention module introduced in `CID`.

    Args:
        in_channels (int): The number of channels of the input instance
            vectors.
        out_channels (int): The number of channels of the transformed instance
            vectors.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(ChannelAttention, self).__init__()
        self.atn = nn.Linear(in_channels, out_channels)

    def forward(self, global_feats: Tensor, instance_feats: Tensor) -> Tensor:
        """Applies attention to the channel dimension of the input tensor."""

        instance_feats = self.atn(instance_feats).unsqueeze(2).unsqueeze(3)
        return global_feats * instance_feats
    

class SpatialAttention(nn.Module):
    """Spatial-wise attention module introduced in `CID`.

    Args:
        in_channels (int): The number of channels of the input instance
            vectors.
        out_channels (int): The number of channels of the transformed instance
            vectors.
    """

    def __init__(self, in_channels, out_channels, conv_type='Conv2d'):
        super(SpatialAttention, self).__init__()
        """
            ChannelAttention and SpatialAttention both define a linear layer named 'atn',
            but they serve distinct purposes and are trained independently.

            - In ChannelAttention, the 'atn' layer learns to scale each channel of
            the instance features, emphasizing or suppressing them as needed.

            - In SpatialAttention, a separate 'atn' layer is used to project the
            instance features into a space suitable for computing spatial masks
            that modulate feature maps spatially.

            Even though the layer names and structures are similar, their roles are
            functionally different within the model's attention mechanisms.
        """
        self.atn = nn.Linear(in_channels, out_channels)
        """
            This indicates the stride in the previous layers of the network (for example, in a convolutional backbone).

            It means that one unit in the feature map is equivalent to 4 pixels in the original image.

            This value is used, for example, when converting coordinates between the original space and the feature map space (as in spatial attention with relative coordinates).
        """
        self.feat_stride = 4
        self.conv = get_conv_operation(conv_type=conv_type, in_channels=3, out_channels=1, kernel_size=5)

    def _get_pixel_coords(self, heatmap_size: Tuple, device: str = 'cpu'):
        """Get pixel coordinates for each element in the heatmap.

        Args:
            heatmap_size (tuple): Size of the heatmap in (W, H) format.
            device (str): Device to put the resulting tensor on.

        Returns:
            Tensor of shape (batch_size, num_pixels, 2) containing the pixel
            coordinates for each element in the heatmap.

        Comment:
            Generates a grid of (x, y) pixel coordinates for the heatmap. For example, if heatmap size is W=3, H=2, output is:
                [[[0.5, 0.5],
                [1.5, 0.5],
                [2.5, 0.5],
                [0.5, 1.5],
                [1.5, 1.5],
                [2.5, 1.5]]]
            This gives precise location information in floating-point values for each pixel.
        """
        w, h = heatmap_size
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
        pixel_coords = torch.stack((x, y), dim=-1).reshape(-1, 2)
        pixel_coords = pixel_coords.float().to(device) + 0.5
        return pixel_coords

    def forward(self, global_feats: Tensor, instance_feats: Tensor,
                instance_coords: Tensor) -> Tensor:
        """Perform spatial attention.

        Args:
            global_feats (Tensor): Tensor containing the global features.
            instance_feats (Tensor): Tensor containing the instance feature
                vectors.
            instance_coords (Tensor): Tensor containing the root coordinates
                of the instances.

        Returns:
            Tensor containing the modulated global features.
        """
        B, C, H, W = global_feats.size()

        instance_feats = self.atn(instance_feats).reshape(B, C, 1, 1)
        feats = global_feats * instance_feats.expand_as(global_feats)
        fsum = torch.sum(feats, dim=1, keepdim=True)

        pixel_coords = self._get_pixel_coords((W, H), feats.device)
        relative_coords = instance_coords.reshape(
            -1, 1, 2) - pixel_coords.reshape(1, -1, 2)

        """
            i.e. 
            instance_coords = [[1.5, 1.5]]  # shape (1, 1, 2)
            pixel_coords = [[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]]  # shape (1, 4, 2)
            relative_coords = instance_coords - pixel_coords =
            [
            [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
            ]  # shape (1, 4, 2)
        """
        relative_coords = relative_coords.permute(0, 2, 1) / 32.0 
        # the channels are ordered in (dx, dy) that are  the offset to the reference point center of instance. 
        # Why 32? This number often comes from the stride or downsampling factor of the backbone or feature map compared to the original input image size.
        
        relative_coords = relative_coords.reshape(B, 2, H, W)

        input_feats = torch.cat((fsum, relative_coords), dim=1)
        mask = self.conv(input_feats).sigmoid()
        return global_feats * mask
    


class CustomGFDModule(BaseModule):
    """Modification of CIDHead's GFD module

    NOTE:
        `gfd_channels` specifies the number of channels in the transformed feature map within the GFD module.

        A higher value enables the model to capture richer and more detailed features, which can improve accuracy.
        However, it also increases the number of parameters, memory consumption, and computational cost,
        potentially slowing down training and inference.

        # TODO: Get a lower but rich representation
        Conversely, a lower value reduces computational requirements but may limit the model's capacity
        to represent complex features.

        Therefore, `gfd_channels` serves as a trade-off hyperparameter between model performance and efficiency.

        """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        gfd_channels: int,
        clamp_delta: float = 1e-4,
        init_cfg: OptConfigType = None,
    ):
        super().__init__(init_cfg=init_cfg)
    
        # TODO: Deformable Convolutions
        
        # Default
        conv_type = init_cfg[-1]["layer"][0]

        self.conv_down = get_conv_operation(
            conv_type=conv_type,
            in_channels=in_channels,
            out_channels=gfd_channels,
            # kernel_size=1)
            kernel_size=3)
        
        
        # TODO: Try to TODO: Implement Dynamic Kernel Generation
        # - Use instance embeddings to generate convolution kernels via a small network.
        # - Apply these dynamic kernels on global features for instance-specific responses.

        # TODO: CBAM ?
        # NOTE: Try: Add positional encodings to the attention module input so the model understands relative location.
        # TODO: we could change the channel attention with something more advanced.
        self.channel_attention = ChannelAttention(in_channels, gfd_channels)
        # TODO: we could change the spatial attention with something more advanced.
        self.spatial_attention = SpatialAttention(in_channels, gfd_channels, conv_type)
        
        self.fuse_attention = get_conv_operation(
            conv_type=conv_type,
            in_channels=gfd_channels * 2,
            out_channels=gfd_channels,
            # kernel_size=1)
            kernel_size=3)
        
        """
        heatmap_conv:
            This layer is also a convolutional layer, but its purpose is to produce K output channels—one for each keypoint.

            It is typically a 1×1 convolution where out_channels = num_keypoints.
        """

        self.heatmap_conv = get_conv_operation(
            conv_type=conv_type,
            in_channels=gfd_channels,
            out_channels=out_channels,
            # kernel_size=1)
            kernel_size=3)
        self.sigmoid = TruncSigmoid(min=clamp_delta, max=1 - clamp_delta)

    def forward(
        self,
        feats: Tensor,
        instance_feats: Tensor,
        instance_coords: Tensor,
        instance_imgids: Tensor,
    ) -> Tensor:
        """Extract decoupled heatmaps for each instance.

        Args:
            feats (Tensor): Input feature maps.
            instance_feats (Tensor): Tensor containing the instance feature
                vectors.
            instance_coords (Tensor): Tensor containing the root coordinates
                of the instances.
            instance_imgids (Tensor): Sample indices of each instances
                in the batch.

        Returns:
            A tensor containing decoupled heatmaps.
        """

        global_feats = self.conv_down(feats)
        global_feats = global_feats[instance_imgids]
        cond_instance_feats = torch.cat(
            (self.channel_attention(global_feats, instance_feats),
             self.spatial_attention(global_feats, instance_feats,
                                    instance_coords)),
            dim=1)
        
        cond_instance_feats = self.fuse_attention(cond_instance_feats)
        cond_instance_feats = torch.nn.functional.relu(cond_instance_feats)
        cond_instance_feats = self.heatmap_conv(cond_instance_feats)
        heatmaps = self.sigmoid(cond_instance_feats)

        return heatmaps



@MODELS.register_module()
class CustomHead(BaseHead):
    """Modification of CIDHead
    
    """
    _version = 1

    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 gfd_channels: int,
                 num_keypoints: int,
                 prior_prob: float = 0.01,
                 # TODO: Check loses
                 coupled_heatmap_loss: OptConfigType = dict(
                     type='FocalHeatmapLoss'),
                 decoupled_heatmap_loss: OptConfigType = dict(
                     type='FocalHeatmapLoss'),
                 contrastive_loss: OptConfigType = dict(type='InfoNCELoss'),

                 # My customizations
                 conv_type: str = 'Conv2d',

                 decoder: OptConfigType = None,
                 init_cfg: OptConfigType = None):
        
        if init_cfg is None:
            init_cfg = self.default_init_cfg
        
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        # My customizations
        self.conv_type = conv_type

        # Initialize bias so that sigmoid output starts near prior_prob (e.g., 1%) to stabilize early training
        # prior_prob is the expected probability of a positive prediction (e.g., keypoint) before training starts.
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        self.custom_iia_module = CustomIIAModule(
            in_channels,
            # TODO: check why did they use num_keypoints + 1
            num_keypoints + 1,
            init_cfg=init_cfg + [
                # This dictionary seems the default do not pay attention to Linear
                dict(
                    type='Normal',
                    layer=[self.conv_type, 'Linear'],
                    std=0.001,
                    override=dict(
                        name='keypoint_root_conv',
                        type='Normal',
                        std=0.001,
                        bias=bias_value))
            ])
        self.custom_gfd_module = CustomGFDModule(
            in_channels,
            num_keypoints,
            gfd_channels,
            init_cfg=init_cfg + [
                dict(
                    type='Normal',
                    layer=[self.conv_type, 'Linear'],
                    std=0.001,
                    override=dict(
                        name='heatmap_conv',
                        type='Normal',
                        std=0.001,
                        bias=bias_value))
            ])

        # TODO check different lose functions
        # build losses
        self.loss_module = ModuleDict(
            dict(
                heatmap_coupled=MODELS.build(coupled_heatmap_loss),
                heatmap_decoupled=MODELS.build(decoupled_heatmap_loss),
                contrastive=MODELS.build(contrastive_loss),
            ))
        
        
        # Register the hook to automatically convert old version state dicts
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    @property
    def default_init_cfg(self):
        init_cfg = [
            dict(type='Normal', layer=['Conv2d', 'Linear'], std=0.001),
            dict(type='Constant', layer='BatchNorm2d', val=1)
        ]
        return init_cfg
    

    def forward(self, feats: Tuple[Tensor]) -> Tensor:
        """Forward the network. The input is multi scale feature maps and the
        output is the heatmap.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            Tensor: output heatmap.
        """
        feats = feats[-1]
        instance_info = self.custom_iia_module.forward_test(feats, {})
        instance_feats, instance_coords, instance_scores = instance_info
        # This tensor maps each instance to the index of its corresponding image in the batch.
        #  Example: 3 instances all from image 0 in batch
        instance_imgids = torch.zeros(
            instance_coords.size(0), dtype=torch.long, device=feats.device)
        # tensor([0, 0, 0, 1, 2, 2, 2, 3, 4, 5, 5], device='cuda:0')
        instance_heatmaps = self.custom_gfd_module(feats, instance_feats,
                                            instance_coords, instance_imgids)

        return instance_heatmaps

    def predict(self,
                feats: Features,
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:
        """Predict results from features.

        Args:
            feats (Tuple[Tensor] | List[Tuple[Tensor]]): The multi-stage
                features (or multiple multi-stage features in TTA)
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            test_cfg (dict): The runtime config for testing process. Defaults
                to {}

        Returns:
            Union[InstanceList | Tuple[InstanceList | PixelDataList]]: If
            ``test_cfg['output_heatmap']==True``, return both pose and heatmap
            prediction; otherwise only return the pose prediction.

            The pose prediction is a list of ``InstanceData``, each contains
            the following fields:

                - keypoints (np.ndarray): predicted keypoint coordinates in
                    shape (num_instances, K, D) where K is the keypoint number
                    and D is the keypoint dimension
                - keypoint_scores (np.ndarray): predicted keypoint scores in
                    shape (num_instances, K)

            The heatmap prediction is a list of ``PixelData``, each contains
            the following fields:

                - heatmaps (Tensor): The predicted heatmaps in shape (K, h, w)
        """
        metainfo = batch_data_samples[0].metainfo

        if test_cfg.get('flip_test', False):
            assert isinstance(feats, list) and len(feats) == 2

            feats_flipped = flip_heatmaps(feats[1][-1], shift_heatmap=False)
            feats = torch.cat((feats[0][-1], feats_flipped))
        else:
            feats = feats[-1]

        instance_info = self.custom_iia_module.forward_test(feats, test_cfg)
        instance_feats, instance_coords, instance_scores = instance_info
        if len(instance_coords) > 0:
            # It is a tensor of size [N], where each value indicates which image in the batch each instance belongs to.
            # Note here is zero because all is only one image I think
            instance_imgids = torch.zeros(
                instance_coords.size(0), dtype=torch.long, device=feats.device)
            if test_cfg.get('flip_test', False):
                instance_coords = torch.cat((instance_coords, instance_coords))
                instance_imgids = torch.cat(
                    (instance_imgids, instance_imgids + 1))
            instance_heatmaps = self.custom_gfd_module(feats, instance_feats,
                                                instance_coords,
                                                instance_imgids)
            if test_cfg.get('flip_test', False):
                flip_indices = batch_data_samples[0].metainfo['flip_indices']
                instance_heatmaps, instance_heatmaps_flip = torch.chunk(
                    instance_heatmaps, 2, dim=0)
                instance_heatmaps_flip = \
                    instance_heatmaps_flip[:, flip_indices, :, :]
                instance_heatmaps = (instance_heatmaps +
                                     instance_heatmaps_flip) / 2.0
            instance_heatmaps = smooth_heatmaps(
                instance_heatmaps, test_cfg.get('blur_kernel_size', 3))

            preds = self.decode((instance_heatmaps, instance_scores[:, None])) # shape = [N, 1]
            preds = InstanceData.cat(preds)
            # Just fitting the heatmap and predictions to the original image
            preds.keypoints[..., 0] += metainfo['input_size'][
                0] / instance_heatmaps.shape[-1] / 2.0
            preds.keypoints[..., 1] += metainfo['input_size'][
                1] / instance_heatmaps.shape[-2] / 2.0
            preds = [preds]

        else:
            preds = [
                InstanceData(
                    keypoints=np.empty((0, self.num_keypoints, 2)),
                    keypoint_scores=np.empty((0, self.num_keypoints)))
            ]
            instance_heatmaps = torch.empty(0, self.num_keypoints,
                                            *feats.shape[-2:])

        if test_cfg.get('output_heatmaps', False):
            pred_fields = [
                PixelData(
                    heatmaps=instance_heatmaps.reshape(
                        -1, *instance_heatmaps.shape[-2:]))
            ]
            return preds, pred_fields
        else:
            return preds
    
    def loss(self,
            feats: Tuple[Tensor],
            batch_data_samples: OptSampleList,
            train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            feats (Tuple[Tensor]): The multi-stage features
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            train_cfg (dict): The runtime config for training process.
                Defaults to {}

        Returns:
            dict: A dictionary of losses.
        """
        # load targets
        gt_heatmaps, gt_instance_coords, keypoint_weights = [], [], []
        heatmap_mask = []
        instance_imgids, gt_instance_heatmaps = [], []
        for i, d in enumerate(batch_data_samples):
            gt_heatmaps.append(d.gt_fields.heatmaps)
            gt_instance_coords.append(d.gt_instance_labels.instance_coords)
            keypoint_weights.append(d.gt_instance_labels.keypoint_weights)
            instance_imgids.append(
                torch.ones(
                    len(d.gt_instance_labels.instance_coords),
                    dtype=torch.long) * i)

            instance_heatmaps = d.gt_fields.instance_heatmaps.reshape(
                -1, self.num_keypoints,
                *d.gt_fields.instance_heatmaps.shape[1:])
            gt_instance_heatmaps.append(instance_heatmaps)

            if 'heatmap_mask' in d.gt_fields:
                heatmap_mask.append(d.gt_fields.heatmap_mask)

        gt_heatmaps = torch.stack(gt_heatmaps)
        heatmap_mask = torch.stack(heatmap_mask) if heatmap_mask else None

        gt_instance_coords = torch.cat(gt_instance_coords, dim=0)
        gt_instance_heatmaps = torch.cat(gt_instance_heatmaps, dim=0)
        keypoint_weights = torch.cat(keypoint_weights, dim=0)
        instance_imgids = torch.cat(instance_imgids).to(gt_heatmaps.device)

        # feed-forward
        feats = feats[-1]
        pred_instance_feats, pred_heatmaps = self.custom_iia_module.forward_train(
            feats, gt_instance_coords, instance_imgids)

        # conpute contrastive loss
        contrastive_loss = 0
        for i in range(len(batch_data_samples)):
            pred_instance_feat = pred_instance_feats[instance_imgids == i]
            contrastive_loss += self.loss_module['contrastive'](
                pred_instance_feat)
        contrastive_loss = contrastive_loss / max(1, len(instance_imgids))

        # limit the number of instances
        max_train_instances = train_cfg.get('max_train_instances', -1)
        # Only apply limit if it is set and the number of instances exceeds it
        if (max_train_instances > 0
                and len(instance_imgids) > max_train_instances):
            selected_indices = torch.randperm(
                len(instance_imgids),
                device=gt_heatmaps.device,
                dtype=torch.long)[:max_train_instances]
            
            # Apply selection to all relevant tensors
            gt_instance_coords = gt_instance_coords[selected_indices]
            keypoint_weights = keypoint_weights[selected_indices]
            gt_instance_heatmaps = gt_instance_heatmaps[selected_indices]
            instance_imgids = instance_imgids[selected_indices]
            pred_instance_feats = pred_instance_feats[selected_indices]

        # calculate the decoupled heatmaps for each instance
        pred_instance_heatmaps = self.custom_gfd_module(feats, pred_instance_feats,
                                                 gt_instance_coords,
                                                 instance_imgids)

        # calculate losses
        losses = {
            'loss/heatmap_coupled':
            self.loss_module['heatmap_coupled'](pred_heatmaps, gt_heatmaps,
                                                None, heatmap_mask)
        }
        if len(instance_imgids) > 0:
            losses.update({
                'loss/heatmap_decoupled':
                self.loss_module['heatmap_decoupled'](pred_instance_heatmaps,
                                                      gt_instance_heatmaps,
                                                      keypoint_weights),
                'loss/contrastive':
                contrastive_loss
            })

        return losses
    
    def _load_state_dict_pre_hook(self, state_dict, prefix, local_meta, *args,
                                  **kwargs):
        """A hook function to convert old-version state dict of
        :class:`CIDHead` (before MMPose v1.0.0) to a compatible format
        of :class:`CIDHead`.

        The hook will be automatically registered during initialization.
        """
        version = local_meta.get('version', None)
        if version and version >= self._version:
            return

        # convert old-version state dict
        keys = list(state_dict.keys())
        for k in keys:
            if 'keypoint_center_conv' in k:
                v = state_dict.pop(k)
                k = k.replace('keypoint_center_conv',
                              'custom_iia_module.keypoint_root_conv')
                state_dict[k] = v

            if 'conv_down' in k:
                v = state_dict.pop(k)
                k = k.replace('conv_down', 'custom_gfd_module.conv_down')
                state_dict[k] = v

            if 'c_attn' in k:
                v = state_dict.pop(k)
                k = k.replace('c_attn', 'custom_gfd_module.channel_attention')
                state_dict[k] = v

            if 's_attn' in k:
                v = state_dict.pop(k)
                k = k.replace('s_attn', 'custom_gfd_module.spatial_attention')
                state_dict[k] = v

            if 'fuse_attn' in k:
                v = state_dict.pop(k)
                k = k.replace('fuse_attn', 'custom_gfd_module.fuse_attention')
                state_dict[k] = v

            if 'heatmap_conv' in k:
                v = state_dict.pop(k)
                k = k.replace('heatmap_conv', 'custom_gfd_module.heatmap_conv')
                state_dict[k] = v