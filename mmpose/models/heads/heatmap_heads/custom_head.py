import math
from typing import Dict, Optional, Sequence, Tuple, Union


import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, DepthwiseSeparableConvModule
from mmengine.model import BaseModule, ModuleDict
from mmengine.structures import InstanceData, PixelData
from mmcv.ops import DeformConv2dPack
from torch import Tensor

from mmpose.models.utils.tta import flip_heatmaps
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

class FeatureExtractor:
    # TODO: Why only an index and not a gaussian (circle around the center of the instance). Maybe it is to optimize computations.
    @staticmethod
    def sample_feats(feats: Tensor, indices: Tensor) -> Tensor:
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
        use_bbox: bool = True
    ):
        super().__init__(init_cfg=init_cfg)
        self.use_bbox = use_bbox

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
            # TODO:
            # I here that we can add information of bbox too here. It could be good, not only keypoint visibility. Maybe this way we can have better results.
            # TODO: Maybe reduce only to root the output.
            # TODO: and HACK: Use the contrastive learning to have a feature map for attention with the similarities between pixels 
            out_channels=out_channels, # it is the number of keypoints + root (center of instance or not)
            # kernel_size=1)
            kernel_size=3)
        self.sigmoid = TruncSigmoid(min=clamp_delta, max=1 - clamp_delta)
        # TODO:
        """
        1. Descriptores geométricos (forma/tamaño/orientación)
            Puedes agregar información global sobre la forma y distribución de los keypoints de cada instancia:

            📏 Bounding box estimado (w, h, aspect ratio)

            🧭 Orientación corporal (e.g., vector de dirección hombros → caderas)

            🧩 Dispersión de los keypoints (cuán extendidos están)

            🔄 Invariante a rotaciones/escalado: Momentos geométricos

            Esto se puede aprender como un vector adicional o como un canal auxiliar.

        2. Estado local del fondo / contexto
            Proporcionar a la red una idea del entorno alrededor del centro de la instancia:

            📦 Un crop contextual alrededor del root (usando deformable conv o atención local).

        🔶 4. Embeddings semánticos
Si sabes qué tipo de instancia es (humano, animal, tipo de postura...), puedes incorporar:

🧠 Un vector semántico de clase o acción (caminar, correr, tumbarse)

🔢 Un embedding aprendido que capture contexto de pose
        
 5. Heatmaps auxiliares
🌪️ Heatmap de gradientes de cambio (para ver bordes de instancias)

7. Embedding posicional absoluto o relativo
        """

    def forward(self, feats: Tensor):
        heatmaps = self.keypoint_root_conv(feats)
        heatmaps = self.sigmoid(heatmaps)
        return heatmaps
    

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
            instance_coords (Tensor): Coordinates of the instance roots. (Ground Truth)
            instance_imgids (Tensor): Sample indices of each instances
                in the batch.

        Returns:
            Tuple[Tensor, Tensor]: Extracted feature vectors and heatmaps
                for the instances.
        """
        heatmaps = self.forward(feats) # (b, c, 128, 128) i.e [6, 20 (num_keypoints + 1), 128, 128]
        indices = torch.cat((instance_imgids[:, None], instance_coords), dim=1)  # (samples in batch, 3) 3 is for img idx, x and y
        instance_feats = FeatureExtractor.sample_feats(feats, indices) # (samples in batch (instances), channel in features) i.e (instances, 480)
       
        instance_bboxes = None
        if self.use_bbox:
            bboxes_heatmaps = heatmaps[:, -4:, :, :] # [b, 4, :, :] This means: 4 => (xmin, ymin, xmax, ymax)
            heatmaps = heatmaps[:, :-4, :, :]
            instance_bboxes = FeatureExtractor.sample_feats(bboxes_heatmaps, indices) # [instances, 4]

        return instance_feats, heatmaps, instance_bboxes
    

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
        # NOTE: It seems one feature map per sample in batch. but if that is the case, we are wasting resources in the convolutions.
        # TODO: The output channel of the convolution should be one check it because they are using only one channel for instance it seems.
        heatmaps = self.forward(feats)
        
        if self.use_bbox:
            bboxes_heatmaps = heatmaps[:, -4:, :, :] # [b, 4, :, :] This means: 4 => (xmin, ymin, xmax, ymax)
            heatmaps = heatmaps[:, :-4, :, :]
        
        heatmaps = heatmaps.narrow(1, -1, 1) # KEY: extracts only the last channel, which is the coupled heatmap of the "root" (instance).
        # TODO: print(heatmaps.shape) # it should be (B, 1, H, W)
        
        
        # If flip_test is enabled, average the original and flipped heatmaps for robustness
        if test_cfg.get('flip_test', False):
            heatmaps = heatmaps.mean(dim=0, keepdims=True)
        
        # Then apply Gaussian smoothing to reduce noise and enhance keypoint peaks
        smoothed_heatmaps = smooth_heatmaps(heatmaps, blur_kernel_size)

        # decode heatmaps
        # This step highlights local maxima (peaks in the heatmap) that are possible centers of individuals or keypoints.
        maximums = self._hierarchical_pool(smoothed_heatmaps) # Key: Applies max pooling to detect local maxima (heatmap peaks).

        # Keep only the original heatmap values that are true local maxima
        # (i.e., values equal to their pooled maximum in the neighborhood).
        # TODO: NOTE: What if in addition to the local maximum that is the center of the instance we include the maximum or minimum points that divide two instances as a feature map. I mean this could be an additional feature map to pay attention to, but the relation will be inverse we want to avoid theses in the feature map not to include them.
        maximums = torch.eq(maximums, smoothed_heatmaps).float() # Then, non-local peaks are masked: only values that are equal to their neighboring maxima are kept.
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
        instance_feats = FeatureExtractor.sample_feats(feats, instance_coords)
        
        instance_bboxes =  None
        if self.use_bbox:
            instance_bboxes = FeatureExtractor.sample_feats(bboxes_heatmaps, instance_coords) # [instances, 4]

        return instance_feats, instance_coords, scores, instance_bboxes


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




        # print("chequeando cantidad de Valores", global_feats.shape, instance_feats.shape)# [9, 32, 128, 128]) torch.Size([9, 480])
        # HACK: I think that because the instance features are too simple they lose information.
        # Global features:[instances, 32, 128, 128] (we have same number because we duplicate the input features per instance in the image).  Instance features: [instances, 480] 
        instance_feats = self.atn(instance_feats).unsqueeze(2).unsqueeze(3)
        # print ("Instance features: ", instance_feats.shape) # [instances, 32, 1, 1])
        return global_feats * instance_feats #We assing importance to each channel
    
    # TODO: TODO: WHy we did not add normal channel attention I mean  chnnel attention to the feature map and channel attention to the instance features 
    # TODO: Craete something like instance aware CBAM 
         
    

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
        B, C, H, W = global_feats.size() # [num_instances, 32, 128, 128]



        instance_feats = self.atn(instance_feats).reshape(B, C, 1, 1) # [num_instances, 480] -> [num_instances, 32, 1, 1]
        feats = global_feats * instance_feats.expand_as(global_feats) # TODO 1: Remove this I think is duplicating channel attention
        fsum = torch.sum(feats, dim=1, keepdim=True) # ([num_instances, 1, 128, 128]
        # TODO: Deberiamos aplicar esto al inico. Es como un positional embedding. Tal vez desde el inicio fuera bueno desde que comenzamos el instance information abstarction

        # instance corrds (from 0 to 128)
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
        # Relative cords shape: [num_instances, 16384 (128 x 128), 2]

        # Why 32? This number often comes from the stride or downsampling factor of the backbone or feature map compared to the original input image size.
        # relative_coords = relative_coords.permute(0, 2, 1) / 32.0 # HACK: This was wrong I think because the coordinates were already in 128 x 128 no need to downsampling from original coordinates.
        relative_coords = relative_coords.permute(0, 2, 1) / H # Normalize between 0-1
        # the channels are ordered in (dx, dy) that are  the offset to the reference point center of instance. 
        
        relative_coords = relative_coords.reshape(B, 2, H, W)

        input_feats = torch.cat((fsum, relative_coords), dim=1) # 3 -> 1 channel # TODO: I think that they are not using the positional encoding well. It is not well use in my opinion.

        mask = self.conv(input_feats).sigmoid() # [num_instances, 1, 128, 128]
        # global_feats = [num_instances, 32, 128, 128]

        return global_feats * mask 
    


def gaussian_heatmap(bboxes, heatmap_size, sigma=3):
    """
    Generate a Gaussian heatmap centered at each bounding box.
    
    Args:
        bboxes (Tensor): shape (N, 4) in format (xmin, ymin, xmax, ymax)
        heatmap_size (tuple): (H, W) size of the heatmap
        sigma (float): standard deviation of the Gaussian

    Returns:
        Tensor: heatmaps of shape (N, H, W)
    """
    H, W = heatmap_size
    device = bboxes.device
    
    # create coordinate grid
    yv, xv = torch.meshgrid(
        torch.arange(H, device=device), 
        torch.arange(W, device=device), indexing="ij"
    )
    
    heatmaps = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        # center of the bounding box
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        
        # Gaussian centered at (cx, cy)
        g = torch.exp(-((xv - cx) ** 2 + (yv - cy) ** 2) / (2 * sigma ** 2))
        g = g / g.max()  # normalize to [0,1]
        heatmaps.append(g)
    
    return torch.stack(heatmaps, dim=0)  # (N, H, W)


def gaussian_mask(h, w, cx, cy, sigma, device):
    """Generate a 2D Gaussian mask centered at (cx, cy)."""
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij"
    )
    g = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return g / g.max()  # normalize to [0,1]


# Coordinate attention
class h_sigmoid(nn.Module):
    """Hard Sigmoid: fast approximation of sigmoid.
    Formula: ReLU6(x + 3) / 6, range [0,1]."""
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    """Hard Swish: efficient Swish approximation.
    Formula: x * h_sigmoid(x)."""
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    """Coordinate Attention: embeds spatial (H,W) info into channel attention.
    """
    # TODO: Put a fix value instead of reduction
    def __init__(self, inp, oup, use_reduction=False, reduction=32, mip_channels=32):
        super(CoordAtt, self).__init__()
        # NOTE: I replace this with normal implementation because it hasa non deterministic implementation
        # self.pool_h = nn.AdaptiveAvgPool2d((None, 1)) # None keeps the dimension size equal to the input width
        # self.pool_w = nn.AdaptiveAvgPool2d((1, None)) # None keeps the dimension size equal to the input height

        # Channel reduction
        mip = mip_channels
        if use_reduction:
            mip = max(8, inp // reduction) # max 8 ensures that it never goes below 8 channels, preventing the module from becoming too small
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x # (b, c, 128, 128)
        n,c,h,w = x.size()

        # x_h = self.pool_h(x) # (b, c, 128, 1)
        # x_w = self.pool_w(x).permute(0, 1, 3, 2) # (B, C, 1, W) → (B, C, W, 1) i.e (b, c, 1, 128) → (b, c, 128, 1)   
        # NOTE: The above implementation is not deterministic, so I replace it with the following
        x_h = F.avg_pool2d(x, kernel_size=(1, w))  # (b, c, h, 1)
        x_w = F.avg_pool2d(x, kernel_size=(h, 1))  # (b, c, 1, w)
        x_w = x_w.permute(0, 1, 3, 2)              # (b, c, w, 1)

        y = torch.cat([x_h, x_w], dim=2) # (b, c, 256 (128 +  128), 1)
        y = self.conv1(y) # (b, mip, 256, 1)
        y = self.bn1(y) 
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2) # (b, mip, 128, 1), (b, mip, 128, 1)
        x_w = x_w.permute(0, 1, 3, 2) # (b, mip, 1, 128)

        # Attention weights
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class InstanceConcatenatedCoordAtt(nn.Module):
    """Coordinate Attention: embeds spatial (H,W) info into channel attention.
    """
    # TODO: Put a fix value instead of reduction
    def __init__(self, inp, oup, inst_emb_dim=480, use_reduction=False, reduction=32, mip_channels=32):
        super(InstanceConcatenatedCoordAtt, self).__init__()
        self.inp = inp
        self.oup = oup
        self.inst_emb_dim = inst_emb_dim
        # NOTE: I replace this with normal implementation because it hasa non deterministic implementation
        # self.pool_h = nn.AdaptiveAvgPool2d((None, 1)) # None keeps the dimension size equal to the input width
        # self.pool_w = nn.AdaptiveAvgPool2d((1, None)) # None keeps the dimension size equal to the input height

        # Channel reduction
        mip = mip_channels
        if use_reduction:
            mip = max(8, inp // reduction) # max 8 ensures that it never goes below 8 channels, preventing the module from becoming too small

        # Embedding instancia -> d
        self.inst_mlp = nn.Sequential(
            nn.Linear(inst_emb_dim, mip),
            nn.ReLU(inplace=True)
        )
        
        self.conv1 = nn.Conv2d(inp + mip, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x, inst_feats):
        identity = x # (b, c, 128, 128)
        n,c,h,w = x.size()

        # x_h = self.pool_h(x) # (b, c, 128, 1)
        # x_w = self.pool_w(x).permute(0, 1, 3, 2) # (B, C, 1, W) → (B, C, W, 1) i.e (b, c, 1, 128) → (b, c, 128, 1)   
        # NOTE: The above implementation is not deterministic, so I replace it with the following
        x_h = F.avg_pool2d(x, kernel_size=(1, w))  # (b, c, h, 1)
        x_w = F.avg_pool2d(x, kernel_size=(h, 1))  # (b, c, 1, w)
        x_w = x_w.permute(0, 1, 3, 2)              # (b, c, w, 1)

        # embedding instance expanded
        inst_emb = self.inst_mlp(inst_feats) # [b, d]
        inst_emb = inst_emb.unsqueeze(-1).unsqueeze(-1) # [b, d, 1, 1]
        inst_emb_h = inst_emb.expand(-1, -1, h, 1) # [b, d, H, 1]
        inst_emb_w = inst_emb.expand(-1, -1, w, 1) # [b, d, W, 1]


        # concat with features pooled
        x_h_cat = torch.cat([x_h, inst_emb_h], dim=1)  # [b, c+d, h, 1]
        x_w_cat = torch.cat([x_w, inst_emb_w], dim=1)  # [b, c+d, w, 1]

        y = torch.cat([x_h_cat, x_w_cat], dim=2) # (b, c+d, 256 (128 +  128), 1)
        y = self.conv1(y) # (b, mip, 256, 1)
        y = self.bn1(y) 
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2) # (b, mip, 128, 1), (b, mip, 128, 1)
        x_w = x_w.permute(0, 1, 3, 2) # (b, mip, 1, 128)

        # Attention weights
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
    
def get_pixel_coords(heatmap_size: Tuple, device: str = 'cpu'):
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
        use_bbox: bool = True,
        rel_pos_enc_start: bool = False,
        coord_att_type: bool = False,
        rel_pos_enc_end: bool = False
    ):
        super().__init__(init_cfg=init_cfg)
        self.use_bbox = use_bbox
        self.rel_pos_enc_start = rel_pos_enc_start
        self.coord_att_type = coord_att_type
        self.rel_pos_enc_end = rel_pos_enc_end
    
        # TODO: Deformable Convolutions
        
        # Default
        conv_type = init_cfg[-1]["layer"][0]

        self.conv_down = get_conv_operation(
            conv_type=conv_type,
            in_channels=in_channels,
            out_channels=gfd_channels,
            kernel_size=3)
        
        
        # TODO: Try to TODO: Implement Dynamic Kernel Generation
        # - Use instance embeddings to generate convolution kernels via a small network.
        # - Apply these dynamic kernels on global features for instance-specific responses.

        # TODO: CBAM ?
        # NOTE: Try: Add positional encodings to the attention module input so the model understands relative location.
        # TODO: we could change the channel attention with something more advanced.
        # TODO: try Criss-Cross Attention (CCA)
        # TODO: Try CBAM

        if self.use_bbox:
            gfd_channels = gfd_channels + 1

        if self.rel_pos_enc_start:
            gfd_channels = gfd_channels + 2

        if self.coord_att_type == "Default":
            # Coordinate attention without instance information
            self.coord_attention = CoordAtt(gfd_channels, gfd_channels, mip_channels=32)
        elif self.coord_att_type == "Concatenated":
            # Coordinate attention with instance information concatenated and repeated per pixel.
            # It applies 1x1 conv in the polled h and polled w axi
            self.coord_attention = InstanceConcatenatedCoordAtt(
                inp=gfd_channels,
                oup=gfd_channels,
                mip_channels=32)
            
        # NOTE: These ones is not going to conditionate the coordinate attention which is bad. (Try to avoid POST and Dual)
        # TODO: We can try post coord attention film 
        # TODO: Dual-path (parallel): apply coordinate attention in one banch and apply film in annoter. COncatenate and 1x1 conv
        elif self.coord_att_type == "PreFiLM":
            self.mlp = nn.Sequential(
                nn.Linear(480, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, gfd_channels),
                nn.Sigmoid()  # escala entre 0-1
            )
            self.coord_attention = CoordAtt(gfd_channels, gfd_channels, mip_channels=32)
        elif self.coord_att_type == "PreFiLM-Gated":
            self.mlp = nn.Sequential(
                nn.Linear(480, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, gfd_channels),
                nn.Sigmoid()  # escala entre 0-1
            )
            self.sigmoid_gating = TruncSigmoid(min=clamp_delta, max=1 - clamp_delta)
            self.coord_attention = CoordAtt(gfd_channels, gfd_channels, mip_channels=32)
        else:
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

        if self.rel_pos_enc_end:
            gfd_channels = gfd_channels + 2

        self.heatmap_conv = get_conv_operation(
            conv_type=conv_type,
            in_channels=gfd_channels,
            out_channels=out_channels, # num_keypoints
            # kernel_size=1)
            kernel_size=3)
        self.sigmoid = TruncSigmoid(min=clamp_delta, max=1 - clamp_delta)

    def forward(
        self,
        feats: Tensor,
        instance_feats: Tensor,
        instance_coords: Tensor,
        instance_imgids: Tensor,
        intance_bboxes: Optional[Tensor] = None
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
        b, _, h, w = feats.size() # [batch_size, 480, 128, 128]
        # TODO: podiramos ahorrarnos el downsamplig si hacemos de mas canales la convolucion de IIA y usamos la segunda mitad
        
        # feats  (b, 480, 128, 128)
        # TODO: Maintain 1x1 convs for feature maps
        global_feats = self.conv_down(feats) #  (b, 32, 128, 128)
        # instance image ids: [num_instances]

        # Duplicating global feats for paralelism
        # Select the global feature maps of the images that each instance belongs to,
        # effectively creating a per-instance batch (duplicates features if multiple
        # instances come from the same image).
        global_feats = global_feats[instance_imgids] # [num_instances, 32, 128, 128]
        # IMPORTANT: TODO: Think if it is better to use mask  before or after first convolution.
        if self.use_bbox:
            b, _, h, w = global_feats.shape
            # b, _ = intance_bboxes.shape
            # TODO TODO: FIND ERROR HERE
            intance_bboxes = (intance_bboxes * torch.tensor([h, w, h, w], device=intance_bboxes.device))
            # print(w, h)
            # print(intance_bboxes.shape)
            # print("intance_bboxes", intance_bboxes[:3])
            
            intance_bboxes = torch.clamp(
                intance_bboxes,
                min=torch.tensor([0, 0, 0, 0], device=intance_bboxes.device),
                max=torch.tensor([h, w, h, w], device=intance_bboxes.device)
            ).long()

            bbox_mask = torch.zeros((b, 1, h, w), device=feats.device) 
            for i in range(b):
                x1, y1, x2, y2 = intance_bboxes[i] # xmin, ymin, xmax, ymax
                # print("HERERERERE:", x1, y1, x2, y2)
                
                # Binary mask
                bbox_mask[i, 0, y1:y2 + 1, x1:x2 + 1] = 1

                # # Gaussian mask
                # cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0 # center
                # bw, bh = max(x2 - x1, 1), max(y2 - y1, 1) # sze
                # sigma = 0.1 * min(bw, bh)  # tunable
                # bbox_mask[i, 0] = gaussian_mask(h, w, cx, cy, sigma, feats.device)
                # # Debug info
                # mask = bbox_mask[i, 0] 
                # print(f"[DEBUG] Gaussian mask {i}: min={mask.min().item():.4f}, max={mask.max().item():.4f}")
                # print(f"Sum={mask.sum().item():.4f}, Nonzero={(mask > 0).sum().item()}")

                # # Scale to [0, 9] for console printing
                # scaled = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6) * 9
                # scaled = scaled.int()

                # # Optional: downsample to 16x16 for readability
                # preview = torch.nn.functional.interpolate(
                #     scaled.unsqueeze(0).unsqueeze(0).float(), size=(16, 16), mode="nearest"
                # ).squeeze()

                # print(preview)




            # TODO: we are using only a binary mask. Use a gaussian
            # Concat bbox mask at the end of global_feats
            global_feats = torch.cat((global_feats, bbox_mask), dim=1)

        if self.rel_pos_enc_start:
            pixel_coords = get_pixel_coords((h, w), feats.device)
            relative_coords = instance_coords.reshape(
                -1, 1, 2) - pixel_coords.reshape(1, -1, 2)
            relative_coords = relative_coords.permute(0, 2, 1) / h # Normalize between 0-1 # NOTE: If different h, w we apply individually
            # the channels are ordered in (dx, dy) that are  the offset to the reference point center of instance. 
            relative_coords = relative_coords.reshape(global_feats.shape[0], 2, h, w)

            global_feats = torch.cat((global_feats, relative_coords), dim=1) # [num_instances, 32 + 2, 128, 128]

        if self.coord_att_type == "Default":
            cond_instance_feats = self.coord_attention(global_feats)
            # NOTE: I am not sing relu because it can remove negative values that are valuable information from the feature map.
        elif self.coord_att_type == "Concatenated":
            cond_instance_feats = self.coord_attention(global_feats, instance_feats)
        elif self.coord_att_type == "PreFiLM":
            # Apply FiLM-like scaling to global features based on instance features before coordinate attention
            scale = self.mlp(instance_feats) # (b, 32)
            scale = scale.unsqueeze(-1).unsqueeze(-1) # (b, 32, 1, 1)
            global_feats = global_feats * scale # modulated_feats  (b, 32, h, w)
            
            cond_instance_feats = self.coord_attention(global_feats)
        elif self.coord_att_type == "PreFiLM-Gated":
            # Apply FiLM-like scaling to global features based on instance features before coordinate attention
            inst_mod  = self.mlp(instance_feats) # (b, 32)
            inst_gate = self.sigmoid_gating(inst_mod) # (b, 32)
            inst_gate = inst_gate.unsqueeze(-1).unsqueeze(-1) # (b, 32, 1, 1)
            global_feats = global_feats * inst_gate # modulated_feats  (b, 32, h, w)
            
            cond_instance_feats = self.coord_attention(global_feats)
        else:
            cond_instance_feats = torch.cat(
                (self.channel_attention(global_feats, instance_feats), # i.e ([num_instances, 32, 128, 128], [instances, 480])
                self.spatial_attention(global_feats, instance_feats,
                                        instance_coords)),
                dim=1)
        
            cond_instance_feats = self.fuse_attention(cond_instance_feats) # [num_instances, 32, 128, 128])
            cond_instance_feats = torch.nn.functional.relu(cond_instance_feats)
            
        if self.rel_pos_enc_end:
            # HACK: WARNING: Using relative positional encoding at the end of GFD might be redundant if already applied at the start.
            pixel_coords = get_pixel_coords((h, w), feats.device)
            relative_coords = instance_coords.reshape(
                -1, 1, 2) - pixel_coords.reshape(1, -1, 2)
            relative_coords = relative_coords.permute(0, 2, 1) / h
            relative_coords = relative_coords.reshape(cond_instance_feats.shape[0], 2, h, w)
            cond_instance_feats = torch.cat((cond_instance_feats, relative_coords), dim=1) # [num_instances, 32 + 2, 128, 128]

        cond_instance_feats = self.heatmap_conv(cond_instance_feats) # num_channels: 32 -> 19 (this is num_keypoints)    i.e. [num_instances, 19, 128, 128]
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
                 use_bbox: bool =  False,
                 rel_pos_enc_start: bool = False,
                 coord_att_type: bool = False,
                 rel_pos_enc_end: bool = False,
                 # TODO: Check loses
                 coupled_heatmap_loss: OptConfigType = dict(
                     type='FocalHeatmapLoss'),
                 decoupled_heatmap_loss: OptConfigType = dict(
                     type='FocalHeatmapLoss'),
                 contrastive_loss: OptConfigType = dict(type='InfoNCELoss'),
                 bbox_loss: OptConfigType = dict(type='IoULoss'), # IoULoss, squared

                 # My customizations
                 conv_type: str = 'Conv2d',

                 decoder: OptConfigType = None,
                 init_cfg: OptConfigType = None):
        
        if init_cfg is None:
            init_cfg = self.default_init_cfg
        
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints

        self.use_bbox = use_bbox
        self.rel_pos_enc_start = rel_pos_enc_start
        self.coord_att_type = coord_att_type
        self.rel_pos_enc_end = rel_pos_enc_end

        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        # My customizations
        self.conv_type = conv_type
        self.bbox_format = decoder["bbox_format"]

        # Initialize bias so that sigmoid output starts near prior_prob (e.g., 1%) to stabilize early training
        # prior_prob is the expected probability of a positive prediction (e.g., keypoint) before training starts.
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        # Add bbox channels if needed 
        out_channels = num_keypoints + 1 # (+ root)
        if self.use_bbox:
            out_channels = num_keypoints + 5 # (+ root, x1, y1, x2, y2)
            if self.bbox_format == 'ltwh':
                out_channels = num_keypoints + 5 # (+ root, l, t, w, h)
            elif self.bbox_format == 'wh':
                out_channels = num_keypoints + 3 # (+ root, w, h)
        
        self.custom_iia_module = CustomIIAModule(
            in_channels,
            # TODO: check why did they use num_keypoints + 1
            out_channels,
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
            ],
            use_bbox=self.use_bbox)
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
            ],
            use_bbox=self.use_bbox,
            rel_pos_enc_start=self.rel_pos_enc_start,
            coord_att_type=self.coord_att_type,
            rel_pos_enc_end=self.rel_pos_enc_end)

        # TODO check different lose functions
        # build losses
        self.loss_module = ModuleDict(
            dict(
                heatmap_coupled=MODELS.build(coupled_heatmap_loss),
                heatmap_decoupled=MODELS.build(decoupled_heatmap_loss),
                contrastive=MODELS.build(contrastive_loss),
                bbox=MODELS.build(bbox_loss)
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

        
        # Feats: [2, 480, 128, 240] 
        # NOTE: 2 is for horizontasl flipping, and 240 instaed of 128 is because we are using resize_mode='expand' in validation. This will not affect our results because for trainning we used BottomupRandomAffine
        # BottomupRandomAffine applies random affine transformations that keep the input size while selecting random sections of the image for augmentation.

        instance_info = self.custom_iia_module.forward_test(feats, test_cfg)

        instance_feats, instance_coords, instance_scores, instance_bboxes = instance_info # [pred_instances (x2 if flipped=true), 480], [pred_instances, 2] [pred_instances], [pred_instances, 4]

        if len(instance_coords) > 0:
            # It is a tensor of size [N], where each value indicates which image in the batch each instance belongs to.
            # Note here is zero because all is only one image I think
            instance_imgids = torch.zeros(
                instance_coords.size(0), dtype=torch.long, device=feats.device)
            if test_cfg.get('flip_test', False):
                # NOTE: We keep the same instance_corrds because at the end we flip  the final heatmaps of teh flipped verions of the images.
                instance_coords = torch.cat((instance_coords, instance_coords))
                instance_imgids = torch.cat(
                    (instance_imgids, instance_imgids + 1))
            
            # Call the custom_gfd_module module to generate decoupled heatmaps for each instance using global features, instance features, coordinates, and IDs.
            instance_heatmaps = self.custom_gfd_module(feats, instance_feats,
                                                instance_coords,
                                                instance_imgids,
                                                intance_bboxes=instance_bboxes)
            # instance_heatmaps = [pred_instances (pred_instances x 2 if flip= true), 19, 128, 240])
            
            if test_cfg.get('flip_test', False):
                # flip_indices es una lista o tensor que indica cómo reordenar los canales del heatmap después del flip horizontal para que los keypoints correspondan correctamente. 
                flip_indices = batch_data_samples[0].metainfo['flip_indices']
                # flip_indices = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 17, 18]
                instance_heatmaps, instance_heatmaps_flip = torch.chunk(
                    instance_heatmaps, 2, dim=0)
                # rearranges the flipped heatmap channels so that the keypoints are in the correct order according to the horizontal flip.
                instance_heatmaps_flip = \
                    instance_heatmaps_flip[:, flip_indices, :, :]
                # combine the prediction of the original and flip images
                instance_heatmaps = (instance_heatmaps +
                                     instance_heatmaps_flip) / 2.0
                
            # instance_heatmaps = [pred_instances, 19, 128, 240]
            # instance_scores = [pred_instances]
            instance_heatmaps = smooth_heatmaps( 
                instance_heatmaps, test_cfg.get('blur_kernel_size', 3))
            

            # The decode function typically converts heatmaps into keypoint coordinates (e.g., by taking the maximum in the heatmap for each keypoint).
            # The preds result contains a list or collection of keypoint predictions for each detected instance.
            preds = self.decode((instance_heatmaps, instance_scores[:, None])) # shape = [N, 1]
            
            # InstanceData.cat(preds) combines all individual predictions into a single object
            preds = InstanceData.cat(preds)
            # Print the results
            # for i, p in enumerate(preds):
            #     print("  keypoints shape:", p.keypoints.shape)
            #     print("  keypoint_scores shape:", p.keypoint_scores.shape)
            # For each instance we have:
                # keypoints shape: (1, 19, 2)
                # keypoint_scores shape: (1, 19)

            # Just fitting the heatmap and predictions to the original image
            # To correctly map the coordinates to the original image size, an offset is added.
            # Adds half the size of a heatmap pixel
            # NOTE: We add half the size of a pixel to the original scale to move the edge point to the center of the pixel.
            preds.keypoints[..., 0] += metainfo['input_size'][
                0] / instance_heatmaps.shape[-1] / 2.0
            preds.keypoints[..., 1] += metainfo['input_size'][
                1] / instance_heatmaps.shape[-2] / 2.0
            preds = [preds]

        else: # if no detections returns empty arrays
            preds = [
                InstanceData(
                    keypoints=np.empty((0, self.num_keypoints, 2)),
                    keypoint_scores=np.empty((0, self.num_keypoints)))
            ]
            instance_heatmaps = torch.empty(0, self.num_keypoints,
                                            *feats.shape[-2:])

        # Return heatmaps if requested
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
        # feats: [b, 480, 128, 128]
        # load targets
        gt_heatmaps, gt_instance_coords, keypoint_weights = [], [], []
        heatmap_mask = []

        instance_imgids, gt_instance_heatmaps, gt_instance_bboxes  = [], [], []
        for i, d in enumerate(batch_data_samples):
            gt_heatmaps.append(d.gt_fields.heatmaps) # [20 (num_keypoints + root), 128, 128]
            gt_instance_coords.append(d.gt_instance_labels.instance_coords) # [num_instances, 2]
            # If a keypoint is not annotated (hidden, out of the image, or marked as not visible), its weight (keypoint_weight) is set to 0.0 so that the loss ignores it.
            keypoint_weights.append(d.gt_instance_labels.keypoint_weights)
            instance_imgids.append(
                torch.ones(
                    len(d.gt_instance_labels.instance_coords),
                    dtype=torch.long) * i)

            instance_heatmaps = d.gt_fields.instance_heatmaps.reshape(
                -1, self.num_keypoints,
                *d.gt_fields.instance_heatmaps.shape[1:])
            # instance_heatmaps = [num_instances, num_keypoints, H, W] i.e. [3, 19, 128, 128]
            gt_instance_heatmaps.append(instance_heatmaps)

            if 'heatmap_mask' in d.gt_fields:
                heatmap_mask.append(d.gt_fields.heatmap_mask)

            if 'instance_bboxes' in d.gt_instance_labels:
                gt_instance_bboxes.append(d.gt_instance_labels.instance_bboxes) # [num_instances, 4] this means 4 => (xmin, ymin, xmax, ymax)


        # gt_heatmaps: per-keypoint heatmaps combining all image instances
        gt_heatmaps = torch.stack(gt_heatmaps)
        heatmap_mask = torch.stack(heatmap_mask) if heatmap_mask else None
        gt_instance_coords = torch.cat(gt_instance_coords, dim=0)
        # gt_instance_heatmaps: Separate per-keypoint heatmaps for each individual instance
        gt_instance_heatmaps = torch.cat(gt_instance_heatmaps, dim=0)
        gt_instance_bboxes = torch.cat(gt_instance_bboxes, dim=0) if gt_instance_bboxes else None
        keypoint_weights = torch.cat(keypoint_weights, dim=0)
        instance_imgids = torch.cat(instance_imgids).to(gt_heatmaps.device)
        

        # feed-forward
        feats = feats[-1] # features from the backbone [3, 480, 128, 128]
        pred_instance_feats, pred_heatmaps, pred_instance_bboxes = self.custom_iia_module.forward_train( # [num_instances, 480], [b, 20, 128, 128], [b, 4]
            feats, gt_instance_coords, instance_imgids)
        
        # conpute contrastive loss
        # TODO: CREO QUE SE ESTA PERDIENDO PODER DISCRIMINATIVO AQUI a pesar de que 
        contrastive_loss = 0
        for i in range(len(batch_data_samples)):
            # filters pred_instance_feats to keep only the features of the instances of that image.
            pred_instance_feat = pred_instance_feats[instance_imgids == i]
            # Hack: For each instance we apply contrastive loss. That way we can separate very well the instances.
            contrastive_loss += self.loss_module['contrastive']( # InfoNCELoss
                pred_instance_feat)
        # Average over the number of total instances
        contrastive_loss = contrastive_loss / max(1, len(instance_imgids)) # Avoid division by zero and avoid dependence on the number of instances.

        # TODO: Here I need a losss for bbox

        if self.use_bbox:
            # pred_bboxes [b, 4, :, :] This means: 4 => (xmin, ymin, xmax, ymax)

            # Get heatmap dimensions from predictions
            #  heatmaps_h, heatmaps_w = d.gt_fields.instance_heatmaps.shape[1:]  # Ej: (128, 128)
            heatmaps_h, heatmaps_w = pred_heatmaps.shape[-2:]  # (H, W) => (128 x 128)
            heatmaps_h -= 1 # [0: 127]
            heatmaps_w -= 1 # [0: 127]
            
            
            # compute bbox loss
            bbox_loss = 0
            for i in range(len(batch_data_samples)):
                # Get Groud Thruths
                gt_bboxes = gt_instance_bboxes[instance_imgids == i]
                # filters pred_instance_bboxes to keep only the features of the instances of that image.
                pred_sample_bboxes = pred_instance_bboxes[instance_imgids == i] # [num_instances, 4]

                # if gt_instance_bboxes is not None:
                if len(gt_bboxes) > 0:
                    # Scale GT
                    scale_vector = [heatmaps_w, heatmaps_h, heatmaps_w, heatmaps_h]
                    if self.bbox_format == 'wh':
                        scale_vector = [heatmaps_w, heatmaps_h]

                    gt_bboxes_scaled = gt_bboxes / torch.tensor(
                        scale_vector,
                        device=pred_sample_bboxes.device
                    )

                    bbox_loss += self.loss_module['bbox'](
                        pred_sample_bboxes, gt_bboxes_scaled
                    )

            # Average over the number of total instances
            bbox_loss = bbox_loss / max(1, len(instance_imgids))

                    
        # TODO: Imopelemnt contrastive learning for borders maybe.

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
        # feats = [b, 480, 128, 128]), pred_instance_feats = [instances , 480] instances = num pig in batch (in all the images)
        intance_bboxes = None
        if self.use_bbox:
            intance_bboxes = pred_instance_bboxes
        pred_instance_heatmaps = self.custom_gfd_module(feats, pred_instance_feats,
                                                 gt_instance_coords,
                                                 instance_imgids, intance_bboxes=intance_bboxes)
        

        # calculate losses
        # pred_heatmap & gt_heatmaps = [b, 20 (keypoints + root), 128, 128]
        #  keypoints of all instances  in each keypoint map.
        losses = { # FocalHeatmapLoss
            'loss/heatmap_coupled':
            self.loss_module['heatmap_coupled'](pred_heatmaps, gt_heatmaps,
                                                None, heatmap_mask)
        } 

        # pred instances & gt_instances: [num_instances, 19, 128, 128]
        if len(instance_imgids) > 0:
            # print("PRED", pred_instance_heatmaps)
            # print("GT", gt_instance_heatmaps)

            loss_data = { # FocalHeatmapLoss
                'loss/heatmap_decoupled':
                self.loss_module['heatmap_decoupled'](pred_instance_heatmaps,
                                                      gt_instance_heatmaps,
                                                      keypoint_weights),
                'loss/contrastive':
                contrastive_loss
            }
            
            if self.use_bbox:
                loss_data['loss/bbox'] = bbox_loss

            losses.update(loss_data)

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