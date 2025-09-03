# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import Optional, Tuple

import numpy as np

from mmpose.registry import KEYPOINT_CODECS
from mmpose.structures.bbox import bbox_xyxy2xywh, bbox_corner2xyxy

from .utils import (generate_gaussian_heatmaps, get_diagonal_lengths,
                    get_instance_bbox)

from .decoupled_heatmap import DecoupledHeatmap



def get_instance_root(keypoints: np.ndarray,
                    keypoints_visible: Optional[np.ndarray] = None,
                    bboxes: np.ndarray = None,
                    bbox_valid: np.ndarray = None,
                    root_type: str = 'bbox_center_real') -> np.ndarray:
        """Calculate the coordinates and visibility of instance roots.

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
            keypoints_visible (np.ndarray): Keypoint visibilities in shape
                (N, K)
            root_type (str): The method to generate the instance root. Options
                are:

                - ``'bbox_center_real'``: Center point of the annotated bounding boxes.
                - ``'kpt_center'``: Average coordinate of all visible keypoints.
                - ``'bbox_center'``: Center point of bounding boxes outlined by
                    all visible keypoints.

                Defaults to ``'bbox_center_real'``

        Returns:
            tuple
            - roots_coordinate(np.ndarray): Coordinates of instance roots in
                shape [N, D]
            - roots_visible(np.ndarray): Visibility of instance roots in
                shape [N]
        """

        roots_coordinate = np.zeros((keypoints.shape[0], 2), dtype=np.float32)
        roots_visible = np.ones((keypoints.shape[0]), dtype=np.float32) * 2

        for i in range(keypoints.shape[0]):

            # collect visible keypoints
            if keypoints_visible is not None:
                visible_keypoints = keypoints[i][keypoints_visible[i] > 0] # create a boolean mask with keypoints_visible[i] > 0
            else:
                visible_keypoints = keypoints[i]
            if visible_keypoints.size == 0: # If no keypoints in the instance,remove the root from visibility
                roots_visible[i] = 0
                continue

            # compute the instance root with visible keypoints
            if root_type == 'bbox_center_real':
                # Filter the invalid bboxes
                if not bbox_valid[i]:
                    roots_visible[i] = 0 # if bbox is invalid remove from visibility
                    continue
                # NOTE: bboxes in format xmin, ymin, xmax, ymax
                roots_coordinate[i] = (bboxes[i][0] + bboxes[i][2]) / 2.0, (bboxes[i][1] + bboxes[i][3]) / 2.0 # (cx, cy)
                roots_visible[i] = 1
            elif root_type == 'kpt_center':
                roots_coordinate[i] = visible_keypoints.mean(axis=0)
                roots_visible[i] = 1
            elif root_type == 'bbox_center':
                roots_coordinate[i] = (visible_keypoints.max(axis=0) +
                                    visible_keypoints.min(axis=0)) / 2.0
                roots_visible[i] = 1
            else:
                raise ValueError(
                    f'the value of `root_type` must be \'kpt_center\' or '
                    f'\'bbox_center\', but got \'{root_type}\'')

        return roots_coordinate, roots_visible

def get_instance_area(bbox: np.ndarray) -> np.ndarray:
    """Calculate the area of an instance bbox.

    Args:
        bbox (np.ndarray): Bounding box in shape (4,) which includes
            coordinates of (xmin, ymin, xmax, ymax).

    Returns:
        np.ndarray: Area of the instance bbox.
    """
    # bbox in format (xmin, ymin, xmax, ymax)
    width = max(0, bbox[2] - bbox[0])
    height = max(0, bbox[3] - bbox[1])
    area = width * height
    return area


@KEYPOINT_CODECS.register_module()
class DecoupledHeatmapBbox(DecoupledHeatmap):
    """Custom DecoupledHeatmap that filters invalid bboxes/keypoints before encoding.
    Args:
        root_type (str): The method to generate the instance root. Options
            are:

            - ``'bbox_center_real'``: Center point of the annotated bounding boxes.
            - ``'kpt_center'``: Average coordinate of all visible keypoints.
            - ``'bbox_center'``: Center point of bounding boxes outlined by
                all visible keypoints.

            Defaults to ``'bbox_center_real'``

    """
    def __init__(
        self,
        input_size: Tuple[int, int],
        heatmap_size: Tuple[int, int],
        root_type: str = 'bbox_center_real',
        heatmap_min_overlap: float = 0.7,
        encode_max_instances: int = 30,
        bbox_format = 'x1y1x2y2', # 'ltwh', "wh"
        **kwargs
    ):
        # Call parent constructor
        super().__init__(
            input_size=input_size,
            heatmap_size=heatmap_size,
            root_type=root_type,
            heatmap_min_overlap=heatmap_min_overlap,
            encode_max_instances=encode_max_instances,
            **kwargs
        )

        self.root_type = root_type
        self.scale_factor = (np.array(input_size) /
                             heatmap_size).astype(np.float32)
        self.bbox_format = bbox_format


    label_mapping_table = dict(
        keypoint_weights='keypoint_weights',
        instance_keypoints='instance_keypoints',
        instance_kps_visible='instance_kps_visible',
        instance_coords='instance_coords',
        instance_bboxes='instance_bboxes',
        instance_areas='instance_areas',
    )


    def encode(self,
               keypoints: np.ndarray,
               keypoints_visible: Optional[np.ndarray] = None,
               bbox: Optional[np.ndarray] = None) -> dict:
        """Encode keypoints into heatmaps.
            Works like DecoupledHeatmap but returns bbox information in the format (xmin, ymin, xmax, ymax).

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
            keypoints_visible (np.ndarray): Keypoint visibilities in shape
                (N, K)
            bbox (np.ndarray): Bounding box in shape (N, 8) which includes
                coordinates of 4 corners.

        Returns:
            dict:
            - heatmaps (np.ndarray): The coupled heatmap in shape
                (1+K, H, W) where [W, H] is the `heatmap_size`. # Extra one due to root
            - instance_heatmaps (np.ndarray): The decoupled heatmap in shape
                (N*K, H, W) where M is the number of instances. # NOTE: probably + 1
            - keypoint_weights (np.ndarray): The weight for heatmaps in shape
                (N*K).
            - instance_coords (np.ndarray): The coordinates of instance roots
                in shape (N, 2)
        """
        if keypoints_visible is None:
            keypoints_visible = np.ones(keypoints.shape[:2], dtype=np.float32)
        if bbox is None: # TODO We can rewrite this to avoid scaling
            # generate pseudo bbox via visible keypoints
            bbox = get_instance_bbox(keypoints, keypoints_visible) # NOTE: We left this as original. we are using other bboxes later
            bbox = np.tile(bbox, 2).reshape(-1, 4, 2)
            # corner order: left_top, left_bottom, right_top, right_bottom
            bbox[:, 1:3, 0] = bbox[:, 0:2, 0] # Copy x-coordinates from points 0–1 to points 1–2 to align bounding box edges vertically

        # keypoint coordinates in heatmap
        _keypoints = keypoints / self.scale_factor
        
        # keypoint visibility filtered
        _keypoints_visible = keypoints_visible.copy()
        # Set keypoints with coordinates outside the heatmap boundaries to invisible
        x_out = (_keypoints[..., 0] < 0) | (_keypoints[..., 0] >= self.heatmap_size[0])
        y_out = (_keypoints[..., 1] < 0) | (_keypoints[..., 1] >= self.heatmap_size[1])
        _keypoints_visible[x_out | y_out] = 0


        # NOTE: bboxes come in corner format (four corner points for each box) 
        # NOTE: This values have the transformations of BottomupRandomAffine that means that the bboxes and keypoints could be outside of the normal range (negative values and more than input_size)
        _bbox = bbox.reshape(-1, 4, 2) / self.scale_factor 


        # NOTE: transformations like translation, scale and rotation introduce values out of range
        # Clip x and y coordinates independently to their respective heatmap dimensions
        bboxes = _bbox.copy()
        bboxes = bbox_corner2xyxy(bboxes)  # transform to (xmin, ymin, xmax, ymax) 

        # Clip the values between [0:heatmap_size-1]
        bboxes[..., [0, 2]] = np.clip(bboxes[..., [0, 2]], 0, self.heatmap_size[0] - 1)  # x
        bboxes[..., [1, 3]] = np.clip(bboxes[..., [1, 3]], 0, self.heatmap_size[1] - 1)  # y

        # Check validity: box is valid if width > 0 and height > 0
        bbox_valid = (bboxes[..., 2] > bboxes[..., 0]) & (bboxes[..., 3] > bboxes[..., 1])

        # compute the root and scale of each instance
        roots, roots_visible = get_instance_root(_keypoints, keypoints_visible, bboxes, bbox_valid,
                                                 self.root_type) # NOTE: We are using original keypoints_visible to determine the root visibility, any way this root has to be positive in the heatmap
        
        sigmas = self._get_instance_wise_sigmas(_bbox)

         # generate global heatmaps
        heatmaps, keypoint_weights = generate_gaussian_heatmaps(
            heatmap_size=self.heatmap_size,
            keypoints=np.concatenate((_keypoints, roots[:, None]), axis=1),
            keypoints_visible=np.concatenate(
                (keypoints_visible, roots_visible[:, None]), axis=1),
            sigma=sigmas)
        
        roots_visible = keypoint_weights[:, -1]

        # select instances
        inst_keypoints, inst_kps_visible,  inst_roots, inst_indices, inst_bboxes, inst_areas = [], [], [], [], [], []
        # Calculates the diagonal length of the smallest rectangle covering all visible keypoints for each instance.
        diagonal_lengths = get_diagonal_lengths(_keypoints, keypoints_visible)
        for i in np.argsort(diagonal_lengths):
            if roots_visible[i] < 1:
                continue
            # rand root point in 3x3 grid
            # Randomly jitters the root point within a 3x3 grid around the original location, clamped to heatmap boundaries.
            x, y = roots[i] + np.random.randint(-1, 2, (2, ))
            x = max(0, min(x, self.heatmap_size[0] - 1))
            y = max(0, min(y, self.heatmap_size[1] - 1))
            if (x, y) not in inst_roots:
                inst_keypoints.append(_keypoints[i] * _keypoints_visible[i][:, None]) # It make zero the invisible keypoints
                inst_kps_visible.append(_keypoints_visible[i])
                inst_roots.append((x, y))
                inst_indices.append(i)
                inst_bboxes.append(bboxes[i]) # NOTE: We are adding cropped bboxes
                inst_areas.append(get_instance_area(bboxes[i]))

        if len(inst_indices) > self.encode_max_instances:
            rand_indices = random.sample(
                range(len(inst_indices)), self.encode_max_instances)
            inst_keypoints = [inst_keypoints[i] for i in rand_indices]
            inst_kps_visible = [inst_kps_visible[i] for i in rand_indices]
            inst_roots = [inst_roots[i] for i in rand_indices]
            inst_indices = [inst_indices[i] for i in rand_indices]
            inst_bboxes = [inst_bboxes[i] for i in rand_indices]
            inst_areas = [inst_areas[i] for i in rand_indices]
        
        # Bbox default format: (xmin, ymin, xmax, ymax)
        empty_instance_bbox = np.empty((0, 4), dtype=np.float32)
        # Transform instance bboxes format
        if self.bbox_format == 'ltwh':
            inst_bboxes = list(map(lambda i_bbox: bbox_xyxy2xywh(np.array(i_bbox).reshape(1, 4)), inst_bboxes))# =>(left, top, width, height) 
            empty_instance_bbox = np.empty((0, 4), dtype=np.float32)
        elif self.bbox_format == 'wh':
            inst_bboxes = list(map(lambda i_bbox: bbox_xyxy2xywh(np.array(i_bbox).reshape(1, 4))[:, 2:], inst_bboxes))# => (left, top, width, height) => (width, height)
            empty_instance_bbox = np.empty((0, 2), dtype=np.float32)

        # generate instance-wise heatmaps
        inst_heatmaps, inst_heatmap_weights = [], []
        for i in inst_indices:
            inst_heatmap, inst_heatmap_weight = generate_gaussian_heatmaps(
                heatmap_size=self.heatmap_size,
                # Uses [i:i + 1] slicing to select the i-th element as a sub-array while preserving the original array’s number of dimensions.
                keypoints=_keypoints[i:i + 1],
                keypoints_visible=keypoints_visible[i:i + 1],
                sigma=sigmas[i].item())
            inst_heatmaps.append(inst_heatmap)
            inst_heatmap_weights.append(inst_heatmap_weight)

        if len(inst_indices) > 0:
            inst_keypoints = np.array(inst_keypoints, dtype=np.float32)
            inst_kps_visible = np.array(inst_kps_visible, dtype=np.float32)
            inst_heatmaps = np.concatenate(inst_heatmaps)
            inst_heatmap_weights = np.concatenate(inst_heatmap_weights)
            inst_roots = np.array(inst_roots, dtype=np.int32)
            inst_bboxes = np.array(inst_bboxes, dtype=np.float32)
            inst_areas = np.array(inst_areas, dtype=np.float32)
        else:
            inst_keypoints = np.empty((0, *keypoints.shape[1:]), dtype=np.float32) # (0, 19, 2)
            inst_kps_visible = np.empty((0, *keypoints_visible.shape[1:]), dtype=np.float32) # (0, 19)
            inst_heatmaps = np.empty((0, *self.heatmap_size[::-1]))
            inst_heatmap_weights = np.empty((0, ))
            inst_roots = np.empty((0, 2), dtype=np.int32)
            inst_bboxes = empty_instance_bbox
            inst_areas = np.empty((0, ), dtype=np.float32)


        encoded = dict(
            instance_keypoints=inst_keypoints,
            instance_kps_visible=inst_kps_visible,
            heatmaps=heatmaps,
            instance_heatmaps=inst_heatmaps,
            keypoint_weights=inst_heatmap_weights,
            instance_coords=inst_roots,
            instance_bboxes=inst_bboxes, # Bboxes in format (xmin, ymin, xmax, ymax) or (left, top, width, height) or (width, height)
            instance_areas=inst_areas,
            )

        return encoded
    
