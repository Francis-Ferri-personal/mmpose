# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import Optional, Tuple

import numpy as np

from mmpose.registry import KEYPOINT_CODECS
from mmpose.structures.bbox import bbox_corner2xyxy

from .utils import (generate_gaussian_heatmaps, get_diagonal_lengths,
                    get_instance_bbox)

from .decoupled_heatmap import DecoupledHeatmap



def get_instance_root(keypoints: np.ndarray,
                    keypoints_visible: Optional[np.ndarray] = None,
                    bboxes: np.ndarray = None,
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
                # NOTE: bboxes come in corner format (four corner points for each box) 
                bbox = bbox_corner2xyxy(bboxes[i]) # transform to xmin, ymin, xmax, ymax
                roots_coordinate[i] = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
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


    label_mapping_table = dict(
        keypoint_weights='keypoint_weights',
        instance_coords='instance_coords',
        instance_bboxes='instance_bboxes',
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
            # TODO: HEre we have to return coordinates of valid bboxes 
        """
        if keypoints_visible is None:
            keypoints_visible = np.ones(keypoints.shape[:2], dtype=np.float32)
        if bbox is None:
            # generate pseudo bbox via visible keypoints
            bbox = get_instance_bbox(keypoints, keypoints_visible)
            bbox = np.tile(bbox, 2).reshape(-1, 4, 2)
            # corner order: left_top, left_bottom, right_top, right_bottom
            bbox[:, 1:3, 0] = bbox[:, 0:2, 0] # Copy x-coordinates from points 0–1 to points 1–2 to align bounding box edges vertically

        # keypoint coordinates in heatmap
        _keypoints = keypoints / self.scale_factor
        # NOTE: bboxes come in corner format (four corner points for each box) 
        # NOTE: This values have the transformations of BottomupRandomAffine that means that the bboxes and keypoints could be outside of the normal range (negative values and more than input_size)
        _bbox = bbox.reshape(-1, 4, 2) / self.scale_factor 


        # NOTE: transformations like translation, scale and rotation introduce values out of range
        # Clip x and y coordinates independently to their respective heatmap dimensions
        bboxes = _bbox.copy()
        bboxes[..., 0] = np.clip(bboxes[..., 0], 0, self.heatmap_size[0] - 1)
        bboxes[..., 1] = np.clip(bboxes[..., 1], 0, self.heatmap_size[1] - 1)
        
        # compute the root and scale of each instance
        roots, roots_visible = get_instance_root(_keypoints, keypoints_visible, bboxes,
                                                 self.root_type)
        
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
        inst_roots, inst_indices, instance_bboxes = [], [], []
        # Calculates the diagonal length of the smallest rectangle covering all visible keypoints for each instance.
        diagonal_lengths = get_diagonal_lengths(_keypoints, keypoints_visible)
        for i in np.argsort(diagonal_lengths):
            # if roots_visible[i] < 1:
            #     continue
            # rand root point in 3x3 grid
            # Randomly jitters the root point within a 3x3 grid around the original location, clamped to heatmap boundaries.
            x, y = roots[i] + np.random.randint(-1, 2, (2, ))
            x = max(0, min(x, self.heatmap_size[0] - 1))
            y = max(0, min(y, self.heatmap_size[1] - 1))
            if (x, y) not in inst_roots:
                inst_roots.append((x, y))
                inst_indices.append(i)
                instance_bboxes.append(bboxes[i]) # NOTE: We are adding cropped bboxes
        if len(inst_indices) > self.encode_max_instances:
            rand_indices = random.sample(
                range(len(inst_indices)), self.encode_max_instances)
            inst_roots = [inst_roots[i] for i in rand_indices]
            inst_indices = [inst_indices[i] for i in rand_indices]
            instance_bboxes = [instance_bboxes[i] for i in rand_indices]

        # Transform instance bboxes format
        instance_bboxes = list(map(lambda i_bbox: bbox_corner2xyxy(i_bbox), instance_bboxes)) # transform to (xmin, ymin, xmax, ymax)

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
            inst_heatmaps = np.concatenate(inst_heatmaps)
            inst_heatmap_weights = np.concatenate(inst_heatmap_weights)
            inst_roots = np.array(inst_roots, dtype=np.int32)
            instance_bboxes = np.array(instance_bboxes, dtype=np.float32)
        else:
            inst_heatmaps = np.empty((0, *self.heatmap_size[::-1]))
            inst_heatmap_weights = np.empty((0, ))
            inst_roots = np.empty((0, 2), dtype=np.int32)
            instance_bboxes = np.empty((0, 4), dtype=np.float32)

        encoded = dict(
            heatmaps=heatmaps,
            instance_heatmaps=inst_heatmaps,
            keypoint_weights=inst_heatmap_weights,
            instance_coords=inst_roots,
            instance_bboxes=instance_bboxes
        )

        return encoded 