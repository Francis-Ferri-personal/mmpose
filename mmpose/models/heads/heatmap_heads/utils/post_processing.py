# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch


def get_heatmaps_maximums(heatmaps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get maximum response location and value from heatmaps (PyTorch version).

    Args:
        heatmaps (torch.Tensor): Heatmaps in shape (K, H, W) or (B, K, H, W)

    Returns:
        tuple:
        - locs (torch.Tensor): locations of maximum heatmap responses in shape
            (K, 2) or (B, K, 2)
        - vals (torch.Tensor): values of maximum heatmap responses in shape
            (K,) or (B, K)
    """
    assert heatmaps.ndim in (3, 4), f'Invalid heatmap shape {heatmaps.shape}'

    single_instance = False
    if heatmaps.ndim == 3:
        # (K, H, W) -> (1, K, H, W)
        heatmaps = heatmaps.unsqueeze(0)
        single_instance = True

    B, K, H, W = heatmaps.shape
    heatmaps_flat = heatmaps.view(B * K, -1)

    # Max values and indices
    vals, idxs = torch.max(heatmaps_flat, dim=1)

    # Convert flat index to 2D coordinates
    y_locs = idxs // W
    x_locs = idxs % W
    locs = torch.stack((x_locs, y_locs), dim=-1).float()

    # Set invalid locations where max <= 0
    locs[vals <= 0] = -1

    # Reshape back to (B, K, 2) and (B, K)
    locs = locs.view(B, K, 2)
    vals = vals.view(B, K)

    if single_instance:
        locs = locs.squeeze(0)  # (K, 2)
        vals = vals.squeeze(0)  # (K,)

    return locs, vals
