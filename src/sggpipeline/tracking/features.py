"""Pool shared OWLv2 visual tokens at fixed boxes without another encoder."""

import numpy as np


def pool_box_features(feature_map: np.ndarray, boxes: np.ndarray,
                      image_size: tuple[int, int]) -> np.ndarray:
    """Area-weighted pooling over a (grid_height, grid_width, D) token map.

    Boxes are original-image xyxy pixels. This follows Stage 1's bottom/right
    square padding. Feature extraction, copying and pooling must be included in
    deployment timing when the detector exports these tokens.
    """
    grid, boxes = np.asarray(feature_map, dtype=np.float32), np.asarray(boxes, dtype=np.float32)
    if grid.ndim != 3 or min(grid.shape) == 0 or not np.isfinite(grid).all():
        raise ValueError("Expected a finite nonempty (H,W,D) visual token map")
    if len(image_size) != 2 or not np.isfinite(image_size).all() or min(image_size) <= 0:
        raise ValueError("image_size must be positive (width, height)")
    width, height = image_size
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all():
        raise ValueError("Expected finite (N,4) boxes")
    if ((boxes[:, :2] < 0).any() or (boxes[:, 2:] > [width, height]).any()
            or (boxes[:, 2:] <= boxes[:, :2]).any()):
        raise ValueError("Boxes must have positive area and be clipped to native image bounds")
    side = float(max(image_size))
    x_edges = np.linspace(0, side, grid.shape[1] + 1)
    y_edges = np.linspace(0, side, grid.shape[0] + 1)
    x_overlap = np.maximum(0, np.minimum(boxes[:, 2, None], x_edges[None, 1:])
                           - np.maximum(boxes[:, 0, None], x_edges[None, :-1]))
    y_overlap = np.maximum(0, np.minimum(boxes[:, 3, None], y_edges[None, 1:])
                           - np.maximum(boxes[:, 1, None], y_edges[None, :-1]))
    weights = y_overlap[:, :, None] * x_overlap[:, None, :]
    pooled = np.einsum("nhw,hwd->nd", weights, grid) / weights.sum(axis=(1, 2))[:, None]
    return pooled.astype(np.float32)
