import cv2
import numpy as np
import torch
from typing import Union



def read_color_image(color_image: Union[str, np.ndarray]) -> np.ndarray:
    if isinstance(color_image, str):
        color = cv2.imread(color_image)
    elif isinstance(color_image, np.ndarray):
        color = color_image
    else:
        raise ValueError(f"Invalid color image type: {type(color_image)}")
    return color

def read_depth_image(depth_image: Union[str, np.ndarray]) -> np.ndarray:
    if isinstance(depth_image, str):
        depth = cv2.imread(depth_image)
    elif isinstance(depth_image, np.ndarray):
        depth = depth_image
    else:
        raise ValueError(f"Invalid depth image type: {type(depth_image)}")
    # convert depth image to meters
    if depth.dtype == np.uint16:
        depth = depth.astype(np.float32) / 1000.0
    return depth

def _to_numpy(x):
    try:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x) if isinstance(x, (list, tuple)) else x