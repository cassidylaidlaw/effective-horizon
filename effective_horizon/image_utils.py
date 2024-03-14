from typing import cast

import cv2
import numpy as np

cv2.ocl.setUseOpenCL(False)


def resize(img: np.ndarray, height: int, width: int) -> np.ndarray:
    return cast(
        np.ndarray, cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    )


def rgb2gray(img: np.ndarray) -> np.ndarray:
    return cast(np.ndarray, cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
