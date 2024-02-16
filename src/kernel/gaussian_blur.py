import numpy as np
from src.kernel.kernel import Kernel


def gaussian_blur() -> np.ndarray:
    return 1 / 16 * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])


GaussianBlur = Kernel(as_ndarray=gaussian_blur, name="gaussian_blur")
