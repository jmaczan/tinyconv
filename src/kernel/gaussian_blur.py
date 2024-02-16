import numpy as np
from kernel.kernel import Kernel


def gaussian_blur() -> np.ndarray:
    return 1/16 * np.array([[1, 2, 1], 
                            [2, 4, 2],
                            [1, 2, 1]])


GaussianBlur = Kernel(gaussian_blur)