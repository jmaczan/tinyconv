import numpy as np
from src.kernel.kernel import Kernel


def box_blur() -> np.ndarray:
    return np.full(9, 1/9).reshape(3,3)


BoxBlur = Kernel(as_ndarray=box_blur, name="box_blur")