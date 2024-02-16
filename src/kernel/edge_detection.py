import numpy as np
from src.kernel.kernel import Kernel


def edge_detection() -> np.ndarray:
    return np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])


EdgeDetection = Kernel(as_ndarray=edge_detection, name="edge_detection")
