import numpy as np
from kernel.kernel import Kernel


def unsharp_masking() -> np.ndarray:
    return -1/256 * np.array([[1, 4, 6, 4, 1], 
                            [4, 16, 24, 16, 4],
                            [6, 24, -476, 24, 6],
                            [1, 4, 6, 4, 1], 
                            [4, 16, 24, 16, 4]])


UnsharpMasking = Kernel(unsharp_masking)