from typing import Callable, NamedTuple, TypeVar
import numpy as np

T = TypeVar("T")


class Kernel(NamedTuple):
    as_ndarray: Callable[[], np.ndarray]