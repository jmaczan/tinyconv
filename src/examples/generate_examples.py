import numpy as np
from PIL import Image

from src.kernel.box_blur import BoxBlur
from src.kernel.gaussian_blur import GaussianBlur
from src.kernel.unsharp_masking import UnsharpMasking
from src.tinyconv import tinyconv


def generate_examples():
    kernels = [BoxBlur, GaussianBlur, UnsharpMasking]

    image = np.array(Image.open('kirby.png').convert('RGB'))

    for kernel in kernels:
        Image.fromarray(tinyconv(image, kernel.as_ndarray())).convert('RGB').save(f"examples/{kernel.name}.png")

if __name__ == "__main__":
    generate_examples()
