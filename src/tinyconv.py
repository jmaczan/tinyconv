import numpy as np
from PIL import Image

from src.kernel.kernel import Kernel
from src.kernel.box_blur import BoxBlur


def tinyconv(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    image_height, image_width, rgb_channels = image.shape

    kernel_width = int((kernel.shape[0] - 1) / 2)
    kernel_height = int((kernel.shape[1] - 1) / 2)

    output = np.copy(image)

    for channel in range(rgb_channels):
        for vertical_index in range(kernel_height, image_height - kernel_height):
            for horizontal_index in range(kernel_width, image_width - kernel_width):
                pixel_surrounding = image[
                    vertical_index - kernel_height : vertical_index + kernel_height + 1,
                    horizontal_index
                    - kernel_width : horizontal_index
                    + kernel_width
                    + 1,
                    channel,
                ]
                output[vertical_index, horizontal_index, channel] = np.sum(
                    pixel_surrounding * kernel
                )

    return output


def image_to_image(
    kernel: Kernel = BoxBlur,
    source_image_path: str = "kirby.png",
    output_image_path: str = "output.png",
):
    image = np.array(Image.open(source_image_path).convert("RGB"))
    image = (
        Image.fromarray(tinyconv(image, kernel.as_ndarray()))
        .convert("RGB")
        .save(output_image_path)
    )
