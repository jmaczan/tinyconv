import numpy as np
from PIL import Image
from kernel.box_blur import BoxBlur
from kernel.gaussian_blur import GaussianBlur

from kernel.kernel import Kernel
from kernel.unsharp_masking import UnsharpMasking


def tinyconv(image: np.ndarray, kernel: Kernel) -> np.ndarray:
    image_height, image_width, rgb_channels = image.shape

    kernel_ndarray = kernel.as_ndarray()
    kernel_width = int((kernel_ndarray.shape[0] - 1)/2)
    kernel_height = int((kernel_ndarray.shape[1] - 1)/2)
    
    output = np.copy(image)

    for channel in range(rgb_channels): 
        for vertical_index in range(kernel_height, image_height - kernel_height):
            for horizontal_index in range(kernel_width, image_width - kernel_width):
                pixel_surrounding = image[vertical_index-kernel_height:vertical_index+kernel_height+1, horizontal_index-kernel_width:horizontal_index+kernel_width+1, channel]
                output[vertical_index, horizontal_index, channel] = np.sum(pixel_surrounding * kernel_ndarray)

    return output

if __name__ == "__main__":
    image = np.array(Image.open('kirby.png').convert('RGB'))
    kernel = UnsharpMasking
    
    image = Image.fromarray(tinyconv(image, kernel)).convert('RGB')
    image.save("output.png")
