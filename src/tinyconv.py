import numpy as np
from PIL import Image


def convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    image = np.pad(image, pad_width=max(kernel.shape[0], kernel.shape[1]), mode="edge") # I can experiment with different modes for different convolution outputs
    output = np.copy(image)

    iterator = np.nditer(image, flags=['multi_index'])
    for pixel in iterator:
        horizonal = (kernel.shape[0] - 1)/2
        vertical = (kernel.shape[1] - 1)/2
        print(horizonal, vertical)
        pixel_surrounding = image[iterator.multi_index[0]:iterator.multi_index[0]+horizonal, iterator.multi_index[1]:iterator.multi_index[1]+vertical]
        output_pixel = np.sum(pixel_surrounding * kernel)
        output[iterator.multi_index] = output_pixel
        print(pixel, iterator.multi_index, output[iterator.multi_index])
        print(kernel.shape)
        
    return output

if __name__ == "__main__":
    a = np.linspace(1, 3, num=3)
    b = np.linspace(4, 6, num=3)
    rng = np.random.default_rng()
    c = (rng.random((64,64, 3)) * 255).astype(np.uint8)
    d = np.full(9, 1/9).reshape(3,3)
    print(c.size)
    print(c.ndim)
    print(c[0][0])
    im = Image.fromarray(c).convert('RGB')
    im.save("output.jpeg")

    print(d)
    convolution(c, d)