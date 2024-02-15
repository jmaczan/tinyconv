import numpy as np
from PIL import Image


def convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    horizontal = int((kernel.shape[0] - 1)/2)
    vertical = int((kernel.shape[1] - 1)/2)
    output = np.copy(image)


    iterator = np.nditer(image, flags=['multi_index'])
    for _ in iterator:
        pixel_surrounding = image[iterator.multi_index[0]:iterator.multi_index[0]+horizontal, iterator.multi_index[1]:iterator.multi_index[1]+vertical]
        output[iterator.multi_index] = np.sum(pixel_surrounding * kernel)
        
    return output

if __name__ == "__main__":
    image = np.array(Image.open('kirby.png').convert('RGB'))
    kernel = np.full(9, 1/9).reshape(3,3)
    
    image = Image.fromarray(convolution(image, kernel)).convert('RGB')
    image.save("output.png")
