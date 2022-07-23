from torchvision.utils import save_image
from PIL import Image
import numpy
import matplotlib.pyplot as plt

def reconstruction_image():
    image_open = Image.open(open('/Users/kaus/Project/Melady Lab/CVAE/output50.jpg', 'rb'))
    array_image = numpy.array(image_open)
    print(array_image.shape)
    image1 = array_image[0].reshape(28,28,3)
    plt.imshow(image1)
    plt.show()
    return image_open

reconstruction_image()