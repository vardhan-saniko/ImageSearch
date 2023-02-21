from skimage.io import imread, imshow
from PIL import Image
import numpy as np

from services.elbp import ELBPGenerator
image = imread('/Users/vishnuvardhan/Desktop/CSE-515/input_data_sets/set1/image-9.png')

img = Image.fromarray(np.uint8(image), 'L')

elbp_generator = ELBPGenerator(img)


elbp_features = elbp_generator.get_elbp_features()


print("ELBP Features: {}".format(elbp_features))