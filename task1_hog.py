from services.hog import HogGenerator

from skimage.io import imread, imshow

img = imread('b.png')

hog_generator = HogGenerator(img)




hog_features = hog_generator.hog()

