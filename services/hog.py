
from skimage.transform import resize
from skimage.feature import hog


class HogGenerator:
    def __init__(self, image, orientations=None, pixels_per_cell=None, cells_per_block=None, visualize=False, block_norm='L2-Hys'):
        self.image = image
        self.orientations = orientations or 9
        self.pixels_per_cell = pixels_per_cell or (8, 8)
        self.cells_per_block = cells_per_block or (2, 2)
        self.visualize = visualize
        self.block_norm = block_norm

    def hog(self):
        img = self.image
        resized_img = resize(img, (128, 64))
        fd = hog(resized_img, orientations=self.orientations, pixels_per_cell=self.pixels_per_cell, cells_per_block=self.cells_per_block, visualize=self.visualize, block_norm=self.block_norm)

        return fd
