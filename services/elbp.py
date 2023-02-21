
from skimage.feature import local_binary_pattern
from scipy import stats


class ELBPGenerator:
    def __init__(self, image, num_circular_points=None, radius=None, method=None):
        self.image = image
        self.num_circular_points = num_circular_points or 8
        self.radius = radius or 2
        self.method = method or 'uniform'

    def get_elbp_features(self):
        img = self.image
        some_grayscale_image = img.convert('L')
        lbp_image = local_binary_pattern(some_grayscale_image, self.num_circular_points, self.radius, method=self.method)
        histogram = list(stats.itemfreq(lbp_image).flatten())

        return histogram
