


import cv2
import numpy as np


def skewness(sub_image, means):
    channel_length = len(means)
    pixel_len, pixel_col, _ = sub_image.shape
    pixels_count = pixel_len * pixel_col
    skews = list()
    for i in range(0, channel_length):
        mean = means[i][0]
        all_values = list()
        for row in range(0, pixel_len):
            for col in range(0, pixel_col):
                all_values.append(sub_image[row][col][i])
        total = 0
        for val in all_values:
            total = total + (val - mean) * (val - mean) * (val - mean)
        skew = [(round(total)/pixels_count) ** (1. /3)]
        skews.append(skew)
    return skews


class ColorMomentsGenerator:
    def __init__(self, image_path, sub_image_args=None):
        self.image = cv2.imread(image_path)
        self.sub_image_args = sub_image_args or (8, 8)
        self.cell_row_size = self.sub_image_args[0]
        self.cell_column_size = self.sub_image_args[1]

    def get_color_moments(self):
        if self.image is None:
            return
        stats = np.array([])
        pixels_row_length, pixels_column_length, _ = self.image.shape
        for row in range(0, pixels_row_length, self.cell_row_size):
            for col in range(0, pixels_column_length, self.cell_column_size):
                sub_image = np.array(self.image[row:row + self.cell_row_size, col:col + self.cell_column_size, :], dtype='uint8')
                means, stds = cv2.meanStdDev(sub_image)
                skews = skewness(sub_image, means)
                stat = np.concatenate([means, stds, skews]).flatten()
                stats = np.concatenate([stats, stat]).flatten()

        return stats
