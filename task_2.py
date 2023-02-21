import os
import sys

from collections import defaultdict
from services.hog import HogGenerator
from services.elbp import ELBPGenerator
from services.color_moments import ColorMomentsGenerator
from skimage.io import imread
from PIL import Image
import numpy as np
import pickle

final_features_data = defaultdict(dict)


def get_hog_features(list_of_image_paths):
    """
        HOG Features
    """
    hog_outputs_by_file_path = dict()
    for image_path in list_of_image_paths:
        if ".png" not in image_path:
            continue
        img = imread(image_path)
        hog_outputs_by_file_path[image_path] = pickle.dumps(HogGenerator(image=img).hog())

    return {'hog': hog_outputs_by_file_path}


def get_elbp_features(list_of_image_paths):
    """
        ELBP Features
    """
    elbp_outputs_by_file_path = dict()
    for image_path in list_of_image_paths:
        if not ".png" in image_path:
            continue
        img = imread(image_path)
        image = Image.fromarray(np.uint8(img), 'L')
        # print("Image: {} ------- value: {}".format(image, ELBPGenerator(image=image).get_elbp_features()))
        elbp_outputs_by_file_path[image_path] = pickle.dumps(ELBPGenerator(image=image).get_elbp_features())

    return {'elbp': elbp_outputs_by_file_path}


def get_color_moment_features(list_of_image_paths):
    """
        Color Moment Features
    """
    color_moment_outputs_by_file_path = dict()
    for image_path in list_of_image_paths:
        color_moment_outputs_by_file_path[image_path] = pickle.dumps(ColorMomentsGenerator(image_path).get_color_moments())

    return {'cm': color_moment_outputs_by_file_path}


def extract_all_features_for_all_images(root_path):
    input_data_sets_dir_name = root_path + '/CSE-515/input_data_sets'
    input_sets = [set_name for set_name in os.listdir(input_data_sets_dir_name) if os.path.isdir(input_data_sets_dir_name + '/' + set_name)]

    for set_name in input_sets:
        list_of_image_paths_by_set = list()
        for file_name in os.listdir(input_data_sets_dir_name + '/' + set_name):
            list_of_image_paths_by_set.append(input_data_sets_dir_name + '/' + set_name + '/' + file_name)
        final_features_data[set_name].update(get_color_moment_features(list_of_image_paths_by_set))
        final_features_data[set_name].update(get_elbp_features(list_of_image_paths_by_set))
        final_features_data[set_name].update(get_hog_features(list_of_image_paths_by_set))

    features_storage_path = root_path + '/CSE-515/feature_stores/' + 'features_by_set.txt'

    with open(features_storage_path, 'w') as f:
        f.write(str(dict(final_features_data)))


root_path = sys.argv[1]

extract_all_features_for_all_images(root_path)

