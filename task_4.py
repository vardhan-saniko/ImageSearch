
import sys
import os

import itertools

from services.eucledean_distance import find_eucledean_distance
from services.manhattan_distance import find_manhattan_distance

import pickle

root_path, set_name, image_name, k, distance_metric = sys.argv[1:]
k = int(k)


def extract_k_similar_image_names():
    input_data_sets_set_dir_name = root_path + '/CSE-515/input_data_sets' + '/' + set_name

    list_of_files_except_query_image = set(os.listdir(input_data_sets_set_dir_name)) - { image_name }

    with open(root_path + '/CSE-515/feature_stores/' + 'features_by_set.txt') as f:
        features_data = f.readlines()

    distances_by_model = dict()
    for model in ['hog', 'elbp', 'cm']:

        get_features_data_of_set_by_model = eval(features_data[0])[set_name][model]

        ref_image_path = input_data_sets_set_dir_name + '/' + image_name
        ref_feature = pickle.loads(get_features_data_of_set_by_model[ref_image_path])
        distances_by_image = dict()
        for image in list_of_files_except_query_image:
            image_path = input_data_sets_set_dir_name + '/' + image
            if not '.png' in image_path:
                continue
            image_feature = pickle.loads(get_features_data_of_set_by_model[image_path])
            distances_by_image[image] = find_eucledean_distance(ref_feature, image_feature) if distance_metric == "euclidean" else find_manhattan_distance(ref_feature, image_feature)
        distances_by_model[model] = distances_by_image

    combine_distances = combine_distances_by_image(distances_by_model)
    resulted_dict = find_minimum(combine_distances, k)

    return resulted_dict


def combine_distances_by_image(data_by_model):
    image_paths = data_by_model['hog'].keys()
    final_data = dict()

    for image_path in image_paths:
        final_data[image_path] = 0.5 * data_by_model['hog'][image_path] + 0.25 * data_by_model['cm'][image_path] + 0.25 * data_by_model['elbp'][image_path]

    return final_data


def find_minimum(distances_dict, n):
    sorted_dict = {k: abs(v) for k, v in sorted(distances_dict.items(), key=lambda item: abs(item[1]))}
    return dict(itertools.islice(sorted_dict.items(), n))


print(extract_k_similar_image_names())


