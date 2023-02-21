

def find_eucledean_distance(ref_feature, input_feature):

    if len(ref_feature) != len(input_feature):
        raise Exception("length not matching")

    sum_of_squares = 0
    for index, value in enumerate(ref_feature):
        sum_of_squares += (value - input_feature[index]) ** 2

    return sum_of_squares ** (1. / 2)
