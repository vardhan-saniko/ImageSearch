

def find_manhattan_distance(ref_feature, input_feature):

    if len(ref_feature) != len(input_feature):
        raise Exception("length not matching")

    sum = 0
    for index, value in enumerate(ref_feature):
        sum += abs((value - input_feature[index]))

    return sum
