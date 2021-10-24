import numpy as np

def find_splits(dataset):
    label_boundaries = {}

    for column in range(dataset.shape[1]-1):
        data_sorted_for_feature = dataset[:, [column, dataset.shape[1]]][dataset[:, column].argsort()]

        for row in (data_sorted_for_feature.shape[0]-1):

            if data_sorted_for_feature[row, 1] != data_sorted_for_feature[row+1, 1]:

                split_value = (data_sorted_for_feature[row, 1] + data_sorted_for_feature[row+1, 1])/2
                label_boundaries[column] = label_boundaries.get(column, []).append(split_value)        

    return label_boundaries

def split_data(dataset, split_feature_index, split_value):
    l_dataset = dataset[ dataset[:, split_feature_index] <= split_value ]
    r_dataset = dataset[ dataset[:, split_feature_index] > split_value ]
    #ideally, no datapoint should have a feature value equal to the split value, 
    #bar in the edge cases where a small number of datapoints of different classes have the same values 
    #or the optimal split is done on a feature where the border points have the same value
    #in the former, split arbitrarily below, in the latter, ignore until the classes are separated by another feature or the latter case emerges
    if r_dataset.size == 0:
        l_dataset, r_dataset = np.array_split(l_dataset, 2)

    return l_dataset, r_dataset

def find_best_split(dataset):

    label_boundaries = find_splits(dataset)

    max_remainder = 0
    optimal_split = {0, 0}

    #might be "simpler" to replace dict by list of tuples, bigger data structure but it seems odd to use a dictionary if only items are used
    for potential_split in label_boundaries.items():
        l_dataset, r_dataset = split_data(dataset, potential_split[0], potential_split[1])
        entropy_remainder = calculate_total_remainder(l_dataset, r_dataset)
        if entropy_remainder > max_remainder:
            max_remainder = entropy_remainder
            optimal_split = potential_split

    return split_data(dataset, potential_split[0], potential_split[1])