import numpy as np
from helper_functions.entropy import calculate_total_entropy

# Returns the best split feature and split value for a dataset
def find_best_split(data):
    best_split = (None, None, 2) # Keeps track of the best split (split_feature, split value, min_entropy)
    labels, total_freq = np.unique(data[:,-1], return_counts=True) # Used calculate remaining frequencies
    total_sum = total_freq.sum()

    # Sorts dataset by each feature to find class borders in each sorted version
    for i in range(data.shape[1]-1):
        running_freq = np.zeros((len(labels),), dtype=int) # Stores running frequencies
        running_sum = 0

        sorted_data = data[:, [i, data.shape[1]-1]][data[:, i].argsort()] # Contains 1 feature and label

        for j in range(sorted_data.shape[0]-1):
            # Increments running frequencies
            running_freq[np.where(labels == sorted_data[j, 1])] += 1 
            running_sum += 1

            # Check if feature values are different
            # If feature values are the same we can't split between these two values
            if sorted_data[j, 0] != sorted_data[j+1, 0]:
                # The reason we pass the sum is to avoid the O(n) cost of computing it, we can track the running sum with O(1)
                current_entropy = calculate_total_entropy(running_freq, total_freq-running_freq, running_sum, total_sum-running_sum)
                # If the current entropy is < the current minimum entropy, we maximise information gain
                # Therefor there is no need to compute the information gain
                if current_entropy < best_split[2]:
                    split_value = (sorted_data[j, 0] + sorted_data[j+1, 0])/2
                    best_split = (i, split_value, current_entropy)     

    assert (best_split[0] != None), "No split found"

    return (best_split[0], best_split[1])

# Returns 2 data sets that have been split on a specified feature and value
def split_data(dataset, split_feature, split_value):
    # We should never have values in the dataset = split_value because we only consider splits where the adjecent values are different
    l_dataset = dataset[ dataset[:, split_feature] < split_value ]
    r_dataset = dataset[ dataset[:, split_feature] > split_value ]


    return l_dataset, r_dataset
