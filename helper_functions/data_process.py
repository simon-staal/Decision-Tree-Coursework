import os
import numpy as np
from numpy.random import default_rng

# Reads in dataset from specified filepath
def read_dataset(filepath):
    data = []
    for line in open(filepath):
        if line.strip() != "": # handle empty rows in file
            row = line.strip().split()
            assert (len(row) == 8 ), "Bad"
            entry = list(map(float, row[:]))
            data.append(entry)

    data = np.array(data)
    return data

# Return np array with randomly shuffled data-sets split into n subsets
def split_dataset(data, splits, random_generator=default_rng()):
    shuffled_indices = random_generator.permutation(len(data))
    data_rand = np.asarray(np.array_split(data[shuffled_indices], splits))
    return data_rand