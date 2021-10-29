import os
import numpy as np
from numpy.random import default_rng

# Reads in dataset from specified filepath
# Stores x-data and corresponding label in same numpy array
def read_dataset(filepath):
    data = []
    for line in open(filepath):
        if line.strip() != "": # Handle empty rows in file
            row = line.strip().split()
            assert (len(row) == 8 ), f'Expected 8 elements, instead read {len(row)}'
            entry = list(map(float, row[:])) # Label is represented as float in noisy data, therefore we use float for labels
            data.append(entry)

    data = np.array(data)
    return data

# Return np array with randomly shuffled data-sets split into n subsets
def split_dataset(data, splits, random_generator=default_rng()):
    shuffled_indices = random_generator.permutation(len(data))
    data_rand = np.asarray(np.array_split(data[shuffled_indices], splits))
    return data_rand