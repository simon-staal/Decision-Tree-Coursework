import os
import numpy as np
from numpy.random import default_rng
import matplotlib as mpl

# Reads in dataset from specified filepath
def read_dataset(filepath):
    x = []
    y_labels = []
    for line in open(filepath):
        if line.strip() != "": # handle empty rows in file
            row = line.strip().split(",")
            x.append(list(map(float, row[:-1]))) 
            y_labels.append(row[-1])
    
    [classes, y] = np.unique(y_labels, return_inverse=True) 

    x = np.array(x) # Numpy array of shape (N, K), N = # of entries, K = # of features (dataset)
    y = np.array(y) # Numpy array of shape (N, ), integers from 0 to C-1 where C is the number of files
    return (x, y, classes)

# Return lists x_rand, y_rand which are randomly shuffled data-sets split into n subsets
def split_dataset(x, y, splits, random_generator=default_rng()):

    shuffled_indices = random_generator.permutation(len(x))
    
    x_rand = np.array_split(x[shuffled_indices], splits)
    y_rand = np.array_split(y[shuffled_indices], splits)

    return (x_rand, y_rand)