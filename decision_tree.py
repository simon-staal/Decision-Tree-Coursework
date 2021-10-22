import os
import numpy as np
from numpy.random import default_rng
import matplotlib as mpl

def main():
    (x, y, classes) = read_dataset("wifi_db/clean_dataset.txt")
    seed = 3
    rg = default_rng(seed)
    (x_10fold, y_10fold) = split_dataset(x, y, 10, rg) # Split dataset into 10 random equally-sized subsets

    # Runs 10-fold cross validation
    for i in range(10):
        x_train = x_10fold[np.arange(len(x_10fold))!=i]
        x_test = x_10fold[i]
        y_train = y_10fold[np.arange(len(y_10fold))!=i]
        y_test = y_10fold[i]

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

# Return np arrays x_rand, y_rand which are randomly shuffled data-sets split into n subsets
def split_dataset(x, y, splits, random_generator=default_rng()):

    shuffled_indices = random_generator.permutation(len(x))
    
    x_rand = np.asarray(np.array_split(x[shuffled_indices], splits))
    y_rand = np.asarray(np.array_split(y[shuffled_indices], splits))

    return (x_rand, y_rand)





if __name__ == "__main__":
    main()