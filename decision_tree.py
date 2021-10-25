import numpy as np
from numpy.random import default_rng

import data_process as dp

def main():
    data = dp.read_dataset("wifi_db/clean_dataset.txt")
    seed = 3
    rg = default_rng(seed)
    data_10fold = dp.split_dataset(data, 10, rg) # Split dataset into 10 random equally-sized subsets

    # Runs 10-fold cross validation
    for i in range(10):
        data_train = data_10fold[np.arange(len(data_10fold))!=i]
        data_test = data_10fold[i]
        decision_tree = build_decision_tree(data_train)



def build_decision_tree(data, depth=0):
    if is_pure(data):
        return classify(data)

    else:
        potential_splits = find_splits(data) # Maybe add this directly in subsequent function
        split_col, split_val = find_best_split(data, potential_splits)



if __name__ == "__main__":
    main()