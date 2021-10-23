import numpy as np
from numpy.random import default_rng

from data_process import read_dataset, split_dataset
from purity import is_pure, classify
from splitting import find_splits, find_best_split, split_data
from tree_plotting import plot_tree


def main():
    data = read_dataset("wifi_db/clean_dataset.txt")
    seed = 3
    rg = default_rng(seed)
    data_10fold = split_dataset(data, 10, rg) # Split dataset into 10 random equally-sized subsets
    total_confusion = np.array(4, 4)
    total_accuracy = 0
    total_metrics = np.array(4, 3)


    # Runs 10-fold cross validation
    for i in range(10):
        data_train = data_10fold[np.arange(len(data_10fold))!=i]
        data_test = data_10fold[i]
        (root, depth) = build_decision_tree(data_train)
        plot_tree(root) # This will changed later with wrapper function
        y_gold = data_test[:,-1]
        y_predict = np.apply_along_axis(predict, 1, data_test[:,:-1]) # Using apply_along_axis is untested
        confusion = confusion_matrix(y_gold, y_predict)
        




def build_decision_tree(data, depth=0):
    if is_pure(data):
        return (classify(data), depth)

    else:
        potential_splits = find_splits(data)
        # split_feature refers to the column of the feature we split our data on, split_val refers to the value at which we seperate out entries on the split_feature
        split_feature, split_val = find_best_split(data, potential_splits)
        l_dataset, r_dataset = split_data(split_feature, split_val)
        l_node, l_depth = build_decision_tree(l_dataset, depth+1)
        r_node, r_depth = build_decision_tree(r_dataset, depth+1)
        node = {
            'attribute': split_feature, 
            'value': split_val,
            'left': l_node,
            'right': r_node
        }
        return (node, max(l_depth, r_depth))



if __name__ == "__main__":
    main()