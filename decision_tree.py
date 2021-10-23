import numpy as np
from numpy.random import default_rng

from helper_functions.data_process import read_dataset, split_dataset
from helper_functions.purity import is_pure, classify
from helper_functions.splitting import find_splits, find_best_split, split_data
from helper_functions.tree_plotting import plot_tree
import helper_functions.evaluate as eval


def main():
    data = read_dataset("wifi_db/clean_dataset.txt")
    seed = 3
    rg = default_rng(seed)
    print(data.shape)
    data_10fold = split_dataset(data, 10, rg) # Split dataset into 10 random equally-sized subsets
    total_confusion = np.array((4, 4))

    print(data_10fold.shape)
    # Runs 10-fold cross validation
    k = 10
    for i in range(k):
        data_train = np.concatenate(data_10fold[np.arange(len(data_10fold))!=i])
        data_test = data_10fold[i]
        print(data_train.shape)
        (root, depth) = build_decision_tree(data_train)
        plot_tree(root, depth, "figures/fold" + str(i) + "_c.png") # This will changed later with wrapper function
        #confusion = confusion_matrix(root)
        y_gold = data_test[:,-1]
        y_predict = eval.predict(data_test[:, :-1])
        confusion_matrix = eval.gen_confusion_matrix(y_gold, y_predict)
        total_confusion += confusion_matrix
        
        

    




def build_decision_tree(data, depth=0):
    if is_pure(data):
        return (classify(data), depth)

    else:
        #potential_splits = find_splits(data)
        # split_feature refers to the column of the feature we split our data on, split_val refers to the value at which we seperate out entries on the split_feature
        split_feature, split_val = find_best_split(data)
        #, potential_splits)
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