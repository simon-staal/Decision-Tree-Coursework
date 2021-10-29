import numpy as np
import os
from numpy.random import default_rng

from helper_functions.data_process import read_dataset, split_dataset
from helper_functions.purity import is_pure, classify
from helper_functions.splitting import find_best_split, split_data
from helper_functions.tree_plotting import plot_tree
import helper_functions.evaluate as eval
from helper_functions.pruning import prune_tree


def main():
    # Reads in dataset from specified filepath
    data = read_dataset("wifi_db/clean_dataset.txt")
    seed = 3
    rg = default_rng(seed)
    k = 10

    # Trains on clean dataset, generates confusion matrix and prints relevant performance metrics.
    (total_confusion, avg_depth) = train_test_k_folds(data, rg, k, "c")
    print("Clean Dataset un-pruned metrics: ")
    eval.print_metrics(total_confusion, k)
    print("Average un-pruned depth:", avg_depth)
    
    # Trains on specified dataset, prunes using validation set, generates confusion matrix and prints relevant performance metrics.
    (total_confusion, avg_depth) = train_test_nested_k_folds(data, rg, k, "c")
    print("Clean Dataset pruned metrics: ")
    eval.print_metrics(total_confusion, k)
    print("Average pruned depth:", avg_depth)

    # Reads in dataset from specified filepath
    data = read_dataset("wifi_db/noisy_dataset.txt")
    seed = 13
    rg = default_rng(seed)
    
    # Trains on noisy dataset, generates confusion matrix and prints relevant performance metrics.
    (total_confusion, avg_depth) = train_test_k_folds(data, rg, k, "n")
    print("Noisy Dataset un-pruned metrics: ")
    eval.print_metrics(total_confusion, k)
    print("Average un-pruned depth:", avg_depth)

    # Trains on noisy dataset, prunes using validation set, generates confusion matrix and prints relevant performance metrics.    
    (total_confusion, avg_depth) = train_test_nested_k_folds(data, rg, k, "n")
    print("Noisy Dataset pruned metrics: ")
    eval.print_metrics(total_confusion, k)
    print("Average pruned depth:", avg_depth)

    return
        

# Returns total confusion matrix for all folds and average tree depth
def train_test_k_folds(data, rg, k=10, file_suffix="c"):
    assert (k > 1), "Invalid folds parameter"
    data_10fold = split_dataset(data, k, rg) # Split dataset into 10 random equally-sized subsets
    total_confusion = np.zeros((4, 4))
    depths = np.zeros((k, )) # Stores depth of each tree generated

    # Folder used for figure storing
    out_dir = ("figures")
    # If folder doesn't exist, then create it.
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Runs 10-fold cross validation
    for i in range(k):
        data_train = np.concatenate(data_10fold[np.arange(len(data_10fold))!=i])
        data_test = data_10fold[i]
        (root, depth) = build_decision_tree(data_train)
        plot_tree(root, depth, "figures/fold" + str(i) + "_" + file_suffix + ".svg")
        y_gold = data_test[:,-1]
        y_predict = eval.predict(root, data_test[:, :-1]) # Remove correct labels from dataset to show we're not cheating
        confusion_matrix = eval.gen_confusion_matrix(y_gold, y_predict)
        total_confusion += confusion_matrix
        depths[i] = depth
    
    return (total_confusion, depths.mean())


# Returns total confusion matrix for all (outer) folds and average tree depth
def train_test_nested_k_folds(data, rg, k=10, file_suffix="c"):
    assert (k > 1), "Invalid folds parameter"
    data_10fold = split_dataset(data, k, rg) # Split dataset into 10 random equally-sized subsets
    total_confusion = np.zeros((4, 4))
    depths = np.zeros((k, )) # Stores depth of each tree generated

    # Folder used for figure storing
    out_dir = ("figures")
    # If folder doesn't exist, then create it.
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Runs 10-fold cross validation
    for i in range(k):
        data_train_outer = data_10fold[np.arange(len(data_10fold))!=i]
        data_test = data_10fold[i]
        # Nested folds for pruning
        pruned_accuracies = [] # Tracks the pruned trees created and their accuracy + depth
        for j in range(k-1):
            data_val = data_train_outer[j] 
            data_train = np.concatenate(data_train_outer[np.arange(len(data_train_outer))!=i])
            (root, _) = build_decision_tree(data_train)
            (root_pruned, depth) = prune_tree(root, root, data_val, data_train, 0)
            y_gold = data_val[:,-1]
            y_predict = eval.predict(root_pruned, data_test[:, :-1]) # Remove correct labels from dataset to show we're not cheating
            acc = eval.accuracy(eval.gen_confusion_matrix(y_gold, y_predict))
            pruned_accuracies.append((acc, root_pruned, depth))

        (_, root, depth) = max(pruned_accuracies, key=lambda x:x[0]) # Chooses the tree with the highest accuracy

        plot_tree(root, depth, "figures/pruned_fold" + str(i) + "_" + file_suffix + ".svg")
        y_gold = data_test[:,-1]
        y_predict = eval.predict(root, data_test[:, :-1])
        confusion_matrix = eval.gen_confusion_matrix(y_gold, y_predict)
        total_confusion += confusion_matrix
        depths[i] = depth
    
    return (total_confusion, depths.mean())


#Recursively builds the decision based on the given dataset and the calculated optimal splits.
def build_decision_tree(data, depth=0):
    if is_pure(data): # If the data only contains 1 label, return class label
        return (classify(data), depth) # Classify returns majority class label

    else:
        # split_feature refers to the column we split our data on
        # split_val refers to the value at which we seperate out entries on the split_feature
        split_feature, split_val = find_best_split(data)
        l_dataset, r_dataset = split_data(data, split_feature, split_val)
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