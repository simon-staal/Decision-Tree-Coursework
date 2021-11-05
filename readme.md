Decision Tree ML Coursework
===========================
This repository contains the python implementation for a nested cross-fold validation decision tree with branch pruning produced as part of COMP70050 - Introduction to Machine Learning (Autmumn 2021). The specification for the coursework can be found [**here**](spec.pdf). Please find a brief guide to the repository below:

Contributors
------------
- Simon Staal (sts219)
- Petra Ratkai (petraratkai)
- Thomas Loureiro van Issum (tl319)
- David Cormier (DavidMael)

Overview
--------
Our decision tree can be run to generate pruned and un-pruned decision trees on both the clean and noisy data by calling `python3 decision_tree.py` (Note this implementation was built using python 3.8.10). The visualised decision trees can be found after execution in the [**figures**](figures/) directory. The report for the project can be found [**here**](report/report.pdf).

[**helper_functions**](helper_functions/)
-----
This directory contains all the helper functions used in our main [*decision_tree.py*](decision_tree.py) file, with functionality for data processing, finding splits, pruning, evaluating and plotting in seperate files. These functions are imported into our main file to be used by our build_decision_tree algorithm.

**Useful functions:**
- `train_test_k_folds(data, rg, k=10, file_suffix='c')` *Defined in [decision_tree.py](decision_tree.py), line 56* takes in a dataset, a random generator, k = number of folds, and a file suffix (used to differentiate between clean and noisy data figures). This functions performs trains a tree and performs simple cross-validation on k folds, returning a total confusion matrix for all k folds and an average tree depth.
- `train_test_nested_k_folds(data, rg, k=10, file_suffix='c')` *Defined in [decision_tree.py](decision_tree.py), line 84* behaves very similarly to the previous function, but performs nested cross-validation instead of simple cross validation, using a validation dataset to prune the trees generated. This function also returns the total confusion matrix for k folds and the average tree depth.
- `build_decision_tree(data, depth=0)` *Defined in [decision_tree.py](decision_tree.py), line 125* takes in a dataset (depth is used to track depth in recursion), and returns the root node of a decision tree and its total depth.
- `plot_tree(root, maxdepth, filename)` *Defined in [tree_plotting.py](helper_functions/tree_plotting.py), line 48* takes in a tree root node, the tree's maximum depth, and a filename. It plots the tree and saves it as filename passed to it as an input.
-`prune_tree(root, current_node, data_val, data_train, depth=0)` *Defined in [pruning.py](helper_functions/pruning.py), line 9* takes in a root node, a current node (which should be the same as the root node when called), a validation dataset, a training dataset and a depth (set to 0 by default, should be 0 when called). It returns the root of pruned version of the original tree, and the new depth.


[**wifi_db**](wifi_db/)
-----
This directory contains the raw data our decision tree was trained on, including both a clean and noisy version of the datasets.