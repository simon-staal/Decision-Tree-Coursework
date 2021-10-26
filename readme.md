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
Our decision tree can be run to generate pruned and un-pruned decision trees on both the clean and noisy data by calling `python3 decision_tree.py` (Note this implementation was built using python 3.8.10). The visualised decision trees can be found after execution in the [**figures**](figures/) directory.

[**helper_functions**](helper_functions/)
-----
This directory contains all the helper functions used in our main [*decision_tree.py*](decision_tree.py) file, with functionality for data processing, finding splits, pruning, evaluating and plotting in seperate files. These functions are imported into our main file to be used by our build_decision_tree algorithm.

[**wifi_db**](wifi_db/)
-----
This directory contains the raw data our decision tree was trained on, including both a clean and noisy version of the datasets.