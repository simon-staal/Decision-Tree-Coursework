import numpy as np

# Predicts label for single entry
def predict(root, attributes):
    if isinstance(root, str):
        return root
    else:
        if attributes[root["attribute"]] < root["value"]:
            return predict(root['left'], attributes)
        else:
            return predict(root["right"], attributes)

# 
def confusion_matrix(y_gold, y_predict)