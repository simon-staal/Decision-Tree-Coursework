import numpy as np

# Predicts label for single entry
def predict_single(attributes, root):
    # Labels are stored as floats => leaf node
    if isinstance(root, float):
        return root
    else: # We are not at a leaf node
        if attributes[int(root["attribute"])] < root["value"]:
            return predict_single(attributes, root['left'])
        else:
            return predict_single(attributes, root["right"])

# Predicts labels for data-set
def predict(root, data_test):
    return np.apply_along_axis(predict_single, 1, data_test, root=root)

# Return confusion matrix based on gold standard vs predictions
def gen_confusion_matrix(y_gold, y_predict):
    # y_gold and y_predict contain the labels stored as floats
    # In this case => {1. 2. 3. 4.}
    class_labels = np.unique(np.concatenate((y_gold, y_predict)))
    confusion_matrix = np.zeros((len(class_labels), len(class_labels)), dtype=np.int)

    assert (len(y_gold) == len(y_predict)),"Mismatched prediction / gold standard results"
    for i in range(len(y_gold)):
        # We map the floats to indices in our confusion matrix by converting to int and subtracting 1
        confusion_matrix[int(y_gold[i])-1][int(y_predict[i])-1] += 1
    
    return confusion_matrix

# Compute accuracy based on confusion matrix
def accuracy(confusion_matrix):
    acc = 0
    if confusion_matrix.sum() > 0:
        acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
    return acc

# Compute per-class and macro-averaged precision based on confusion matrix
def precision(confusion_matrix):
    # Compute the precision per class
    i = confusion_matrix.sum(axis=0) != 0 # Avoid div by 0 error
    p = np.zeros((len(confusion_matrix), ))
    p[i] = np.diag(confusion_matrix)[i]/confusion_matrix.sum(axis=0)[i]
    
    # Compute the macro-averaged precision
    macro_p = 0
    if len(p) != 0:
      macro_p = p.mean()

    return (p, macro_p)

# Compute per-class and macro-averaged recall based on confusion matrix
def recall(confusion_matrix):
    # Compute the recall per class
    i = confusion_matrix.sum(axis=1) != 0 # Avoid div by 0 error
    r = np.zeros((len(confusion_matrix), ))
    r[i] = np.diag(confusion_matrix)[i] / confusion_matrix.sum(axis=1)[i]
    
    # Compute the macro-averaged recall
    macro_r = 0
    if len(r) != 0:
      macro_r = r.mean()
    
    return (r, macro_r)

# Compute per-class and macro-averaged f1 score based on confusion matrix
# NB: Macro-averaged F1 score is defined as the mean of F1 scores across the classes
def f1_score(confusion_matrix):
    (p, _) = precision(confusion_matrix) # Obtain per-class precision
    (r, _) = recall(confusion_matrix) # Obtain per-class recall

    # Make sure recall and precision are of the same length
    assert (len(p) == len(r)), "Mismatched length of precision and recall"

    # Compute the per-class F1
    f = np.zeros((len(p), ))
    i = p + r != 0 # Avoid div by 0 errors
    f[i] = 2*p[i]*r[i]/(p[i]+r[i])

    # Compute the macro-averaged F1
    macro_f = 0
    if len(f) != 0:
      macro_f = f.mean()
    
    return (f, macro_f)

# Prints all the relevant metrics for a given confusion matrix + number of folds
def print_metrics(confusion_matrix, k):
    print("Average confusion matrix:")
    print(confusion_matrix/k)

    print("Accuracy:", accuracy(confusion_matrix))

    (p, macro_p) = precision(confusion_matrix)
    print("Precision per-class:", p)
    print("Macro-averged precision", macro_p)

    (r, macro_r) = recall(confusion_matrix)
    print("Recall per-class:", r)
    print("Macro-averaged recall:", macro_r)

    (f, f_macro) = f1_score(confusion_matrix)
    print("F1 score per-class", f)
    print("Macro-averaged F1 score", f_macro)