import numpy as np

# Returns whether all the instances in data have the same label
# Data: an np array representing the entire set of data, the last colomn of the data is the labels
def is_pure(data): 
    return np.all(data[:,-1] == data[0][-1])

# Returns the most frequently appearing labels from data
def classify(data):
    labels, counts = np.unique(data[:,-1], return_counts = True)
    return labels[np.argmax(counts)]
