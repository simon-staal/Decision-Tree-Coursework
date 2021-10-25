import numpy as np

def is_pure(data): #returns whether all the instances in data have the same label
    #data: an np array representing the entire set of data, the last colomn of the data is the labels
    return np.all(data[:,-1] == data[0][-1])


'''array = np.zeros((10,10))
array[4][9]+=5
array[7][9]+=2
array[0][9]+=2
print(array)
print(is_pure(array))'''

def classify(data): #returns the most frequently appearing labels from data
    labels, counts = np.unique(data[:,-1], return_counts = True)
    return str(labels[np.argmax(counts)])

'''array = np.ones((10,10))
array[3][9] =2
array[4][9]=2
array[5][9] = 2
array[6][9] =4
array[7][9]=4
array[8][9]=4
array[9][9] = 4
#array[4][9]+=5
#array[7][9]+=2
#array[0][9]+=2
print(array)
print(classify(array))'''