import numpy as np
from numpy.random import default_rng

# Calculate the entropy of a specific dataset
def calculate_entropy(data):
	labels = data[:,-1]
	_,freq = np.unique(labels, return_counts=True)

	prob = freq/freq.sum()
	entropy = sum(prob * -np.log2(prob))

	return entropy

# Calculate the entropy of the system after having been split at any given point
def calculate_total_entropy(l_data, r_data):
	l_entropy = calculate_entropy(l_data)
	r_entropy = calculate_entropy(r_data)

	remainder = (l_data.size*l_entropy)/(l_data.size+r_data.size) + (r_data.size*r_entropy)/(l_data.size+r_data.size)

	return remainder

