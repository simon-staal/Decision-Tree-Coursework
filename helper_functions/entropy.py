import numpy as np
from numpy.random import default_rng

# Calculate the entropy of a dataset using frequency of different labels and their sum (optional)
# If sum is not specified will calculate within function
def calculate_entropy(freq, sum=0):
	if sum == 0:
		sum = freq.sum()
	
	freq = freq[freq != 0]
	prob = freq/sum
	entropy = np.sum(prob * -np.log2(prob))
	return entropy

# Calculate the entropy of the system after having been split at any given point
# Takes in the frequency of the labels of both sets post split, as well as their sum (optional)
# If sum is not specified will calculate within function
def calculate_total_entropy(l_freq, r_freq, l_sum=0, r_sum=0):
	if l_sum == 0:
		l_sum = l_freq.sum()

	if r_sum == 0:
		r_sum = r_freq.sum()

	
	l_entropy = calculate_entropy(l_freq, l_sum)
	r_entropy = calculate_entropy(r_freq, r_sum)

	remainder = (l_sum*l_entropy)/(l_sum+r_sum) + (r_sum*r_entropy)/(l_sum+r_sum)

	return remainder

