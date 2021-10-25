import helper_functions.splitting as sf
import numpy as np

testData = np.array([[1, 2, 0], [2, 2, 0], [10, 11, 1], [12, 12, 1]])
print(testData.shape)

print("test 1 start")
print( sf.find_splits(testData) )
print("test 1 end")

x, y = sf.split_data(testData, 0, 5)
print("test 2 start")
print( x )
print( y )
print("test 2 end")

x, y = sf.find_best_split(testData)
print("test 3 start")
print( x )
print( y )
print("test 3 end")