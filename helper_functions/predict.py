import numpy as np


# Predicts label for single entry
def predict_single(attributes, root):
    if isinstance(root, str):
        return root
    else:
        if attributes[root["attribute"]] < root["value"]:
            return predict_single(attributes, root['left'])
        else:
            return predict_single(attributes, root["right"])

# Predicts labels for data-set
def predict(root, data_test):
    return np.apply_along_axis(predict_single, 1, data_test, root=root)
    


child6 = {
    "attribute": 3,
}
child5 = {
    "attribute": 1
}
child4 = {
    "value": -57,
    "attribute": 4,
    "left": "1",
    "right": "2"
}
child3 = {
    "attribute": 0
}
child2 = {
    "attribute": 0,
    "value": -44.5,
    "left": "1",
    "right": '2'
}
child1 = {
    "value": -56.5,
    "attribute": 3,
    "left": "3",
    "right": child4,
}
root = {
    "value": -55.5,
    "attribute": 0,
    "left":child1,
    "right": child2,
}

test_data = [[-50,1,2,3,4],
                [-40,1,2,3,4],
                [-60,1,2,-60,4],
                [-60,1,2,-50,-60],
                [-60,1,2,-50,-50]]
test_data = np.asarray(test_data)

print(predict(root, test_data))
