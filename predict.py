import numpy as np


def predict(root, attributes):
    if isinstance(root, str):
        return root
    else:
        if attributes[root["attribute"]] < root["value"]:
            return predict(root['left'], attributes)
        else:
            return predict(root["right"], attributes)
    


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

attributes = [-50,1,2,3,4]
attributes2 = [-40,1,2,3,4]
attributes3 = [-60,1,2,-60,4]
attributes4 = [-60,1,2,-50,-60]
attributes5 = [-60,1,2,-50,-50]
print(predict(root, attributes5))
