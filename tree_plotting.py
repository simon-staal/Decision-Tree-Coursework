import matplotlib.pyplot as plt
import numpy as np

def plot_tree(root, depth, x, y, yspacing):
    if isinstance(root, str):
        #draw leaf node:
        plt.text(x, y, "leaf: " + root, size='smaller', rotation=0,
         ha="center", va="center",
         bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   )
         )
        return
    #draw current node
    plt.text(x, y, str(root['attribute']) + " < " + str(root['value']), size='smaller', rotation=0,
         ha="center", va="center",
         bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   )
         )
    #have to draw left line and right line, so calc xchildren, ychildren
    yChildLeft = y - yspacing
    yChildRight = y - yspacing
    xChildLeft = x - 1/np.power(2, depth)
    xChildRight = x + 1/np.power(2, depth)
    x_values = [xChildLeft, x, xChildRight]
    y_values = [yChildLeft, y, yChildRight]
    plt.plot(x_values, y_values)
    #plt.plot([x, xChildRight], [y, yChildRight])
    plot_tree(root['left'], depth+1, xChildLeft, yChildLeft, yspacing)
    plot_tree(root['right'], depth+1, xChildRight, yChildRight, yspacing)
    return




child6 = {
    "attribute": 3,
}
child5 = {
    "attribute": 1
}
child4 = {
    "value": -57,
    "attribute": "x4",
    "left": "1",
    "right": "2"
}
child3 = {
    "attribute": 0
}
child2 = {
    "attribute": "x0",
    "value": -44.5,
    "left": "1",
    "right": '2'
}
child1 = {
    "value": -56.5,
    "attribute": "x3",
    "left": "3",
    "right": child4,
}
root = {
    "value": -55.5,
    "attribute": "x0",
    "left":child1,
    "right": child2,
}

from matplotlib.pyplot import figure

#figure(figsize=(15, 5), dpi=80)
plot_tree(root, 1, 0,50, 5)
#plt.xlim([-3,3])
plt.axis('off')
plt.savefig("matplotlib.png")