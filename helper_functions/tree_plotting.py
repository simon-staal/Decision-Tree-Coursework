import matplotlib 
import matplotlib.pyplot as plt
import numpy as np

# Plots the tree recursively using Matplotlib
# Parameters: root - current node, depth - current depth the function is at, x,x- coordinates of the current node, 
#           yspacing - height of each level in the tree
# In order to call this function, the matplotlib figure must be already initiated
def plot_tree_helper(root, depth=0, x=0, y=50, yspacing=5):
    # Check if the current node is a leaf (leaves represented as floats which are the room numbers)
    if isinstance(root, float):
        # Draw leaf node:
        plt.text(x, y, str(int(root)), size='smaller', rotation=0,
         ha="center", va="center",
         bbox=dict(boxstyle="round",
                   ec=(0., 0., 0.),
                   fc=(1., 1., 1.),
                   )
         )
        return
    # If it is not a leaf node, we draw a regular node (x < value format)
    plt.text(x, y, str("x" + str(root['attribute'])) + " < " + str(root['value']), size='smaller', rotation=0,
         ha="center", va="center",
         bbox=dict(boxstyle="round",
                   ec=(0., 0., 0.),
                   fc=(1., 1., 1.),
                   )
         )
    # Then have to draw left and right line to the children, so calculate the coordinates of the children
    # The change in y coordinate is is going to be proportional to 1/power(2,depth), since at every level there are at most 2^depth nodes,
    #           so we need to split the width of the figure up into 2^depth equal parts
    yChild = y - yspacing # Children are going to be on the same level, so y coordinate is the same
    xChildLeft = x - 2*1/np.power(2, depth)
    xChildRight = x + 2*1/np.power(2, depth)
    # Plotting the left and right lines as one dataset so that they get the same color
    x_values = [xChildLeft, x, xChildRight]
    y_values = [yChild, y, yChild]
    plt.plot(x_values, y_values)
    # Call the function again on the children so that children are plotted as well
    plot_tree_helper(root['left'], depth+1, xChildLeft, yChild, yspacing)
    plot_tree_helper(root['right'], depth+1, xChildRight, yChild, yspacing)
    return

# Wrapper for the recursive plot_tree_helper function
# Initialises the matplotlib figure, calls the recursive function then closes the figure and saves the file
# Parameters: root of the tree, the maximal depth the tree could achieve, and the filename the figure must be saved as
def plot_tree(root, maxdepth, filename) :
    matplotlib.use('Agg') # Set the backend to Agg, this is to make sure the code works with the lab machines
    plt.figure(figsize=(min(2**maxdepth, 2**5), maxdepth), dpi=80) # Figure size cannot be too big so we cap it at 2^7
    plot_tree_helper(root) # Call the recursive function
    plt.axis('off') # Remove the axes
    plt.savefig(filename) 
    plt.close()
    return

