from numbers import Number

from helper_functions.purity import classify, is_pure
from helper_functions.splitting import split_data
from helper_functions.evaluate import predict, accuracy, gen_confusion_matrix

# Returns root and depth of a pruned tree based on an existing tree+depth and validation dataset
# data_val and data_train are the portions of the datasets that enter the given node during prediction
def prune_tree(root, current_node, data_val, data_train, depth):

    assert( data_val.size != 0 ), "We shouldn't recurse to a node that has no validation examples"

    y_gold = data_val[:, -1] # Extract correct labels for validation data

    y_predict = predict(current_node, data_val[:, :-1]) # Remove correct labels from dataset to show we're not cheating
    ref_accuracy = accuracy(gen_confusion_matrix(y_gold, y_predict))

    assert (not (isinstance(current_node["left"], Number) and isinstance(current_node["right"], Number))), "You cannot prune a node with leaf only children"

    data_train_l, data_train_r = split_data(data_train, current_node["attribute"], current_node["value"])
    maj_class_l = classify(data_train_l)
    maj_class_r = classify(data_train_r)

    data_val_l, data_val_r = split_data(data_val, current_node["attribute"], current_node["value"])

    # Prune any nodes which would split their validation input into empty and non empty sets, since it is impossible to evaluate a node receving an empty set
    # Nodes higher up the tree have more influence than those below, and nodes with a split value not included in their validation set have very poor performance. 
    # Therefore it is beneficial to prune these nodes along with their children, even if the latter have good performance. 
    # We have indeed found experimentally that pruning these nodes increases performance.
    if( not isinstance(current_node["left"], Number) ):
        data_val_l_l, data_val_l_r = split_data(data_val_l, current_node["left"]["attribute"], current_node["left"]["value"])
        if data_val_l_l.size == 0 or data_val_l_r.size == 0:
            current_node["left"] = maj_class_l

    if( not isinstance(current_node["right"], Number) ):
        data_val_r_l, data_val_r_r = split_data(data_val_r, current_node["right"]["attribute"], current_node["right"]["value"])
        if data_val_r_l.size == 0 or data_val_r_r.size == 0:
            current_node["right"] = maj_class_r

    left_depth = 0
    right_depth = 0

    # Recursive calls. Since these take place before the accuracy-based pruning, newly prunable nodes will be examined in the next section
    if( not isinstance(current_node["left"], Number) ):
        # Check the children of the left child are not leaves, if they are leaves don't recurse
        if( not (isinstance(current_node["left"]["left"], Number) and isinstance(current_node["left"]["right"], Number) ) ):
            root, left_depth = prune_tree(root, current_node["left"], data_val_l, data_train_l, depth+1 )
                

    # Same logic applies to right
    if( not isinstance(current_node["right"], Number) ):
        if( not (isinstance(current_node["right"]["left"], Number) and isinstance(current_node["right"]["right"], Number) ) ):
            root, right_depth = prune_tree(root, current_node["right"], data_val_r, data_train_r, depth+1 )

    max_depth = max(left_depth, right_depth, depth)

    # Try to prune if a child node only has leaves
    if( not isinstance(current_node["left"], Number) ):
        if( isinstance(current_node["left"]["left"], Number) and isinstance(current_node["left"]["right"], Number) ):
            saved_prunee = dict(current_node["left"])
            current_node["left"] = maj_class_l

            y_predict_leaf_l = predict(current_node, data_val[:, :-1])
            left_leaf_accuracy = accuracy(gen_confusion_matrix(y_gold, y_predict_leaf_l))

            if(left_leaf_accuracy < ref_accuracy):
                current_node["left"] = saved_prunee
            else:
                ref_accuracy = left_leaf_accuracy
    
    if( not isinstance(current_node["right"], Number) ):
        if( isinstance(current_node["right"]["left"], Number) and isinstance(current_node["right"]["right"], Number) ):
            saved_prunee = dict(current_node["right"])
            current_node["right"] = maj_class_r

            y_predict_leaf_r = predict(current_node, data_val[:, :-1]) # Slicing is unnecessary here but just to show we don't use the correct labels in our prediction (we don't look at them in our predict function anyways)
            right_leaf_accuracy = accuracy(gen_confusion_matrix(y_gold, y_predict_leaf_r))

            if(right_leaf_accuracy < ref_accuracy):
                current_node["right"] = saved_prunee

    return root, max_depth