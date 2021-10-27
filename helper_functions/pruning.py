from numbers import Number

from helper_functions.purity import classify
from helper_functions.splitting import split_data
from helper_functions.evaluate import predict, accuracy, gen_confusion_matrix

#The depth calculation and some other aspects are not very based, I'll fix them later
def prune_tree(root, current_node, val_dataset, train_dataset, depth):
    y_gold = val_dataset[:, -1] # Extract correct labels for validation data
    y_predict = predict(root, val_dataset[:, :-1]) # Slicing is unnecessary here but just to show we don't use the correct labels in our prediction (we don't look at them in our predict function anyways)
    ref_accuracy = accuracy(gen_confusion_matrix(y_gold, y_predict))

    #check if the root only has leaves, for safety
    if( isinstance(current_node["left"], Number) and isinstance(current_node["right"], Number) ):
        if(root == current_node ):
            return root, 0
        else:
            #for testing purposes
            print("this shouldn't happen, we should't recurse to leaf-only nodes")

    train_dataset_at_l, train_dataset_at_r = split_data(train_dataset, current_node["attribute"], current_node["value"])
    maj_class_l = classify(train_dataset_at_l)
    maj_class_r = classify(train_dataset_at_r)

    left_depth = 0
    right_depth = 0

    #Recursive calls. Since these take place before the pruning, newly prunable nodes will be examined in the next section
    if( not isinstance(current_node["left"], Number) ):
        if( not (isinstance(current_node["left"]["left"], Number) and isinstance(current_node["left"]["right"], Number) ) ):
            root, left_depth = prune_tree(root, current_node["left"], val_dataset, train_dataset_at_l, depth+1 )

    if( not isinstance(current_node["right"], Number) ):
        if( not (isinstance(current_node["right"]["left"], Number) and isinstance(current_node["right"]["right"], Number) ) ):
            root, right_depth = prune_tree(root, current_node["right"], val_dataset, train_dataset_at_r, depth+1 )

    max_depth = max(left_depth, right_depth, depth)
    #max_depth = max(max_depth, depth)

    #try to prune if a child node only has leaves
    if( not isinstance(current_node["left"], Number) ):
        if( isinstance(current_node["left"]["left"], Number) and isinstance(current_node["left"]["right"], Number) ):
            saved_prunee = dict(current_node["left"])
            #there should be no need to operate on a copy of the tree, and that would save quite a bit of complexity
            current_node["left"] = maj_class_l

            y_predict_left_leaf = predict(root, val_dataset[:, :-1]) # Slicing is unnecessary here but just to show we don't use the correct labels in our prediction (we don't look at them in our predict function anyways)
            left_leaf_accuracy = accuracy(gen_confusion_matrix(y_gold, y_predict_left_leaf))

            #if the accuracy is the same, we might as well keep the simpler tree
            if(left_leaf_accuracy < ref_accuracy):
                current_node["left"] = saved_prunee
            else:
                #?
                ref_accuracy = left_leaf_accuracy
                #also check the inverse scenario if this scenario is possible

    
    if( not isinstance(current_node["right"], Number) ):
        if( isinstance(current_node["right"]["left"], Number) and isinstance(current_node["right"]["right"], Number) ):
            saved_prunee = dict(current_node["right"])
            #there should be no need to operate on a copy of the tree, and that would save quite a bit of complexity
            current_node["right"] = maj_class_r

            y_predict_right_leaf = predict(root, val_dataset[:, :-1]) # Slicing is unnecessary here but just to show we don't use the correct labels in our prediction (we don't look at them in our predict function anyways)
            right_leaf_accuracy = accuracy(gen_confusion_matrix(y_gold, y_predict_right_leaf))

            if(right_leaf_accuracy < ref_accuracy):
                current_node["right"] = saved_prunee

    #shouldn't be necessary to return root
    return root, max_depth