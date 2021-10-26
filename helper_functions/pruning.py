from numbers import Number

from helper_functions.purity import classify
from helper_functions.splitting import split_data
from helper_functions.evaluate import predict, accuracy, gen_confusion_matrix

def prune_tree(root, current_node, val_dataset, train_dataset):
    y_gold = val_dataset[:, -1] # Extract correct labels for validation data
    y_predict = predict(root, val_dataset[:, :-1]) # Slicing is unnecessary here but just to show we don't use the correct labels in our prediction (we don't look at them in our predict function anyways)
    ref_accuracy = accuracy(gen_confusion_matrix(y_gold, y_predict))

    #check if the root only has leaves, for safety
    if( isinstance(current_node["left"], Number) and isinstance(current_node["right"], Number) ):
        if(root == current_node ):
            return
        else:
            print("this shouldn't happen, we should't recurse to leaf-only nodes")

    train_dataset_at_l, train_dataset_at_r = split_data(train_dataset, current_node["attribute"], current_node["value"])
    maj_class_l = classify(train_dataset_at_l)
    maj_class_r = classify(train_dataset_at_r)

    #Recursive calls. Since these take place before the pruning, newly prunable nodes will be examined in the next section
    if( not isinstance(current_node["left"], Number) ):
        if( not (isinstance(current_node["left"]["left"], Number) and isinstance(current_node["left"]["right"], Number) ) ):
            prune_tree(root, current_node["left"], val_dataset, train_dataset_at_l )

    if( not isinstance(current_node["right"], Number) ):
        if( not (isinstance(current_node["right"]["left"], Number) and isinstance(current_node["right"]["right"], Number) ) ):
            prune_tree(root, current_node["right"], val_dataset, train_dataset_at_r )

    #try to prune if a child node only has leaves
    if( not isinstance(current_node["left"], Number) ):
        if( isinstance(current_node["left"]["left"], Number) and isinstance(current_node["left"]["right"], Number) ):
            saved_prunee = dict(current_node["left"])
            #there should be no need to operate on a copy of the tree, and that would save quite a bit of complexity
            current_node["left"] = maj_class_l
            left_leaf_accuracy = accuracy(root, val_dataset)

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
            right_leaf_accuracy = accuracy(root, val_dataset)

            if(right_leaf_accuracy < ref_accuracy):
                current_node["right"] = saved_prunee