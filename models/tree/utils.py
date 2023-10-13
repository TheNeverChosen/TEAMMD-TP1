import numpy as np

def node_to_string(node, feature_names) -> str:
    # This is just a helper method to print
    # the question in a readable format.
    condition = "=="
    if node.value != str:
        condition = ">="
    return f"Is {feature_names[node.feature]} {condition} {str(node.value)}?"

def gini(array):
    _, counts = np.unique(array, return_counts=True)
    impurity = 1
    
    value_probabilities = counts / len(array)
    impurity = 1 - np.square(value_probabilities).sum()
    return impurity

def information_gain(left, right, current_uncertainty):
    """Information Gain.

    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)

def partition(X, y, feature, value):
    if type(value) != str:
        true_partition = (X[:, feature] > value)
    else:
        true_partition = (X[:, feature] == value)
    
    true_partition = true_partition.flatten()    

    X_true = X[true_partition]
    y_true = y[true_partition]
    X_false = X[~true_partition]
    y_false = y[~true_partition]
    
    return X_true, y_true, X_false, y_false

def find_best_split(X, y):
    best_gain = 0  # keep track of the best information gain
    best_feature = None  # keep train of the feature / value that produced it
    best_value = None
    current_uncertainty = gini(y)
    number_features = X.shape[1]
    
    for feature in range(number_features):
        for value in np.unique(X[:, feature]):
            
            # Skip this split if it doesn't divide the
            # dataset.
            X_true, y_true, X_false, y_false = partition(X, y, feature, value)
            
            if len(X_true) == 0 or len(X_false) == 0:
                continue
            
            # Calculate the information gain from this split
            gain = information_gain(y_true, y_false, current_uncertainty)
            
            if gain > best_gain:
                best_gain, best_feature, best_value = gain, feature, value
    return best_gain, best_feature, best_value

