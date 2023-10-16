import numpy as np

def node_to_string(node, feature_names) -> str:
    """This is just a helper method to print the question in a readable format."""
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
    """
    Tries to partition the data with every unique value of each column. 
    Returns the column and value split with with highest information gain.
    """
    best_gain = 0
    best_feature = None
    best_value = None
    current_uncertainty = gini(y)
    number_features = X.shape[1]
    
    for feature in range(number_features):
        for value in np.unique(X[:, feature]):
            

            X_true, y_true, X_false, y_false = partition(X, y, feature, value)

            # Skips split if it doesn't divide the dataset.            
            if len(X_true) == 0 or len(X_false) == 0:
                continue
            
            # Calculates the information gain from split
            gain = information_gain(y_true, y_false, current_uncertainty)
            
            if gain > best_gain:
                best_gain, best_feature, best_value = gain, feature, value
    return best_gain, best_feature, best_value

