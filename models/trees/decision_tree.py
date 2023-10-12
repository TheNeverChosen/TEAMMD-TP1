import numpy as np
import pandas as pd
from utils import find_best_split, partition, node_to_string
from base import DecisionNode, Leaf
   
def build_tree(X, y):
    """Builds the tree.

    Rules of recursion: 1) Believe that it works. 2) Start by checking
    for the base case (no further information gain). 3) Prepare for
    giant stack traces.
    """

    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.
    gain, feature, value = find_best_split(X, y)

    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    if gain == 0:
        return Leaf(y)

    # If we reach here, we have found a useful feature / value
    # to partition on.
    X_true, y_true, X_false, y_false = partition(X, y, feature, value)

    # Recursively build the true branch.
    true_branch = build_tree(X_true, y_true)

    # Recursively build the false branch.
    false_branch = build_tree(X_false, y_false)

    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # dependingo on the answer.
    return DecisionNode(feature, value, true_branch, false_branch)

def predict_example(example, node):
    """See the 'rules of recursion' above."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.predictions

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.compare(example):
        return predict_example(example, node.true_branch)
    else:
        return predict_example(example, node.false_branch)

def print_tree(node, feature_names, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print (spacing + node_to_string(node, feature_names))

    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")

class DecisionTree():
    def __init__(self) -> None:
        self.tree = None
        self.feature_names = None
    
    def fit(self, X, y):
        if type(X) == pd.DataFrame:
            self.feature_names = X.columns
        else:
            self.feature_names = range(X.shape[1])
        self.tree = build_tree(X,y)
    
    def predict(self, X: np.ndarray):    
        y_pred = np.empty(len(X), dtype=np.object_)
        for i, example in enumerate(X):
            predictions = predict_example(example, self.tree)
            prediction = max(predictions)
            
            y_pred[i] = prediction
        return y_pred
    
    def print_tree(self) -> str:
        print_tree(self.tree, self.feature_names)