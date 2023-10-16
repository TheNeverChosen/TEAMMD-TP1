import numpy as np
import pandas as pd
from .utils import find_best_split, partition, node_to_string
from .base import DecisionNode, Leaf
import graphviz

class DecisionTree():
    def __init__(self) -> None:
        self.root = None
        self.feature_names = None
    
    def fit(self, X, y):
        if type(X) == pd.DataFrame:
            self.feature_names = X.columns
            X = X.to_numpy()
        else:
            self.feature_names = range(X.shape[1])
            
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
            
        self.root = build_tree(X,y)
    
    def predict(self, X):    
        y_pred = list()
        if type(X) == pd.DataFrame:
            X = X.to_numpy()
            
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
        
        for i, example in enumerate(X):
            predictions = predict_example(example, self.root)
            prediction = max(predictions)
            
            y_pred.append(prediction)
        return y_pred
    
    def print_tree(self) -> str:
        dot = graphviz.Digraph()
        id_fn = lambda x: str(id(x))
        dot.node(id_fn(self.root), node_to_string(self.root, self.feature_names))
        
        def build_graph(node):
            """World's most elegant tree printing function."""

            # Base case: we've reached a leaf
            if isinstance(node, Leaf):
                # print (spacing + "Predict", node.predictions)
                return

            # Print the question at this node
            # print (spacing + node_to_string(node, feature_names))

            # Call this function recursively on the true branch
            # print (spacing + '--> True:')
            if isinstance(node.true_branch, DecisionNode):
                dot.node(id_fn(node.true_branch), node_to_string(node.true_branch, self.feature_names))
                dot.edge(id_fn(node), id_fn(node.true_branch), "True")
                build_graph(node.true_branch)
            else:
                dot.node(id_fn(node.true_branch), str(node.true_branch), shape='rectangle')
                dot.edge(id_fn(node), id_fn(node.true_branch), "True")

            # Call this function recursively on the false branch
            # print (spacing + '--> False:')
            if isinstance(node.false_branch, DecisionNode):
                dot.node(id_fn(node.false_branch), node_to_string(node.false_branch, self.feature_names))
                dot.edge(id_fn(node), id_fn(node.false_branch), "False")
                build_graph(node.false_branch)
            else:
                dot.node(id_fn(node.false_branch), str(node.false_branch), shape='rectangle')
                dot.edge(id_fn(node), id_fn(node.false_branch), "False")
            
        build_graph(self.root)
        return dot


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    X, y = load_iris(return_X_y=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    tree = DecisionTree()
    tree.fit(X, y)
    y_pred = tree.predict(X)
    
    