import numpy as np

class Leaf:
    """A Leaf node classifies data.

    This holds a dictionary of class (e.g., "Apple") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, labels):
        
        self.predictions = dict(zip(*np.unique(labels, return_counts=True)))
        self.prediction = max(self.predictions)
    def __repr__(self) -> str:
        return str(self.prediction)

class DecisionNode:
    """A Decision Node asks a question.

    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self,
                 feature,
                 value,
                 true_branch,
                 false_branch):
        self.feature = feature
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch
    
    def compare(self, X):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = X[self.feature]
        if type(val) != str:
            return val >= self.value
        else:
            return val == self.value
 