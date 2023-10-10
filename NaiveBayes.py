from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# categorical naive bayes
class NaiveBayes:
  def _init_(self):
    pass

  def fit(self, x : pd.DataFrame, y):
    # calculate prior probabilities
    self.prior = {}
    for y in y.unique():
      prior[y] = len(y[y == y]) / len(y)
    
    # calculate likelihood
    self.likelihood = {}
    for column in x.columns:
      likelihood[column] = {}
      for y in y.unique():
        likelihood[column][y] = {} #column | y
        for value in x[column].unique():
          likelihood[column][y][value] = len(x[column][x[column] == value & y == y]) / len(y[y == y])
          # probability of 'value' appearing in column 'column' given 'y'
          # in 'column': P(value | y)

  # predict one x, returning y, using prior and likelihood calculated in fit
  def predict_one(self, x : pd.Series):
    # calculate posterior probabilities
    posterior = {}
    for y in self.prior.keys():
      posterior[y] = self.prior[y]
      for column in x.columns:
        posterior[y] *= self.likelihood[column][y][x[column]]
    
    # return the max posterior
    return max(posterior, key = posterior.get)
  
  # predict x, returning y, using prior and likelihood calculated in fit
  def predict(self, x : pd.DataFrame):
    y = []
    for i in range(len(x)):
      y.append(self.predict_one(x.iloc[i]))
    return y