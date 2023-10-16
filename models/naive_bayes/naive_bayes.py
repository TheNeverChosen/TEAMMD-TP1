import numpy as np
import pandas as pd

from enum import Enum

class NBDataType(Enum):
  CATEGORICAL = 0
  GAUSSIAN = 1

class NaiveBayes:
  def __init__(self, types : list[NBDataType]):
    self.types = types
    self.prior = None
    self.likelihood = None

  def fit(self, X, y):
    #USING ONLY NUMPY.NDARRAY
    if type(X) == pd.DataFrame:
      X = X.to_numpy()
    if type(y) == pd.Series:
      y = y.to_numpy()
    if type(X) != np.ndarray or type(y) != np.ndarray:
      raise TypeError(f"X and y must be numpy.ndarray or pandas.DataFrame and pandas.Series, respectively")

    if len(self.types) != X.shape[1]:
      raise TypeError(f"The number of types passed ({len(self.types)}) does not correspond to the number of columns in the dataset ({len(features)})")
    
    # calculate prior probabilities
    labels, counts = np.unique(y, return_counts=True)
    self.prior = dict(zip(labels, counts / len(y)))

    # calculate likelihood probabilities
    self.likelihood = {}

    for label in self.prior.keys():
      self.likelihood[label] = {}
      for i in range(X.shape[1]):
        if self.types[i] == NBDataType.GAUSSIAN:
          self.likelihood[label][i] = {
            "mean": X[y == label, i].mean(),
            "std": X[y == label, i].std()
          }
        elif self.types[i] == NBDataType.CATEGORICAL:
          self.likelihood[label][i] = np.unique(X[y == label, i], return_counts=True)[1] / len(X[y == label, i])
        else:
          raise TypeError(f"Invalid type {self.types[i]}")

  def gaussian(self, x, mean, std):
    return 1 / (np.sqrt(2 * np.pi) * std) * np.exp(-((x - mean) ** 2) / (2 * std ** 2))
  
  def predict_row(self, x_row):
    # calculate posterior probabilities
    posterior = {}
    for label in self.prior.keys():
      posterior[label] = self.prior[label]
      for i in range(len(x_row)):
        if self.types[i] == NBDataType.GAUSSIAN:
          posterior[label] *= self.gaussian(x_row[i], self.likelihood[label][i]["mean"], self.likelihood[label][i]["std"])
        elif self.types[i] == NBDataType.CATEGORICAL:
          posterior[label] *= self.likelihood[label][i][x_row[i]]
        else:
          raise TypeError(f"Invalid type {self.types[i]}")
    
    # return the label with the highest posterior probability
    return max(posterior, key=posterior.get)
  
  def predict(self, X):
    #USING ONLY NUMPY.NDARRAY
    if type(X) == pd.DataFrame:
      X = X.to_numpy()
    if type(X) != np.ndarray:
      raise TypeError(f"X must be numpy.ndarray or pandas.DataFrame")

    return np.array([self.predict_row(x_row) for x_row in X])