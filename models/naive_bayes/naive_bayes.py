from enum import Enum
import pandas as pd
import numpy as np

class NBDataType(Enum):
  CATEGORICAL = 0
  GAUSSIAN = 1
  MULTINOMIAL = 2

class NaiveBayes:
  def __init__(self, types : list[NBDataType]):
    self.types = types
    self.prior = None
    self.likelihood = None

  def fit(self, X, y):
    if type(X) == pd.DataFrame:
      features = X.columns
    else:
      features = np.arange(X.shape[1])
      
    if len(features) != len(self.types):
      raise TypeError(f"The number of types passed ({len(self.types)}) does not correspond to the number of columns in the dataset ({len(features)})")
    
    # calculate prior probabilities
    labels, counts = np.unique(y, return_counts=True)
    self.prior = dict(zip(*(labels, counts/ len(y))))
    
    # calculate likelihood probabilities
    self.likelihood = {}
    for i, column in enumerate(features):
      self.likelihood[column] = {}
      for label in np.unique(y):
        self.likelihood[column][label] = {}
        if self.types[i] == NBDataType.CATEGORICAL:
          for x in np.unique(X[column]):
            self.likelihood[column][label][x] = len(X[(X[column] == x) & (y == label)]) / len(X[y == label])
            #likelihood[column][y][x] =  P(Xi|y) | Xi in 'column'
        elif self.types[i] == NBDataType.GAUSSIAN:
          self.likelihood[column][label]['mean'] = np.mean(X[y == label]) # mean of column
          self.likelihood[column][label]['std'] = np.std(X[y == label])   # std of column
        elif self.types[i] == NBDataType.MULTINOMIAL:
          for x in np.unique(X[column]):
            self.likelihood[column][label][x] = len(X[(X[column] == x) & (y == label)]) / len(X[y == label])

  # gaussian probability density function
  def _gaussian(self, x, mean, std):
    return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-((x - mean) ** 2) / (2 * (std ** 2)))

  def predict_row(self, x_row):
    y_predict = None
    max_prob = 0
    for y in self.prior.keys():
      prob = self.prior[y]
      for i, column in enumerate(x_row.index):
        if self.types[i] == NBDataType.CATEGORICAL:
          prob *= self.likelihood[column][y][x_row[column]]
        elif self.types[i] == NBDataType.GAUSSIAN:
          prob *= self._gaussian(x_row[column], self.likelihood[column][y]['mean'], self.likelihood[column][y]['std'])
        elif self.types[i] == NBDataType.MULTINOMIAL:
          prob *= self.likelihood[column][y][x_row[column]]
      
      # TODO: TEM UM ERRO ACONTECENDO AQUI, AJEITA PLEASE
      print(prob, max_prob)
      if prob > max_prob:
        max_prob = prob
        y_predict = y
    return y_predict

  def predict(self, x_predict : pd.DataFrame):
    y_predict = []
    for index, row in x_predict.iterrows():
      y_predict.append(self.predict_row(row))
    return y_predict