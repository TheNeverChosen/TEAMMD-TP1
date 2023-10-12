from enum import Enum
import pandas as pd
import numpy as np

class DataType(Enum):
  CATEGORICAL = 1
  GAUSSIAN = 2
  MULTINOMIAL = 3

# categorical naive bayes
class NaiveBayes:
  # types: list of DataType, one for each column in x
  def _init_(self, types : list):
    self.types = types
    self.prior = None
    self.likelihood = None

  def fit(self, x_fit : pd.DataFrame, y_fit):
    # calculate prior probabilities
    self.prior = {}
    for y in y_fit.unique():
      self.prior[y] = len(y_fit[y_fit == y]) / len(y_fit)
    
    # calculate likelihood probabilities
    self.likelihood = {}
    for column in x_fit.columns:
      self.likelihood[column] = {}
      for y in y_fit.unique():
        self.likelihood[column][y] = {}
        if self.types[column] == DataType.CATEGORICAL:
          for x in x_fit[column].unique():
            self.likelihood[column][y][x] = len(x_fit[(x_fit[column] == x) & (y_fit == y)]) / len(x_fit[y_fit == y])
            #likelihood[column][y][x] =  P(Xi|y) | Xi in 'column'
        elif self.types[column] == DataType.GAUSSIAN:
          self.likelihood[column][y]['mean'] = np.mean(x_fit[y_fit == y]) # mean of column
          self.likelihood[column][y]['std'] = np.std(x_fit[y_fit == y])   # std of column
        elif self.types[column] == DataType.MULTINOMIAL:
          for x in x_fit[column].unique():
            self.likelihood[column][y][x] = len(x_fit[(x_fit[column] == x) & (y_fit == y)]) / len(x_fit[y_fit == y])

  # gaussian probability density function
  def _gaussian(self, x, mean, std):
    return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-((x - mean) ** 2) / (2 * (std ** 2)))

  # predict y, given x
  def predict_row(self, x_row):
    y_predict = None
    max_prob = 0
    for y in self.prior.keys():
      prob = self.prior[y]
      for column in x_row.index:
        if self.types[column] == DataType.CATEGORICAL:
          prob *= self.likelihood[column][y][x_row[column]]
        elif self.types[column] == DataType.GAUSSIAN:
          prob *= self._gaussian(x_row[column], self.likelihood[column][y]['mean'], self.likelihood[column][y]['std'])
        elif self.types[column] == DataType.MULTINOMIAL:
          prob *= self.likelihood[column][y][x_row[column]]
      if prob > max_prob:
        max_prob = prob
        y_predict = y
    return y_predict

  # predict x, returning y, using prior and likelihood calculated in fit
  def predict(self, x_predict : pd.DataFrame):
    y_predict = []
    for index, row in x_predict.iterrows():
      y_predict.append(self._predict(row))
    return y_predict