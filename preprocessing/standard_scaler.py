from data_preprocessor import DataPreprocessor
from sklearn import preprocessing
import numpy as np


class StandardScaler(DataPreprocessor):
    def __init__(self, stats):
        self.preprocessor = preprocessing.StandardScaler()
        self.preprocessor.mean_ = np.array(stats['mean'])
        self.preprocessor.var_ = np.array([s ** 2 for s in stats['std']])

    def transform(self, data):
        return self.preprocessor.transform(data)