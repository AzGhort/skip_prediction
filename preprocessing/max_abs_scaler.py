from data_preprocessor import DataPreprocessor
from sklearn import preprocessing
import numpy as np


class MaxAbsScaler(DataPreprocessor):
    def __init__(self, stats):
        self.preprocessor = preprocessing.MaxAbsScaler()
        max_abs = []
        for i in range(len(stats['max'])):
            max_abs.append(np.abs(stats['max'][i]) if np.abs(stats['max'][i]) > np.abs(stats['min'][i])
                           else np.abs(stats['min'][i]))
        self.preprocessor.max_abs_ = np.array(max_abs)

    def transform(self, data):
        return self.preprocessor.transform(data)