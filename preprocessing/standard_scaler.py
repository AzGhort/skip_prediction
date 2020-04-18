from sklearn import preprocessing
from preprocessing.feature_stats import *
from preprocessing.data_preprocessor import DataPreprocessor


class StandardScaler(DataPreprocessor):
    def __init__(self):
        preprocessor = preprocessing.StandardScaler(copy=False)
        preprocessor.mean_ = Stats['mean']
        preprocessor.var_ = np.array([s ** 2 for s in Stats['std']])
        preprocessor.scale_ = Stats['std']
        self.preprocessor = preprocessor

    def transform(self, data):
        return self.preprocessor.transform(data)
