from preprocessing.data_preprocessor import DataPreprocessor
from preprocessing.track_feature_stats import *
from sklearn import preprocessing


class MinMaxScaler(DataPreprocessor):
    def __init__(self):
        preprocessor = preprocessing.MinMaxScaler(copy=False)
        preprocessor.data_max_ = Stats['max']
        preprocessor.data_min_ = Stats['min']
        preprocessor.data_range_ = preprocessor.data_max_ - preprocessor.data_min_
        # we always fit just one sample, so we use global min and max for X.min() and X.max()
        preprocessor.scale_ = np.ones(preprocessor.data_range_.shape) / preprocessor.data_range_
        preprocessor.min_ = np.zeros(preprocessor.data_min_.shape) - preprocessor.data_min_ * preprocessor.scale_
        self.preprocessor = preprocessor

    def transform(self, data):
        return self.preprocessor.transform(data)