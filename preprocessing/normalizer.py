from sklearn import preprocessing
from preprocessing.data_preprocessor import DataPreprocessor


class Normalizer(DataPreprocessor):
    def __init__(self):
        preprocessor = preprocessing.Normalizer(copy=False)
        self.preprocessor = preprocessor

    def transform(self, data):
        return self.preprocessor.transform(data)