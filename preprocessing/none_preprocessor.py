from preprocessing.data_preprocessor import DataPreprocessor


class NonePreprocessor(DataPreprocessor):
    def transform(self, data):
        return data
