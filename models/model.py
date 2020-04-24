import numpy as np
from dataset_description import *
from preprocessing.standard_scaler import StandardScaler
from preprocessing.normalizer import Normalizer
from preprocessing.none_preprocessor import NonePreprocessor
from preprocessing.min_max_scaler import MinMaxScaler


class Model:
    def train(self, set):
        raise NotImplementedError()

    def evaluate(self, set):
        accuracies = []
        for i in range(len(set.data[DatasetDescription.SF_FIRST_HALF])):
            sf_first = set.data[DatasetDescription.SF_FIRST_HALF][i]
            sf_second = set.data[DatasetDescription.SF_SECOND_HALF][i]
            tf_first = set.data[DatasetDescription.TF_FIRST_HALF][i]
            tf_second = set.data[DatasetDescription.TF_SECOND_HALF][i]
            skips = set.data[DatasetDescription.SKIPS][i]
            prediction = self(sf_first, sf_second, tf_first, tf_second)
            accuracies.append(self.average_accuracy(prediction, skips))
        return np.mean(accuracies)

    def average_accuracy(self, prediction, target):
        prediction = np.array(prediction)
        t = len(prediction)
        prediction.shape = (t, 1)
        aa = 0
        correct = 0
        for i in range(t):
            if prediction[i] == target[i][0]:
                correct += 1
                aa += (correct * 1.0 / (i + 1))
        return aa * 1.0 / t

    def __call__(self, sf_first, sf_second, tf_first, tf_second):
        raise NotImplementedError()


