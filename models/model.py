import numpy as np
from dataset_description import *


class Model:
    def train(self, set):
        raise NotImplementedError()

    def evaluate(self, set):
        average_accuracies = []
        first_prediction_accuracies = []
        for i in range(len(set.data[DatasetDescription.SF_FIRST_HALF])):
            sf_first = set.data[DatasetDescription.SF_FIRST_HALF][i]
            sf_second = set.data[DatasetDescription.SF_SECOND_HALF][i]
            tf_first = set.data[DatasetDescription.TF_FIRST_HALF][i]
            tf_second = set.data[DatasetDescription.TF_SECOND_HALF][i]
            skips = set.data[DatasetDescription.SKIPS][i]
            prediction = self(sf_first, sf_second, tf_first, tf_second)
            average_accuracies.append(self.average_accuracy(prediction, skips))
            first_prediction_accuracies.append(self.first_prediction_accuracy(prediction, skips))
        return np.mean(average_accuracies), np.mean(first_prediction_accuracies)

    @staticmethod
    def average_accuracy(prediction, target):
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

    @staticmethod
    def first_prediction_accuracy(prediction, target):
        if prediction[0] == target[0][0]:
            return 1
        else:
            return 0

    def __call__(self, sf_first, sf_second, tf_first, tf_second):
        raise NotImplementedError()

    def save_model(self, file):
        raise NotImplementedError()
