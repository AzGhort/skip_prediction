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

            prediction_len = prediction.shape[0]
            prediction = prediction.reshape((1, prediction_len, 1))
            skips = skips.reshape((1, prediction_len, 1))

            average_accuracies.append(self.average_accuracy(prediction, skips))
            first_prediction_accuracies.append(self.first_prediction_accuracy(prediction, skips))
        return np.mean(average_accuracies), np.mean(first_prediction_accuracies)

    @staticmethod
    def average_accuracy(predictions, targets):
        aas = []
        targets_length = targets.shape[0]
        # for each session
        for i in range(targets_length):
            aa = 0
            correct = 0
            t = targets[i].shape[0]
            # for each track
            for j in range(t):
                if predictions[i][j][0] == targets[i][j][0]:
                    correct += 1
                    aa += (correct * 1.0 / (j + 1))
            aas.append(aa * 1.0 / t)
        return np.mean(aas)

    @staticmethod
    def first_prediction_accuracy(predictions, targets):
        fpas = []
        targets_length = targets.shape[0]
        # for each session
        for i in range(targets_length):
            if predictions[i][0][0] == targets[i][0][0]:
                fpas.append(1.0)
            else:
                fpas.append(0.0)
        return np.mean(fpas)

    def __call__(self, sf_first, sf_second, tf_first, tf_second):
        raise NotImplementedError()
