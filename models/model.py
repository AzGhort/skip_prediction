import numpy as np


class Model:

    def train(self, set):
        pass

    def evaluate(self, set):
        pass

    def average_accuracy(self, prediction, target):
        t = len(prediction)
        prediction.shape = (t, 1)
        aa = 0
        correct = 0
        for i in range(t):
            if prediction[i] == target[i][0]:
                correct += 1
                aa += (correct * 1.0 / (i + 1))
        return aa * 1.0 / t

    def __call__(self, sf, tf, *args, **kwargs):
        pass
