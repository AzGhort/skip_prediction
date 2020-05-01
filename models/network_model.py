from models.model import Model
from dataset_description import DatasetDescription
import numpy as np


class NetworkModel(Model):
    def __init__(self, batch_size, verbose_each):
        self.batch_size = batch_size
        self.verbose_each = verbose_each

    def prepare_batch(self, batch):
        raise NotImplementedError()

    def call_on_batch(self, batch_input):
        raise NotImplementedError()

    def __call__(self, sf_first, sf_second, tf_first, tf_second):
        raise NotImplementedError()

    def save_model(self, file):
        raise NotImplementedError()

    def train_on_batch(self, x, y):
        raise NotImplementedError()

    def train(self, set):
        batch_index = 0
        for batch in set.batches(self.batch_size):
            batch_index += 1
            x, y = self.prepare_batch(batch)
            loss, metric = self.train_on_batch(x, y)
            if batch_index % self.verbose_each == 0:
                print("--- loss of batch number " + str(batch_index) + ": " + str(loss))
                print("--- metric of batch number " + str(batch_index) + ": " + str(metric))

    def evaluate(self, set):
        average_accuracies = []
        first_prediction_accuracies = []
        for batch in set.batches(self.batch_size):
            skips = batch[DatasetDescription.SKIPS]
            x, _ = self.prepare_batch(batch)
            predictions = self.call_on_batch(x)
            average_accuracies.append(self.average_accuracy(predictions, skips))
            first_prediction_accuracies.append(self.first_prediction_accuracy(predictions, skips))
        return np.mean(average_accuracies), np.mean(first_prediction_accuracies)