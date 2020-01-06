import numpy as np
from spotify_dataset import SpotifyDataset


class Predictor:
    def __init__(self, model):
        self.model = model

    def predict(self, sf, tf):
        return self.model(sf, tf)

    def train(self, episodes, train_folder, tf_folder):
        for e in range(episodes):
            spotify = SpotifyDataset(train_folder, tf_folder)
            dev_accs = []
            for train_set, dev_set in spotify.get_dataset():
                self.model.train(train_set)
                dev_accs.append(self.model.evaluate(dev_set))
            print("Evaluating after " + e + " episodes:" + np.mean(dev_accs) + " mean average accuracy")

    def evaluate_on_files(self, folder, tf_folder):
        accs = []
        spotify = SpotifyDataset(folder, tf_folder)
        for test_set in spotify.get_dataset(False):
            acc = self.model.evaluate(test_set)
            print("current mean average accuracy is: " + str(acc))
            accs.append(acc)
        mean_average_accuracy = np.mean(accs)
        return mean_average_accuracy
