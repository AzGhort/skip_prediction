import numpy as np
from spotify_dataset import SpotifyDataset


class Predictor:
    def __init__(self, model, tf_preprocessor_name=None, sf_preprocessor_name=None):
        self.model = model
        self.tf_preprocessor_name = tf_preprocessor_name
        self.sf_preprocessor_name = sf_preprocessor_name

    def predict(self, sf, tf_first, tf_second):
        return self.model(sf, tf_first, tf_second)

    def train(self, episodes, train_folder, tf_folder):
        print("TRAINING")
        print("Initializing, session features folder: \'" + str(train_folder)
              + "\', track features folder: \'" + str(tf_folder) + "\'.")
        spotify = SpotifyDataset(train_folder, tf_folder, self.tf_preprocessor_name)
        for e in range(episodes):
            print("Episode " + str(e) + " starts.")
            dev_accs = []
            for train_set, dev_set in spotify.get_dataset():
                print("Dataset created succesfully.")
                print("Training on dataset starts.")
                self.model.train(train_set)
                print("Training on dataset ends.")
                print("Evaluating on dev set.")
                dev_accs.append(self.model.evaluate(dev_set))
            print("Evaluating after " + str(e) + " episodes:" + str(np.mean(dev_accs)) + " mean average accuracy")

    def evaluate(self, folder, tf_folder):
        accs = []
        print("EVALUATING")
        if folder is None:
            return "No test folder"
        print("Creating dataset, session features folder: \'" + str(folder) + "\', track features folder: \'" + str(tf_folder) + "\'.")
        spotify = SpotifyDataset(folder, tf_folder, self.tf_preprocessor_name)
        for test_set in spotify.get_dataset(False):
            print("Dataset created succesfully.")
            acc = self.model.evaluate(test_set)
            print("current mean average accuracy is: " + str(acc))
            accs.append(acc)
        mean_average_accuracy = np.mean(accs)
        return mean_average_accuracy
