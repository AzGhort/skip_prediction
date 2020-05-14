import numpy as np
from spotify_dataset import SpotifyDataset
import datetime


class Predictor:
    def __init__(self, model, tf_preprocessor_name=None):
        self.model = model
        self.tf_preprocessor_name = tf_preprocessor_name

    def predict(self, sf, tf_first, tf_second):
        return self.model(sf, tf_first, tf_second)

    def train(self, episodes, train_folder, tf_folder):
        print("[Predictor]: TRAINING")
        if train_folder is None or tf_folder is None:
            raise AttributeError("Must specify log and tf folder.")
        print("[Predictor]: Initializing, session features folder: \'" + str(train_folder) + "\', track features folder: \'" + str(tf_folder) + "\'.")
        spotify = SpotifyDataset(train_folder, tf_folder, self.tf_preprocessor_name)
        for e in range(episodes):
            print("[Predictor]: Episode " + str(e) + " starts.")
            dev_aas = []
            dev_fpas = []
            for train_set, dev_set in spotify.get_dataset():
                print("[Predictor]: Dataset created succesfully.")
                print("[Predictor]: Training on dataset starts.")
                self.model.train(train_set)
                print("[Predictor]: Training on dataset ends.")
                print("[Predictor]: Evaluating on dev set.")
                maa, fpa = self.model.evaluate(dev_set)
                print("[Predictor]: Dev set mean average accuracy: " + str(maa))
                print("[Predictor]: Dev set first prediction accuracy: " + str(fpa))
                dev_aas.append(maa)
                dev_fpas.append(fpa)
                print("[Predictor]: Processing set ends, current time is " + str(datetime.datetime.now()))
            print("[Predictor]: Evaluating after " + str(e) + " episodes: " + str(np.mean(dev_aas)) + " mean average accuracy")
            print("[Predictor]: Evaluating after " + str(e) + " episodes: " + str(np.mean(dev_fpas)) + " first prediction accuracy")
            print("[Predictor]: Episode ends, current time is " + str(datetime.datetime.now()))
            print("---------------------------------")

    def evaluate(self, folder, tf_folder):
        aas = []
        fas = []
        print("[Predictor]: EVALUATING")
        if folder is None or tf_folder is None:
            raise AttributeError("Must specify log and tf folder.")
        print("[Predictor]: Initializing, session features folder: \'" + str(folder) + "\', track features folder: \'" + str(tf_folder) + "\'.")
        spotify = SpotifyDataset(folder, tf_folder, self.tf_preprocessor_name)
        for test_set in spotify.get_dataset(False):
            print("[Predictor]: Dataset created succesfully.")
            aa, fpa = self.model.evaluate(test_set)
            print("[Predictor]: Test set mean average accuracy: " + str(aa))
            print("[Predictor]: Test set first prediction accuracy: " + str(fpa))
            aas.append(aa)
            fas.append(fpa)
        mean_average_accuracy = np.mean(aas)
        first_prediction_accuracy = np.mean(fas)
        return mean_average_accuracy, first_prediction_accuracy
