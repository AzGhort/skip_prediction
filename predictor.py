import numpy as np
from spotify_dataset import SpotifyDataset
import datetime
from models.encoder_decoder_sf import EncoderDecoderSF


class Predictor:
    def __init__(self, model, tf_preprocessor_name=None):
        self.model = model
        self.tf_preprocessor_name = tf_preprocessor_name

    def predict(self, sf, tf_first, tf_second):
        return self.model(sf, tf_first, tf_second)

    def train(self, episodes, train_folder, tf_folder):
        print("[Predictor]: TRAINING")
        if train_folder is None or tf_folder is None:
            return

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
        if tf_folder is None or folder is None:
            raise AttributeError("Must specify tf folder.")

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

    def evaluate_skip_accuracies(self, folder, tf_folder):
        aas = [[] for _ in range(10)]
        means = []
        print("[Predictor]: EVALUATING TRUE ACCURACIES")
        if folder is None or tf_folder is None:
            return

        print("[Predictor]: Initializing, session features folder: \'" + str(folder) + "\', track features folder: \'" + str(tf_folder) + "\'.")
        spotify = SpotifyDataset(folder, tf_folder, self.tf_preprocessor_name)
        for test_set in spotify.get_dataset(False):
            print("[Predictor]: Dataset created succesfully.")
            accuracies, mean = self.model.evaluate_skip_accuracies(test_set)
            print("[Predictor]: Test set average accuracy: " + str(mean))
            i = 0
            for acc in accuracies:
                aas[i].append(acc)
                print("[Predictor]: Test set " + str(11 + i) + "th prediction accuracy: " + str(acc))
                i += 1
            means.append(mean)

        mean_accuracy = np.mean(means)
        average_accuracies = [np.mean(acc) for acc in aas]
        return average_accuracies, mean_accuracy

    def evaluate_all_features_accuracies(self, folder, tf_folder):
        feature_accuracies = [[[] for _ in range(10)] for _ in range(SpotifyDataset.SESSION_PREDICTABLE_FEATURES)]
        feature_mean_accuracies = [[0 for _ in range(10)] for _ in range(SpotifyDataset.SESSION_PREDICTABLE_FEATURES)]
        print("[Predictor]: EVALUATING ACCURACIES OF ALL FEATURES")
        if folder is None or tf_folder is None:
            return

        print("[Predictor]: Initializing, session features folder: \'" + str(folder) + "\', track features folder: \'" + str(tf_folder) + "\'.")
        spotify = SpotifyDataset(folder, tf_folder, self.tf_preprocessor_name)
        for test_set in spotify.get_dataset(False):
            print("[Predictor]: Dataset created succesfully.")
            accuracies = self.model.evaluate_all_feature_accuracies(test_set)
            for feature_index in range(SpotifyDataset.SESSION_PREDICTABLE_FEATURES):
                for song_index in range(10):
                    acc = accuracies[feature_index][song_index]
                    feature_accuracies[feature_index][song_index].append(acc)
                    print("[Predictor]: Dev set " + str(feature_index) + "th feature accuracy of " + str(11 + song_index) +"th song: " + str(acc))

        for feature_index in range(SpotifyDataset.SESSION_PREDICTABLE_FEATURES):
            for song_index in range(10):
                feature_mean_accuracies[feature_index][song_index] = np.mean(feature_accuracies[feature_index][song_index])

        return feature_mean_accuracies
