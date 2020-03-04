from model import Model
from sklearn import ensemble
import numpy as np
from dataset_description import *


class SessionFeaturesRandomForestModel(Model):
    def __init__(self, estimators=1, window=5):
        self.regressor = ensemble.RandomForestRegressor(warm_start=True, n_estimators=estimators)
        self.set_estimators = estimators
        self.window_size = window

    def train(self, set):
        windows = []
        session_features = []
        for j in range(len(set.data[DatasetDescription.SF_FIRST_HALF])):
            sf_first_half, sf_second_half = set.data[DatasetDescription.SF_FIRST_HALF][j].tolist(),\
                                            set.data[DatasetDescription.SF_SECOND_HALF][j].tolist()
            for i in range(len(sf_second_half)):
                window = self.get_window(sf_first_half)
                # initial fit, so that the dimensions match
                if j == 0 and i == 0:
                    y = np.array(sf_second_half[0]).reshape(1, -1)
                    self.regressor.fit(window, y)
                windows.append(window)
                # append real target session features
                session_features.append(np.array(sf_second_half[i]).reshape(1, -1))
                sf_predicted = self.regressor.predict(window)
                # append predicted session features to window buffer
                sf_first_half.append(sf_predicted.tolist())
        x = np.array(windows)
        y = np.array(session_features)
        self.regressor.fit(x, y)
        self.regressor.n_estimators += self.set_estimators

    def get_window(self, sf):
        start_index = len(sf) - self.window_size
        return np.array(np.array(w) for w in sf[start_index:]).flatten().reshape(1, -1)

    def predict_on_session(self, sf_first):
        sfs = []
        ret = []
        for i in range(len(sf_first)):
            window = self.get_window(sfs, i)
            sf_predicted = self.regressor.predict(window)
            ret.append(sf_predicted[3])
            sfs.append(sf_predicted)
        return np.array(ret)

    def __call__(self, sf_first, sf_second, tf_first, tf_second):
        return self.predict_on_session(sf_first)


if __name__ == "__main__":
    import argparse
    from predictor import Predictor

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_folder", default="..\\one_file_train_set", type=str, help="Name of the train log folder.")
    parser.add_argument("--test_folder", default="..\\one_file_test_set", type=str, help="Name of the test log folder.")
    parser.add_argument("--tf_folder", default="..\\tf", type=str, help="Name of track features folder")
    parser.add_argument("--estimators", default=8, type=int, help="Number of estimators for one set of random forest")
    parser.add_argument("--episodes", default=1, type=int, help="Number of episodes to train")
    parser.add_argument("--window", default=5, type=int, help="Number of tracks previous to the track predicted the model sees")
    args = parser.parse_args()

    model = SessionFeaturesRandomForestModel(args.estimators, args.window)
    predictor = Predictor(model)
    predictor.train(args.episodes, args.train_folder, args.tf_folder)
    maa = predictor.evaluate_on_files(args.test_folder, args.tf_folder)
    print("Last user skip model achieved " + str(maa) + " mean average accuracy")
    print(model.regressor.feature_importances_)
