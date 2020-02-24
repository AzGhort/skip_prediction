from model import Model
from sklearn import ensemble
import numpy as np
import constants


class SessionFeaturesRandomForestModel(Model):
    def __init__(self, estimators=1, window=5):
        self.classifier = ensemble.RandomForestClassifier(warm_start=True, n_estimators=estimators)
        self.set_estimators = estimators
        self.window_size = window

    def train(self, set):
        windows = []
        skips = []
        for j in range(len(set.data[constants.SF_FIRST_HALF])):
            sf, session_skip = set.data[constants.SF_FIRST_HALF][j].tolist(), set.data[constants.SKIPS][j].ravel()
            for i in range(len(session_skip)):
                window = self.get_window(sf, i)
                windows.append(window)
                # append predicted session features
                skips.append(session_skip[i])
                skip = self.classifier.predict(window)
                sf.append(self.classifier.predict(skip[3]))
        x = np.array(windows)
        y = np.array(skips)
        self.classifier.fit(x, y)
        self.classifier.n_estimators += self.set_estimators

    def get_window(self, sf, predicted_index):
        start_index = len(sf) - self.window_size + predicted_index
        end_index = start_index + self.window_size
        return np.array(sf[start_index:end_index])

    def predict_on_session(self, sf_first):
        sfs = []
        ret = []
        for i in range(len(sf_first)):
            window = self.get_window(sfs, i)
            sf_predicted = self.classifier.predict(window)
            ret.append(sf_predicted[3])
            sfs.append(sf_predicted)
        return np.array(ret)

    def __call__(self, sf_first, sf_second, tf_first, tf_second):
        return self.predict_on_session(sf_first)


if __name__ == "__main__":
    import argparse
    from predictor import Predictor

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_folder", default="..\\training_set_mini", type=str, help="Name of the train log folder.")
    parser.add_argument("--test_folder", default="..\\test_set_mini", type=str, help="Name of the test log folder.")
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
