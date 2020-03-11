from model import Model
from sklearn import ensemble
import numpy as np
from dataset_description import *


class TrackFeaturesRandomForestModel(Model):
    def __init__(self, estimators=1):
        self.classifier = ensemble.RandomForestClassifier(warm_start=True, n_estimators=estimators)
        self.set_estimators = estimators

    def train(self, set):
        tfs = []
        skips = []
        for j in range(len(set.data[DatasetDescription.SF_FIRST_HALF])):
            tf_second, session_skip = set.data[DatasetDescription.TF_SECOND_HALF][j], set.data[DatasetDescription.SKIPS][j].ravel()
            for i in range(len(session_skip)):
                tfs.append(tf_second[i])
                skips.append(session_skip[i])
        x = np.array(tfs)
        y = np.array(skips)
        self.classifier.fit(x, y)
        self.classifier.n_estimators += self.set_estimators

    def __call__(self, sf_first, sf_second, tf_first, tf_second):
        return self.classifier.predict(tf_second)


if __name__ == "__main__":
    import argparse
    from predictor import Predictor

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_folder", default="..\\training_set_mini", type=str, help="Name of the train log folder.")
    parser.add_argument("--test_folder", default="..\\test_set_mini", type=str, help="Name of the test log folder.")
    parser.add_argument("--tf_folder", default="..\\tf", type=str, help="Name of track features folder")
    parser.add_argument("--estimators", default=16, type=int, help="Number of estimators for one set of random forest")
    parser.add_argument("--episodes", default=1, type=int, help="Number of episodes to train")
    args = parser.parse_args()

    model = TrackFeaturesRandomForestModel(args.estimators)
    predictor = Predictor(model)
    predictor.train(args.episodes, args.train_folder, args.tf_folder)
    maa = predictor.evaluate_on_files(args.test_folder, args.tf_folder)
    print("Track features random forest model achieved " + str(maa) + " mean average accuracy")
    print(model.classifier.feature_importances_)
