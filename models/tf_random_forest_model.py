from model import Model
from sklearn import ensemble
import numpy as np
from dataset_description import *
import os


class TrackFeaturesRandomForestModel(Model):
    def __init__(self, estimators=1, preprocessor=None):
        self.classifier = ensemble.RandomForestClassifier(warm_start=True, n_estimators=estimators)
        self.set_estimators = estimators
        super(TrackFeaturesRandomForestModel, self).__init__(preprocessor)

    def train(self, set):
        tfs = []
        skips = []
        for j in range(len(set.data[DatasetDescription.SF_FIRST_HALF])):
            tf_second, session_skip = set.data[DatasetDescription.TF_SECOND_HALF][j], set.data[DatasetDescription.SKIPS][j].ravel()
            for i in range(len(session_skip)):
                tfs.append(self.preprocess(tf_second[i]))
                skips.append(session_skip[i])
        x = np.concatenate(tfs)
        y = np.array(skips)
        self.classifier.fit(x, y)
        self.classifier.n_estimators += self.set_estimators

    def __call__(self, sf_first, sf_second, tf_first, tf_second):
        ret = []
        for tf in tf_second:
            ret.append(self.classifier.predict(self.preprocess(tf))[0])
        return np.array(ret)

    def preprocess(self, data):
        tf_spotify = data[:21].reshape(1, -1)
        acoustic_vectors = data[21:].reshape(1, -1)
        preprocessed = super(TrackFeaturesRandomForestModel, self).preprocess(tf_spotify)
        return np.concatenate((preprocessed, acoustic_vectors), axis=1)


if __name__ == "__main__":
    import argparse
    from predictor import Predictor

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_folder", defaultdefault=".." + os.sep + ".." + os.sep + "one_file_train_set", type=str, help="Name of the train log folder.")
    parser.add_argument("--test_folder", default=".." + os.sep + ".." + os.sep + "one_file_test_set", type=str, help="Name of the test log folder.")
    parser.add_argument("--tf_folder", default=".." + os.sep + "tf", type=str, help="Name of track features folder")
    parser.add_argument("--estimators", default=32, type=int, help="Number of estimators for one set of random forest")
    parser.add_argument("--episodes", default=1, type=int, help="Number of episodes to train")
    parser.add_argument("--preprocessor", default="MinMaxScaler", type=str, help="Name of the preprocessor to use.")
    parser.add_argument("--seed", default=0, type=int, help="Seed to use in numpy and tf.")
    args = parser.parse_args()

    np.random.seed(args.seed)

    preprocessor = Model.get_preprocessor(args.preprocessor)
    model = TrackFeaturesRandomForestModel(args.estimators, preprocessor)
    predictor = Predictor(model)
    predictor.train(args.episodes, args.train_folder, args.tf_folder)
    maa = predictor.evaluate_on_files(args.test_folder, args.tf_folder)
    print("Track features random forest model achieved " + str(maa) + " mean average accuracy")
    print(model.classifier.feature_importances_)
    print("------------------------------------")
