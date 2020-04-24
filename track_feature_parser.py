import numpy as np
import csv
import os
from dataset_description import *
from preprocessing.min_max_scaler import MinMaxScaler
from preprocessing.none_preprocessor import NonePreprocessor
from preprocessing.normalizer import Normalizer
from preprocessing.standard_scaler import StandardScaler


class TrackFeatureParser:
    def __init__(self, preprocessor):
        self.preprocessor = self.get_preprocessor(preprocessor)

    def _update_features_map(self, file, map):
        features_map = {}
        with open(file) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                id = row[TrackFeatureFields.TRACK_ID]
                del row[TrackFeatureFields.TRACK_ID]
                row[TrackFeatureFields.MODE] = 1. if row[TrackFeatureFields.MODE] == TrackMode.MAJOR else 0.
                ls = list(row.values())
                tf_spotify = np.array(ls[:21], np.float32).reshape(1, -1)
                acoustic_vectors = np.array(ls[21:], np.float32).reshape(1, -1)
                preprocessed = np.array(self.preprocessor.transform(tf_spotify), np.float32)
                features_map[id] = np.concatenate((preprocessed, acoustic_vectors), axis=1).flatten()
        map.update(features_map)

    def get_track_features(self, folder):
        map = {}
        for filename in os.listdir(folder):
            if filename.endswith('.csv'):
                self._update_features_map(os.path.join(folder, filename), map)
        return map

    @staticmethod
    def get_preprocessor(name):
        if name == "StandardScaler":
            return StandardScaler()
        elif name == "Normalizer":
            return Normalizer()
        elif name == "MinMaxScaler":
            return MinMaxScaler()
        else:
            return NonePreprocessor()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="tf", type=str)
    args = parser.parse_args()

    tfp = TrackFeatureParser(None)
    map = tfp.get_track_features(args.folder)