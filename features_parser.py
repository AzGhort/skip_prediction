import numpy as np


class FeaturesParser:
    @staticmethod
    def get_features_map(file):
        features_map = {}
        data = np.genfromtxt(file, delimiter=',', skip_header=1, dtype=str)
        for row in data:
            features_map[row[0]] = np.array(row[1:], np.float32)
        return features_map
