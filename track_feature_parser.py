import numpy as np
import csv
import enums


class TrackFeatureParser:
    @staticmethod
    def _update_features_map(file, map):
        features_map = {}
        with open(file) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                id = row['track_id']
                del row['track_id']
                row['mode'] = 1. if row['mode'] == 'major' else 0.
                ls = list(row.values())
                features_map[id] = np.array(ls, np.float32)
        map.update(features_map)

    @staticmethod
    def get_track_features(files):
        map = {}
        for file in files:
            TrackFeatureParser._update_features_map(file, map)
        return map


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="tf_0.csv", type=str)
    args = parser.parse_args()

    map = TrackFeatureParser.get_track_features([args.file])
    pass