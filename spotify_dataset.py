import numpy as np
from data_parser import DataParser
import os
from dataset_description import *


class SpotifyDataset:
    SKIP = 1
    TRACK_FEATURES = 29
    SESSION_FEATURES = 18
    SESSION_PREDICTABLE_FEATURES = 16

    class Dataset:
        def __init__(self, data, shuffle_batches, seed=42):
            self._data = data
            self._size = len(self._data[DatasetDescription.SF_FIRST_HALF])
            self._shuffler = np.random.RandomState(seed) if shuffle_batches else None

        @property
        def data(self):
            return self._data

        @property
        def size(self):
            return self._size

        def batches(self, size=None):
            permutation = self._shuffler.permutation(self._size) if self._shuffler else np.arange(self._size)
            while len(permutation):
                batch_size = min(size or np.inf, len(permutation))
                batch_perm = permutation[:batch_size]
                permutation = permutation[batch_size:]

                batch = {}
                for key in self._data:
                    batch[key] = np.array(self._data[key])[batch_perm]
                yield batch

    def __init__(self, log_folder, tf_folder, tf_preprocessor):
        self.parser = DataParser(tf_folder, tf_preprocessor)
        self.log_folder = log_folder

    def _split_to_dev_train(self, data, percents):
        train_sf_first, dev_sf_first = self._split_to_percents(data[DatasetDescription.SF_FIRST_HALF], percents)
        train_sf_second, dev_sf_second = self._split_to_percents(data[DatasetDescription.SF_SECOND_HALF], percents)
        train_tf_first, dev_tf_first = self._split_to_percents(data[DatasetDescription.TF_FIRST_HALF], percents)
        train_tf_second, dev_tf_second = self._split_to_percents(data[DatasetDescription.TF_SECOND_HALF], percents)
        train_sk, dev_sk = self._split_to_percents(data[DatasetDescription.SKIPS], percents)
        train_data = {DatasetDescription.SF_FIRST_HALF: train_sf_first,
                      DatasetDescription.SF_SECOND_HALF: train_sf_second,
                      DatasetDescription.TF_FIRST_HALF: train_tf_first,
                      DatasetDescription.TF_SECOND_HALF: train_tf_second,
                      DatasetDescription.SKIPS: train_sk}
        dev_data = {DatasetDescription.SF_FIRST_HALF: dev_sf_first,
                    DatasetDescription.SF_SECOND_HALF: dev_sf_second,
                    DatasetDescription.TF_FIRST_HALF: dev_tf_first,
                    DatasetDescription.TF_SECOND_HALF: dev_tf_second,
                    DatasetDescription.SKIPS: dev_sk}
        return self.Dataset(train_data, shuffle_batches=True), self.Dataset(dev_data, shuffle_batches=False)

    def _split_to_percents(self, data, percents):
        length = np.shape(data)[0]
        fraction = int(length * percents / 100.0)
        return data[:fraction], data[fraction:]

    def get_dataset(self, split_to_train_dev=True, percents=95):
        session_file_count = len([f for f in os.listdir(self.log_folder) if f.endswith('.csv')])
        processed = 0
        for filename in os.listdir(self.log_folder):
            if filename.endswith('.csv'):
                percents = processed * 100.0 / session_file_count
                #if percents <= 33:
                #    processed += 1
                #   continue
                #if percents > 33:
                #    break
                print("[Spotify Dataset]: " + str(percents) + " % of logs already processed.")
                print("[Spotify Dataset]: Creating dataset from session log file " + filename)
                data = self.parser.get_data_from_file(os.path.join(self.log_folder, filename))
                processed += 1
                if split_to_train_dev:
                    train, dev = self._split_to_dev_train(data, percents)
                    yield train, dev
                else:
                    yield self.Dataset(data, shuffle_batches=False)
