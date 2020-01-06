import numpy as np
from data_parser import DataParser
import os


class SpotifyDataset:
    class Dataset:
        def __init__(self, data, shuffle_batches, seed=42):
            self._data = data
            self._size = len(self._data['session_features'])
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
                    batch[key] = self._data[key][batch_perm]
                yield batch

    def __init__(self, log_folder, tf_folder):
        self.parser = DataParser(tf_folder)
        self.log_folder = log_folder

    def _split_to_dev_train(self, data, percents):
        train_sf, dev_sf = self._split_to_percents(data['session_features'], percents)
        train_tf, dev_tf = self._split_to_percents(data['track_features'], percents)
        train_sk, dev_sk = self._split_to_percents(data['skips'], percents)
        train_data = {'session_features': train_sf, 'track_features': train_tf, 'skips': train_sk}
        dev_data = {'session_features': dev_sf, 'track_features': dev_tf, 'skips': dev_sk}
        return self.Dataset(train_data, shuffle_batches=True), self.Dataset(dev_data, shuffle_batches=False)

    def _split_to_percents(self, data, percents):
        length = np.shape(data)[0]
        fraction = int(length * percents / 100.0)
        return data[:fraction], data[fraction:]

    def get_dataset(self, split_to_train_dev=True, percents=80):
        for filename in os.listdir(self.log_folder):
            print("getting dataset from file " + filename)
            if filename.endswith('.csv'):
                data = self.parser.get_data_from_file(os.path.join(self.log_folder, filename))
                if split_to_train_dev:
                    train, dev = self._split_to_dev_train(data, percents)
                    yield train, dev
                else:
                    yield self.Dataset(data, shuffle_batches=False)
