import numpy as np
from data_parser import DataParser
import enums


class Spotify:
    class Dataset:
        def __init__(self, data, shuffle_batches, seed=42):
            self._data = data
            self._size = len(self._data['features'])
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

    def _split_to_dev_train(self, i, o, percents):
        length = np.shape(i)[0]
        fraction = int(length * percents / 100.0)
        train_i = i[:fraction, :]
        train_o = o[:fraction]
        dev_i = i[fraction:, :]
        dev_o = o[fraction:]
        train_data = {'features': train_i, 'skip': train_o}
        dev_data = {'features': dev_i, 'skip': dev_o}
        return self.Dataset(train_data, shuffle_batches=True), self.Dataset(dev_data, shuffle_batches=False)

    def __init__(self, mode, logs, tf_files=None):
        self.parser = DataParser(mode, tf_files)
        self.logs = logs

    def get_next_data(self, split_to_train_dev=True, percents=80):
        for logfile in self.logs:
            i, o = self.parser.get_data_from_file(logfile)
            if split_to_train_dev:
                yield self._split_to_dev_train(i, o, percents)
            else:
                data = {'features': i, 'skip': o}
                yield self.Dataset(data, shuffle_batches=False)

