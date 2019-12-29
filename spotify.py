import numpy as np
import log_parser as lp


class Spotify:
    SKIP = 1
    LOG = 14
    
    class Dataset:
        def __init__(self, data, shuffle_batches, seed=42):
            self._data = data
            self._size = len(self._data["logs"])
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

    def __init__(self, train_filename):
        i = lp.LogParser.get_input_data_from_csv_file(train_filename)
        o = lp.LogParser.get_output_data_from_csv_file(train_filename)
        length = np.shape(i)
        eighty_percent = int(length[0]*0.8)
        train_i = i[:eighty_percent, :]
        train_o = o[:eighty_percent]
        dev_i = i[eighty_percent:, :]
        dev_o = o[eighty_percent:]
        train_data = {'logs': train_i, 'skip': train_o}
        setattr(self, 'train', self.Dataset(train_data, shuffle_batches=True))
        dev_data = {'logs': dev_i, 'skip': dev_o}
        setattr(self, 'dev', self.Dataset(dev_data, shuffle_batches=False))
