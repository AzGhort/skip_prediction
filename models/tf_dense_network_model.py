from model import Model
import tensorflow as tf
import numpy as np
from dataset_description import *
from spotify_dataset import SpotifyDataset
from standard_scaler import StandardScaler

class TrackFeaturesDenseNetwork(Model):
    def __init__(self, hidden_layer_size, hidden_layers_count, batch_size):
        layers = [tf.keras.layers.InputLayer(SpotifyDataset.TRACK_FEATURES)]
        for _ in range(hidden_layers_count):
            layers.append(tf.keras.layers.Dense(hidden_layer_size, activation=tf.nn.relu))
        layers.append(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))
        self.network = tf.keras.Sequential(layers)
        self.batch_size = batch_size
        self.verbose_each = 1000

        self.network.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.BinaryAccuracy()],
        )

    def train(self, set):
        batch_index = 0
        for batch in set.batches(self.batch_size):
            batch_index += 1
            tfs = []
            skips = []
            for j in range(len(batch[DatasetDescription.SF_FIRST_HALF])):
                tf_second, session_skip = batch[DatasetDescription.TF_SECOND_HALF][j], batch[DatasetDescription.SKIPS][j].ravel()
                for i in range(len(session_skip)):
                    tfs.append(tf_second[i])
                    skips.append(session_skip[i])
            x = np.array(tfs)
            y = np.array(skips)
            loss, metric = self.network.train_on_batch(x, y)
            if batch_index % self.verbose_each == 0:
                print("---- loss of batch number " + str(batch_index) + " batches: " + str(loss))
                print("---- binary accuracy of batch number " + str(batch_index) + " batches: " + str(metric))

    def __call__(self, sf_first, sf_second, tf_first, tf_second):
        return np.around(self.network(tf_second))


if __name__ == "__main__":
    import argparse
    from predictor import Predictor

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_folder", default="..\\one_file_train_set", type=str, help="Name of the train log folder.")
    parser.add_argument("--test_folder", default="..\\one_file_test_set", type=str, help="Name of the test log folder.")
    parser.add_argument("--tf_stats_file", default="..\\stats\\track_feature_stats", type=str, help="Name of track feature stats file.")
    parser.add_argument("--tf_folder", default="..\\tf", type=str, help="Name of track features folder")
    parser.add_argument("--episodes", default=1, type=int, help="Number of episodes.")
    parser.add_argument("--hidden_layer", default=32, type=int, help="Size of the hidden layer.")
    parser.add_argument("--layers", default=4, type=int, help="Number of layers.")
    parser.add_argument("--batch_size", default=32, type=int, help="Size of the batch.")
    args = parser.parse_args()

    preprocessor = StandardScaler()
    model = TrackFeaturesDenseNetwork(args.hidden_layer, args.layers, args.batch_size)
    predictor = Predictor(model)
    predictor.train(args.episodes, args.train_folder, args.tf_folder)
    maa = predictor.evaluate_on_files(args.test_folder, args.tf_folder)
    print("Track features dense network model achieved " + str(maa) + " mean average accuracy")
