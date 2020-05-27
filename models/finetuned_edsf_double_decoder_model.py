import tensorflow as tf
from models.finetuned_encoder_decoder_model import FinetunedEncoderDecoderModel
import os
import numpy as np
from spotify_dataset import *
import logging
from prediction_importances import *


class FinetunedEDSFDoubleDecoderModel(FinetunedEncoderDecoderModel):
    def __init__(self, batch_size, verbose_each, saved_model_file, weighted_loss, trainable_decoder):
        super(FinetunedEDSFDoubleDecoderModel, self).__init__(batch_size, verbose_each, saved_model_file, trainable_decoder)
        second_half_sf_predicted = self.pretrained_model.output

        decoders = [
            tf.keras.layers.LSTM(256, return_sequences=True, return_state=True, name="SkipDecoder"),
            tf.keras.layers.Dropout(0.5, name="SkipDropout"),
            tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, name="SkipPrediction")
        ]

        all_predictions = []
        for i in range(0, 10):
            x = tf.keras.layers.Concatenate(name="SkipConcatenatedTo_PredictedSF_" + str(i))([
                self.encoder_level_bidirs[i],
                self._get_nth_lambda_layer(second_half_sf_predicted, i)
            ])
            if i == 0:
                x, state_h, state_c = decoders[0](x)
            else:
                x, state_h, state_c = decoders[0](x, initial_state=[state_h, state_c])
            x = decoders[1](x)
            skip_prediction = decoders[2](x)
            all_predictions.append(skip_prediction)

        predictions_combined = tf.keras.layers.Lambda(lambda x: tf.keras.layers.Concatenate(axis=1)(x), name="Skip_predictions")(all_predictions)

        self.network = tf.keras.Model(
            inputs=self.pretrained_model.inputs,
            outputs=[predictions_combined])

        self.network.compile(
            optimizer=tf.optimizers.Adam(),
            loss=tf.keras.losses.BinaryCrossentropy(),
            sample_weight_mode='temporal' if weighted_loss else None,
            metrics=[tf.keras.metrics.BinaryAccuracy()]
        )

        self.weighted_loss = weighted_loss
        #os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
        #tf.keras.utils.plot_model(self.network, to_file='finetuned_edsf_double_decoder.png')

    def __call__(self, sf_first, sf_second, tf_first, tf_second):
        second_half_real_length = sf_second.shape[0]
        last_sf_first_half = self._get_last_session_features(sf_first).reshape((1, 1, SpotifyDataset.SESSION_PREDICTABLE_FEATURES))
        sf_first = self._pad_input(sf_first).reshape((1, 10, SpotifyDataset.SESSION_FEATURES))
        tf_first = self._pad_input(tf_first).reshape((1, 10, SpotifyDataset.TRACK_FEATURES))
        tf_second = self._pad_input(tf_second).reshape((1, 10, SpotifyDataset.TRACK_FEATURES))
        network_output = self.network([sf_first, tf_first, tf_second, last_sf_first_half])
        predictions = []
        assert network_output.shape == (1, 10, 1)
        for i in range(second_half_real_length):
            predictions.append(np.around(network_output[0][i][0]))
        return predictions

    @staticmethod
    def _get_nth_lambda_layer(tensor, n):
        return tf.keras.layers.Lambda(lambda x: tf.slice(x, [0, n, 0], [-1, 1, -1]))(tensor)

    @staticmethod
    def _pad_input(features):
        return np.pad(features, [(0, 10 - features.shape[0]), (0, 0)])

    def _pad_targets(self, skips):
        constant = -1 if self.weighted_loss else 0
        return np.pad(skips, [(0, 10 - skips.shape[0]), (0, 0)], constant_values=constant)

    @staticmethod
    def _get_last_session_features(sf_first_half):
        last_sf = sf_first_half[-1, :SpotifyDataset.SESSION_PREDICTABLE_FEATURES]
        return last_sf.reshape((1, SpotifyDataset.SESSION_PREDICTABLE_FEATURES))

    def call_on_batch(self, batch_input):
        batch_len = batch_input[0].shape[0]
        network_output = self.network.predict_on_batch(batch_input)
        return np.around(network_output[:, :]).reshape((batch_len, 10, 1))

    def train_on_batch(self, inputs, targets):
        if not self.weighted_loss:
            return self.network.train_on_batch(inputs, targets)
        else:
            current_batch_size = min(self.batch_size, len(targets))
            sample_weights = np.zeros((current_batch_size, 10), dtype=np.float32)
            i = 0
            for sample in targets:
                j = 0
                for skip in sample:
                    if skip == -1:
                        break
                    sample_weights[i][j] = Importances[j]
                    j += 1
                i += 1
            return self.network.train_on_batch(inputs, targets, sample_weight=sample_weights)

    def prepare_batch(self, batch):
        current_batch_size = min(self.batch_size, batch[DatasetDescription.SF_FIRST_HALF].shape[0])
        sfs_first_half = np.zeros((current_batch_size, 10, SpotifyDataset.SESSION_FEATURES), dtype=np.float32)
        tfs_first_half = np.zeros((current_batch_size, 10, SpotifyDataset.TRACK_FEATURES), dtype=np.float32)
        tfs_second_half = np.zeros((current_batch_size, 10, SpotifyDataset.TRACK_FEATURES), dtype=np.float32)
        targets = np.zeros((current_batch_size, 10, 1), dtype=np.float32)
        last_sfs_first_half = np.zeros((current_batch_size, 1, SpotifyDataset.SESSION_PREDICTABLE_FEATURES), dtype=np.float32)

        for i in range(current_batch_size):
            last_sfs_first_half[i] = self._get_last_session_features(batch[DatasetDescription.SF_FIRST_HALF][i])
            targets[i] = self._pad_targets(batch[DatasetDescription.SKIPS][i])
            sfs_first_half[i] = self._pad_input(batch[DatasetDescription.SF_FIRST_HALF][i])
            tfs_first_half[i] = self._pad_input(batch[DatasetDescription.TF_FIRST_HALF][i])
            tfs_second_half[i] = self._pad_input(batch[DatasetDescription.TF_SECOND_HALF][i])

        return [sfs_first_half, tfs_first_half, tfs_second_half, last_sfs_first_half], targets


if __name__ == "__main__":
    import argparse
    from predictor import Predictor

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_folder", default=".." + os.sep + ".." + os.sep + "mini_train_set", type=str, help="Name of the train log folder.")
    parser.add_argument("--test_folder", default=".." + os.sep + ".." + os.sep + "mini_test_set", type=str, help="Name of the test log folder.")
    parser.add_argument("--tf_folder", default=".." + os.sep + "tf", type=str, help="Name of track features folder")
    parser.add_argument("--episodes", default=1, type=int, help="Number of episodes.")
    parser.add_argument("--batch_size", default=512, type=int, help="Size of the batch.")
    parser.add_argument("--seed", default=0, type=int, help="Seed to use in numpy and tf.")
    parser.add_argument("--tf_preprocessor", default="MinMaxScaler", type=str, help="Name of the track features preprocessor to use.")
    parser.add_argument("--result_dir", default="results", type=str, help="Name of the results folder.")
    parser.add_argument("--model_name", default="finetuned_edsf_double_decoder", type=str, help="Name of the model to save.")
    parser.add_argument("--saved_weights_folder", default=".." + os.sep + "saved_models" + os.sep + "edsf_5" + os.sep + "encoder_decoder_sf", type=str, help="Name of the folder of saved lengths.")
    parser.add_argument("--weighted_loss", default=True, type=bool, help="Whether to use weighted loss.")
    parser.add_argument("--trainable_decoder", default=True, type=bool, help="Whether the original decoder layers are trainable.")
    args = parser.parse_args()

    # no warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    logging.disable(logging.WARNING)

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    model = FinetunedEDSFDoubleDecoderModel(args.batch_size, 100, args.saved_weights_folder, args.weighted_loss, args.trainable_decoder)
    #model.network.load_weights(args.result_dir + os.sep + args.model_name)

    predictor = Predictor(model, args.tf_preprocessor)
    predictor.train(args.episodes, args.train_folder, args.tf_folder)

    #maa, fpa = predictor.evaluate(args.test_folder, args.tf_folder)

    model.save_model(args.result_dir + os.sep + args.model_name)

    print(str(args))
    #print("Finetuned edsf double decoder prediction model achieved " + str(maa) + " mean average accuracy")
    #print("Finetuned edsf double decoder prediction model achieved " + str(fpa) + " first prediction accuracy")
    print("------------------------------------")