import tensorflow as tf
from models.finetuned_encoder_decoder_model import FinetunedEncoderDecoderModel
import os
import numpy as np
from spotify_dataset import *
import logging
from prediction_importances import *


class FinetunedEDSFFirstPredictionModel(FinetunedEncoderDecoderModel):
    def __init__(self, batch_size, verbose_each, saved_model_file, train_on_predicted, eval_on_predicted):
        super(FinetunedEDSFFirstPredictionModel, self).__init__(batch_size, verbose_each, saved_model_file, False)
        self.train_on_predicted = train_on_predicted
        self.eval_on_predicted = eval_on_predicted
        encoder_out_states = self.encoder_out_states
        self.first_half_sf_predicted = self.pretrained_model.output

        previous_predicted_input = tf.keras.layers.Input(shape=(1, 1))

        transformer = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        decoders = [
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, return_state=True), name="TFDecoder"),
            tf.keras.layers.Dropout(0.5, name="SkipDropout"),
            tf.keras.layers.Dense(128, activation=tf.nn.relu, name='SkipFeedforward_1'),
            tf.keras.layers.Dense(64, activation=tf.nn.relu, name='SkipFeedforward_2'),
            tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, name="SkipPrediction")
        ]

        x = tf.keras.layers.Concatenate(name="DecoderInput_0")([
            tf.keras.layers.RepeatVector(1, name="SecondHalf_SessionRepresentation_0")(self.session_representation),
            tf.keras.layers.RepeatVector(1, name="SecondHalf_TF_0")(
                self._get_nth_lambda_layer(self.second_half_tf_transformer, 0))
        ])
        x = transformer(x)
        x, forward_h, forward_c, backward_h, backward_c = decoders[0](x, initial_state=[encoder_out_states[0],
                                                                                        encoder_out_states[1],
                                                                                        encoder_out_states[2],
                                                                                        encoder_out_states[3]])
        x = tf.keras.layers.Concatenate(name="ConcatenatedTo_Skip_-1")([
            x,
            previous_predicted_input
        ])
        x = decoders[1](x)
        x = decoders[2](x)
        x = decoders[3](x)
        prediction = decoders[4](x)

        features_inputs = self.pretrained_model.inputs[:3]
        self.network = tf.keras.Model(
            inputs=[features_inputs, previous_predicted_input],
            outputs=[prediction])

        self.network.compile(
            optimizer=tf.optimizers.Adam(),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()]
        )
        #os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
        #tf.keras.utils.plot_model(self.network, to_file='finetuned_edsf_first_prediction.png')

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
    def _get_nth_session_feature_window(first_half, second_half, n):
        if n == 0:
            return first_half
        batch_len = first_half.shape[0]

        if first_half.shape[2] != second_half.shape[2]:
            last_two_columns = np.full((batch_len, n, 2), 0.)
            cut_second_half = second_half[:, :n, :]
            second_half_columns = np.concatenate((cut_second_half, last_two_columns), 2)
        else:
            second_half_columns = second_half[:, :n, :]
        first_half_columns = first_half[:, n:, :]
        return np.concatenate((first_half_columns, second_half_columns), 1)

    @staticmethod
    def _get_nth_lambda_layer(tensor, n):
        return tf.keras.layers.Lambda(lambda x: x[:, n])(tensor)

    @staticmethod
    def _get_last_session_skips(sf_first_half):
        last_sf = sf_first_half[-1, 2]
        return last_sf.reshape((1, 1))

    @staticmethod
    def _get_last_session_features(sf_first_half):
        last_sf = sf_first_half[-1, :SpotifyDataset.SESSION_PREDICTABLE_FEATURES]
        return last_sf.reshape((1, SpotifyDataset.SESSION_PREDICTABLE_FEATURES))

    @staticmethod
    def _pad_input(features):
        return np.pad(features, [(0, 10 - features.shape[0]), (0, 0)])

    @staticmethod
    def _pad_targets(skips):
        return np.pad(skips, [(0, 10 - skips.shape[0]), (0, 0)], constant_values=0)

    def call_on_batch(self, inputs):
        sf_second_half = self.pretrained_model.predict_on_batch(inputs[:-2]) if self.eval_on_predicted else inputs[5]
        sf_first_half = inputs[0]
        outputs = []
        for i in range(0, 10):
            shifted_inputs = self._get_nth_session_feature_window(sf_first_half, sf_second_half, i)
            network_output = self.network.predict_on_batch([shifted_inputs, inputs[1], inputs[2], inputs[4]])
            outputs.append(np.around(network_output[:, :]))
        return np.concatenate(outputs, 1)

    def train_on_batch(self, inputs, targets):
        sf_first_half = inputs[0]
        sf_second_half = self.pretrained_model.predict_on_batch(inputs[:-2]) if self.train_on_predicted else inputs[5]
        batch_len = sf_first_half.shape[0]
        loss, metric = 0, 0
        for i in range(10):
            shifted_inputs = self._get_nth_session_feature_window(sf_first_half, sf_second_half, i)
            shifted_targets = targets[:, i, :].reshape((batch_len, 1, 1))
            l, m = self.network.train_on_batch([shifted_inputs, inputs[1], inputs[2], inputs[4]], shifted_targets)
            loss += l
            metric += m
        return loss / 10.0, metric / 10.0

    def prepare_batch(self, batch):
        current_batch_size = min(self.batch_size, batch[DatasetDescription.SF_FIRST_HALF].shape[0])
        sfs_first_half = np.zeros((current_batch_size, 10, SpotifyDataset.SESSION_FEATURES), dtype=np.float32)
        sfs_second_half = np.zeros((current_batch_size, 10, SpotifyDataset.SESSION_FEATURES), dtype=np.float32)
        tfs_first_half = np.zeros((current_batch_size, 10, SpotifyDataset.TRACK_FEATURES), dtype=np.float32)
        tfs_second_half = np.zeros((current_batch_size, 10, SpotifyDataset.TRACK_FEATURES), dtype=np.float32)
        targets = np.zeros((current_batch_size, 10, 1), dtype=np.float32)
        last_sfs_first_half = np.zeros((current_batch_size, 1, SpotifyDataset.SESSION_PREDICTABLE_FEATURES), dtype=np.float32)
        last_skips_first_half = np.zeros((current_batch_size, 1, 1), dtype=np.float32)

        for i in range(current_batch_size):
            last_skips_first_half[i] = self._get_last_session_skips(batch[DatasetDescription.SF_FIRST_HALF][i])
            last_sfs_first_half[i] = self._get_last_session_features(batch[DatasetDescription.SF_FIRST_HALF][i])
            targets[i] = self._pad_targets(batch[DatasetDescription.SKIPS][i])
            sfs_first_half[i] = self._pad_input(batch[DatasetDescription.SF_FIRST_HALF][i])
            sfs_second_half[i] = self._pad_input(batch[DatasetDescription.SF_SECOND_HALF][i])
            tfs_first_half[i] = self._pad_input(batch[DatasetDescription.TF_FIRST_HALF][i])
            tfs_second_half[i] = self._pad_input(batch[DatasetDescription.TF_SECOND_HALF][i])

        return [sfs_first_half, tfs_first_half, tfs_second_half, last_sfs_first_half, last_skips_first_half, sfs_second_half], targets


if __name__ == "__main__":
    import argparse
    from predictor import Predictor

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_folder", default=".." + os.sep + ".." + os.sep + "small_train_set", type=str, help="Name of the train log folder.")
    parser.add_argument("--test_folder", default=".." + os.sep + ".." + os.sep + "mini_test_set", type=str, help="Name of the test log folder.")
    parser.add_argument("--tf_folder", default=".." + os.sep + "tf", type=str, help="Name of track features folder")
    parser.add_argument("--episodes", default=1, type=int, help="Number of episodes.")
    parser.add_argument("--batch_size", default=1024, type=int, help="Size of the batch.")
    parser.add_argument("--seed", default=0, type=int, help="Seed to use in numpy and tf.")
    parser.add_argument("--tf_preprocessor", default="MinMaxScaler", type=str, help="Name of the track features preprocessor to use.")
    parser.add_argument("--result_dir", default="results", type=str, help="Name of the results folder.")
    parser.add_argument("--model_name", default="finetuned_edsf_first_prediction", type=str, help="Name of the model to save.")
    parser.add_argument("--saved_weights_folder", default=".." + os.sep + "saved_models" + os.sep + "edsf_5" + os.sep + "encoder_decoder_sf", type=str, help="Name of the folder of saved lengths.")
    parser.add_argument("--train_on_predicted", default=True, type=bool, help="Whether to train using predicted sf for second half.")
    parser.add_argument("--eval_on_predicted", default=False, type=bool, help="Whether to evaluate using predicted sf for second half.")
    args = parser.parse_args()

    # no warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    logging.disable(logging.WARNING)

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    model = FinetunedEDSFFirstPredictionModel(args.batch_size, 100, args.saved_weights_folder, args.train_on_predicted, args.eval_on_predicted)

    predictor = Predictor(model, args.tf_preprocessor)
    predictor.train(args.episodes, args.train_folder, args.tf_folder)

    maa, fpa = predictor.evaluate(args.test_folder, args.tf_folder)

    model.save_model(args.result_dir + os.sep + args.model_name)

    print(str(args))
    print("Finetuned edsf first prediction model achieved " + str(maa) + " mean average accuracy")
    print("Finetuned edsf first prediction model achieved " + str(fpa) + " first prediction accuracy")
    print("------------------------------------")
