import numpy as np
from models.skip_prediction_nn_model import SkipPredictionNNModel
from dataset_description import *
from spotify_dataset import SpotifyDataset
import os
import tensorflow as tf
from prediction_importances import *
import logging
from preprocessing.session_feature_stats import *


# encoder-decoder architecture, predicting the session features 2 - 18
class EncoderDecoderSF(SkipPredictionNNModel):
    def __init__(self, batch_size, verbose_each=10):
        # SESSION REPRESENTATION
        # ---------------------------------------------------------------------------
        # session features
        sf_input = tf.keras.layers.Input(shape=(10, SpotifyDataset.SESSION_FEATURES), dtype=tf.float32, name="SF_Input")
        # sf_flatten = tf.keras.layers.Flatten(name="SF_Flatten")(sf_input)
        # sf_embed = tf.keras.layers.Embedding(2048, 32, name="SF_Embedding", mask_zero=True)(sf_flatten)
        sf_batch_norm = tf.keras.layers.BatchNormalization(name="SF_BatchNorm", autocast=False)(sf_input)
        sf_transformer = tf.keras.layers.Dense(64, activation=tf.nn.relu, name="SF_Transformer")(sf_batch_norm)
        self.sf_batch_norm = sf_batch_norm

        # track features
        # use same embedding for first and second half track features
        # tf_flatten = tf.keras.layers.Flatten(name="TF_Flatten")
        # tf_embed = tf.keras.layers.Embedding(2048, 32, name="TF_Embedding", mask_zero=True)
        tf_batch_norm = tf.keras.layers.BatchNormalization(name="TF_BatchNorm", autocast=False)
        tf_transformer = tf.keras.layers.Dense(64, activation=tf.nn.relu, name="TF_Transformer")

        # first half tf
        first_half_tf_input = tf.keras.layers.Input(shape=(10, SpotifyDataset.TRACK_FEATURES), dtype=tf.float32, name="FirstHalf_TF_Input")
        # first_half_tf_flatten = tf_flatten(first_half_tf_input)
        # first_half_tf_embed = tf_embed(first_half_tf_flatten)
        first_half_tf_batch_norm = tf_batch_norm(first_half_tf_input)
        first_half_tf_transformer = tf_transformer(first_half_tf_batch_norm)

        # second half tf
        second_half_tf_input = tf.keras.layers.Input(shape=(10, SpotifyDataset.TRACK_FEATURES), dtype=tf.float32, name="SecondHalf_TF_Input")
        # second_half_tf_flatten = tf_flatten(second_half_tf_input)
        # second_half_tf_embed = tf_embed(second_half_tf_flatten)
        second_half_tf_batch_norm = tf_batch_norm(second_half_tf_input)
        second_half_tf_transformer = tf_transformer(second_half_tf_batch_norm)

        # representation
        first_half_features = tf.keras.layers.Concatenate(axis=1, name="FirstHalf_Concatenate")([
            sf_transformer,
            first_half_tf_transformer
        ])
        sf_bidir = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64), name="SF_Bidirectional")(first_half_features)

        tf_features = tf.keras.layers.Concatenate(axis=1, name="SecondHalfConcatenate")([
            first_half_tf_transformer,
            second_half_tf_transformer
        ])
        tf_bidir = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64), name="TF_Bidirectional")(tf_features)
        session_representation = tf.keras.layers.Concatenate(name="SessionRepresentation")([
            sf_bidir,
            tf_bidir
        ])

        # ENCODER
        # ---------------------------------------------------------------------------
        first_half_session_representation = tf.keras.layers.Lambda(self._repeat_vector, name="FirstHalfRepresentation",
                                                                   output_shape=(None, 256))([session_representation, 10])

        x = tf.keras.layers.Concatenate(name="EncoderInput")([
            first_half_session_representation,
            first_half_tf_transformer
        ])

        base_transformer = tf.keras.layers.Dense(256, activation=tf.nn.relu, name="BasicTransformer")
        encoder = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, return_state=True),
                                                name="Encoder")

        x = base_transformer(x)
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(x)

        forward_h = tf.keras.layers.Dropout(0.2, name="Forward_Hidden_Dropout")(forward_h)
        forward_c = tf.keras.layers.Dropout(0.2, name="Forward_Dropout")(forward_c)
        backward_h = tf.keras.layers.Dropout(0.2, name="Backward_Hidden_Dropout")(backward_h)
        backward_c = tf.keras.layers.Dropout(0.2, name="Backward_Dropout")(backward_c)

        self.encoder_out_states = [forward_h, forward_c, backward_h, backward_c]

        # DECODER
        # ---------------------------------------------------------------------------
        previous_predicted_input = tf.keras.layers.Input(shape=(1, SpotifyDataset.SESSION_PREDICTABLE_FEATURES),
                                                         name="PreviousWindowPredicted_SF")

        decoders = [
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, return_state=True), name="Decoder_0"),
            tf.keras.layers.LSTM(256, return_sequences=True, return_state=True, name="Decoder_1"),
            tf.keras.layers.Dropout(0.5, name="Decoder_2"),
            tf.keras.layers.Dense(SpotifyDataset.SESSION_PREDICTABLE_FEATURES, activation=None, name="Decoder_3")
        ]

        all_predictions = []
        encoder_level_bidirs = []

        x = tf.keras.layers.Concatenate(name="DecoderInput_0")([
            tf.keras.layers.RepeatVector(1, name="DecoderSecondHalf_SessionRepresentation_0")(session_representation),
            tf.keras.layers.RepeatVector(1, name="DecoderSecondHalf_TF_0")(self._get_nth_lambda_layer(second_half_tf_transformer, 0))
        ])

        self.session_representation = session_representation
        self.second_half_tf_transformer = second_half_tf_transformer

        x = base_transformer(x)
        x, forward_h, forward_c, backward_h, backward_c = decoders[0](x, initial_state=[forward_h, forward_c, backward_h, backward_c])
        encoder_level_bidirs.append(x)
        x = tf.keras.layers.Concatenate(name="DecoderConcatenatedTo_SF_-1")([
            x,
            previous_predicted_input
        ])
        x, state_h, state_c = decoders[1](x)
        x = decoders[2](x)
        prediction = decoders[3](x)
        all_predictions.append(prediction)

        for i in range(1, 10):
            x = tf.keras.layers.Concatenate(name="DecoderInput_" + str(i))([
                tf.keras.layers.RepeatVector(1, name="DecoderSecondHalf_SessionRepresentation_" + str(i))(session_representation),
                tf.keras.layers.RepeatVector(1, name="DecoderSecondHalf_TF_" + str(i))(self._get_nth_lambda_layer(second_half_tf_transformer, i))
            ])

            x = base_transformer(x)
            x, forward_h, forward_c, backward_h, backward_c = decoders[0](x, initial_state=[forward_h, forward_c, backward_h, backward_c])
            encoder_level_bidirs.append(x)
            x = tf.keras.layers.Concatenate(name="DecoderConcatenatedTo_SF_" + str(i-1))([
                x,
                prediction
            ])
            x, state_h, state_c = decoders[1](x, initial_state=[state_h, state_c])
            x = decoders[2](x)
            prediction = decoders[3](x)
            all_predictions.append(prediction)

        predictions_combined = tf.keras.layers.Lambda(lambda x: tf.keras.layers.Concatenate(axis=1)(x), name="DecoderSessionFeatures_Predictions")(all_predictions)
        self.encoder_level_bidirs = encoder_level_bidirs

        self.network = tf.keras.Model(inputs=[sf_input, first_half_tf_input, second_half_tf_input, previous_predicted_input],
                                      outputs=[predictions_combined])

        super().__init__(batch_size, verbose_each)

        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam()
        self.metric = tf.keras.metrics.MeanSquaredError()

        self.network.compile(
            optimizer=self.optimizer,
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanAbsolutePercentageError()]
        )
        #os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
        #tf.keras.utils.plot_model(self.network, to_file='encoder_decoder_session_features_prediction.png')

    @staticmethod
    def _repeat_vector(args):
        layer_to_repeat = args[0]
        repeat_count = args[1]
        return tf.keras.layers.RepeatVector(repeat_count, name="FeaturesRepeat")(layer_to_repeat)

    @staticmethod
    def _get_nth_lambda_layer(tensor, n):
        return tf.keras.layers.Lambda(lambda x: x[:, n])(tensor)

    @staticmethod
    def _pad_input(features):
        return np.pad(features, [(0, 10 - features.shape[0]), (0, 0)])

    @staticmethod
    def _pad_targets(features):
        features = features[:, :SpotifyDataset.SESSION_PREDICTABLE_FEATURES]
        return np.pad(features, [(0, 10 - features.shape[0]), (0, 0)])

    @staticmethod
    def _upscale_session_features(sf):
        sf[:, :, :10] = np.around(sf[:, :, :10])
        sf[:, :, 10] = np.around(np.exp(sf[:, :, 10] * LogMaximums[SessionFeaturesFields.HIST_USER_BEHAVIOR_N_SEEKFWD]) - 1)
        sf[:, :, 11] = np.around(np.exp(sf[:, :, 11] * LogMaximums[SessionFeaturesFields.HIST_USER_BEHAVIOR_N_SEEKBACK]) - 1)
        sf[:, :, 12] = np.around(sf[:, :, 12])
        sf[:, :, 13] = np.around(sf[:, :, 13] * Maximums[SessionFeaturesFields.HOUR_OF_DAY])
        sf[:, :, 14] = np.around(sf[:, :, 14])
        sf[:, :, 15] = np.around(sf[:, :, 15] * Maximums[SessionFeaturesFields.CONTEXT_TYPE])

    @staticmethod
    def _upscale_targets(sf):
        sf[:][:][10] = np.around(np.exp(sf[:][:][10] * LogMaximums[SessionFeaturesFields.HIST_USER_BEHAVIOR_N_SEEKFWD])-1)
        sf[:][:][11] = np.around(np.exp(sf[:][:][11] * LogMaximums[SessionFeaturesFields.HIST_USER_BEHAVIOR_N_SEEKBACK])-1)
        sf[:][:][13] = np.around(sf[:][:][13] * Maximums[SessionFeaturesFields.HOUR_OF_DAY])
        sf[:][:][15] = np.around(sf[:][:][15] * Maximums[SessionFeaturesFields.CONTEXT_TYPE])

    @staticmethod
    def _get_last_session_features(sf_first_half):
        last_sf = sf_first_half[-1, :SpotifyDataset.SESSION_PREDICTABLE_FEATURES]
        return last_sf.reshape((1, SpotifyDataset.SESSION_PREDICTABLE_FEATURES))

    def prepare_batch(self, batch):
        current_batch_size = min(self.batch_size, batch[DatasetDescription.SF_FIRST_HALF].shape[0])
        sfs_first_half = np.zeros((current_batch_size, 10, SpotifyDataset.SESSION_FEATURES), dtype=np.float32)
        tfs_first_half = np.zeros((current_batch_size, 10, SpotifyDataset.TRACK_FEATURES), dtype=np.float32)
        tfs_second_half = np.zeros((current_batch_size, 10, SpotifyDataset.TRACK_FEATURES), dtype=np.float32)
        targets = np.zeros((current_batch_size, 10, SpotifyDataset.SESSION_PREDICTABLE_FEATURES), dtype=np.float32)
        last_sfs_first_half = np.zeros((current_batch_size, 1, SpotifyDataset.SESSION_PREDICTABLE_FEATURES), dtype=np.float32)

        for i in range(current_batch_size):
            last_sfs_first_half[i] = self._get_last_session_features(batch[DatasetDescription.SF_FIRST_HALF][i])
            targets[i] = self._pad_targets(batch[DatasetDescription.SF_SECOND_HALF][i])
            sfs_first_half[i] = self._pad_input(batch[DatasetDescription.SF_FIRST_HALF][i])
            tfs_first_half[i] = self._pad_input(batch[DatasetDescription.TF_FIRST_HALF][i])
            tfs_second_half[i] = self._pad_input(batch[DatasetDescription.TF_SECOND_HALF][i])

        return [sfs_first_half, tfs_first_half, tfs_second_half, last_sfs_first_half], targets

    def predict_all_on_batch(self, batch_input):
        batch_len = batch_input[0].shape[0]
        network_output = self.network.predict_on_batch(batch_input)
        return network_output.reshape((batch_len, 10, SpotifyDataset.SESSION_PREDICTABLE_FEATURES))

    def evaluate_all_feature_accuracies(self, set):
        accuracies = [[[] for _ in range(10)] for _ in range(SpotifyDataset.SESSION_PREDICTABLE_FEATURES)]
        mean_accuracies = [[0 for _ in range(10)] for _ in range(SpotifyDataset.SESSION_PREDICTABLE_FEATURES)]
        for batch in set.batches(self.batch_size):
            actual_features = batch[DatasetDescription.SF_SECOND_HALF]
            x, _ = self.prepare_batch(batch)
            predicted_features = self.predict_all_on_batch(x)
            self.append_all_features_accuracies(predicted_features, actual_features, accuracies)
        for feature_index in range(SpotifyDataset.SESSION_PREDICTABLE_FEATURES):
            for song_index in range(10):
                mean_accuracies[feature_index][song_index] = np.mean(accuracies[feature_index][song_index])
        return mean_accuracies

    def append_all_features_accuracies(self, predicted, actual, accuracies):
        self._upscale_session_features(predicted)
        self._upscale_targets(actual)
        batch_len = predicted[0].shape[0]
        for batch_index in range(batch_len):
            for session_index in range(actual[batch_index].shape[0]):
                for feature_index in range(SpotifyDataset.SESSION_PREDICTABLE_FEATURES):
                    act = actual[batch_index][session_index][feature_index]
                    pred = predicted[batch_index][session_index][feature_index]
                    if act == pred:
                        accuracies[feature_index][session_index].append(1.0)
                    else:
                        accuracies[feature_index][session_index].append(0.0)

    def call_on_batch(self, batch_input):
        batch_len = batch_input[0].shape[0]
        network_output = self.network.predict_on_batch(batch_input)
        return np.around(network_output[:, :, 2]).reshape((batch_len, 10, 1))

    def train_on_batch(self, inputs, targets):
        return self.network.train_on_batch(inputs, targets)

    def __call__(self, sf_first, sf_second, tf_first, tf_second):
        second_half_real_length = sf_second.shape[0]
        last_sf_first_half = self._get_last_session_features(sf_first).reshape((1, 1, SpotifyDataset.SESSION_PREDICTABLE_FEATURES))
        sf_first = self._pad_input(sf_first).reshape((1, 10, SpotifyDataset.SESSION_FEATURES))
        tf_first = self._pad_input(tf_first).reshape((1, 10, SpotifyDataset.TRACK_FEATURES))
        tf_second = self._pad_input(tf_second).reshape((1, 10, SpotifyDataset.TRACK_FEATURES))
        network_output = self.network([sf_first, tf_first, tf_second, last_sf_first_half]).numpy()
        predictions = []
        assert network_output.shape == (1, 10, SpotifyDataset.SESSION_PREDICTABLE_FEATURES)
        for i in range(second_half_real_length):
            predictions.append(np.around(network_output[0][i][2]))
        return predictions

    def save_model(self, file):
        self.network.save_weights(file)


if __name__ == "__main__":
    import argparse
    from predictor import Predictor

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_folder", default=".." + os.sep + ".." + os.sep + "example_set", type=str, help="Name of the train log folder.")
    parser.add_argument("--test_folder", default=".." + os.sep + ".." + os.sep + "mini_test_set", type=str, help="Name of the test log folder.")
    parser.add_argument("--tf_folder", default=".." + os.sep + "tf", type=str, help="Name of track features folder")
    parser.add_argument("--episodes", default=1, type=int, help="Number of episodes.")
    parser.add_argument("--batch_size", default=2048, type=int, help="Size of the batch.")
    parser.add_argument("--seed", default=0, type=int, help="Seed to use in numpy and tf.")
    parser.add_argument("--tf_preprocessor", default="MinMaxScaler", type=str, help="Name of the track features preprocessor to use.")
    parser.add_argument("--result_dir", default="results", type=str, help="Name of the results folder.")
    parser.add_argument("--model_name", default="encoder_decoder_sf", type=str, help="Name of the model to save.")
    parser.add_argument("--saved_weights_folder", default=".." + os.sep + "saved_models" + os.sep + "edsf_5" + os.sep + "encoder_decoder_sf", type=str, help="Name of the folder of saved lengths.")
    args = parser.parse_args()

    # no warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    logging.disable(logging.WARNING)

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    model = EncoderDecoderSF(args.batch_size, 10)
    if args.saved_weights_folder is not None:
        model.network.load_weights(args.saved_weights_folder)

    predictor = Predictor(model, args.tf_preprocessor)

    predictor.train(args.episodes, args.train_folder, args.tf_folder)
    maa, fpa = predictor.evaluate(args.test_folder, args.tf_folder)

    model.save_model(args.result_dir + os.sep + args.model_name)

    print(str(args))
    print("Encoder-decoder session features prediction model achieved " + str(maa) + " mean average accuracy")
    print("Encoder-decoder session features prediction model achieved " + str(fpa) + " first prediction accuracy")
    print("------------------------------------")
