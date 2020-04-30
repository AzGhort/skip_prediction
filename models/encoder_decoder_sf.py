import tensorflow as tf
import numpy as np
from models.model import Model
from dataset_description import *
from spotify_dataset import SpotifyDataset
import os
from prediction_importances import *


# encoder-decoder architecture, predicting the whole session features
class EncoderDecoderSF(Model):
    def __init__(self, batch_size, weighted_loss=False):
        # SESSION REPRESENTATION
        # ---------------------------------------------------------------------------
        # session features
        sf_input = tf.keras.layers.Input(shape=(10, SpotifyDataset.SESSION_FEATURES), dtype=tf.float32, name="SF_Input")
        # sf_flatten = tf.keras.layers.Flatten(name="SF_Flatten")(sf_input)
        # sf_embed = tf.keras.layers.Embedding(2048, 32, name="SF_Embedding", mask_zero=True)(sf_flatten)
        sf_batch_norm = tf.keras.layers.BatchNormalization(name="SF_BatchNorm", autocast=False)(sf_input)
        sf_transformer = tf.keras.layers.Dense(64, activation=tf.nn.relu, name="SF_Transformer")(sf_batch_norm)

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
                                                                   output_shape=(None, 256))([
                                                                        session_representation,
                                                                        first_half_tf_transformer
                                                                        ])

        x = tf.keras.layers.Concatenate(name="EncoderInput")([
            first_half_session_representation,
            first_half_tf_transformer
        ])

        base_transformer = tf.keras.layers.Dense(256, activation=tf.nn.relu, name="BasicTransformer")
        encoder = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, return_state=True),
                                                name="Encoder")

        x = base_transformer(x)
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(x)

        forward_h = tf.keras.layers.Dropout(0.2)(forward_h)
        forward_c = tf.keras.layers.Dropout(0.2)(forward_c)
        backward_h = tf.keras.layers.Dropout(0.2)(backward_h)
        backward_c = tf.keras.layers.Dropout(0.2)(backward_c)

        # DECODER
        # ---------------------------------------------------------------------------
        previous_predicted_input = tf.keras.layers.Input(shape=(1, SpotifyDataset.SESSION_FEATURES),
                                                         name="PreviousWindowPredicted_SF")

        decoders = [
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, return_state=True), name="Decoder_0"),
            tf.keras.layers.LSTM(256, return_sequences=True, return_state=True, name="Decoder_1"),
            tf.keras.layers.Dropout(0.5, name="Decoder_2"),
            tf.keras.layers.Dense(SpotifyDataset.SESSION_FEATURES, activation=None, name="Decoder_3")
        ]

        all_outputs = []

        x = tf.keras.layers.Concatenate(name="DecoderInput_0")([
            tf.keras.layers.RepeatVector(1, name="SecondHalf_SessionRepresentation_0")(session_representation),
            tf.keras.layers.RepeatVector(1, name="SecondHalf_TF_0")(self._get_nth_lambda_layer(second_half_tf_transformer, 0))
        ])

        x = base_transformer(x)
        x, forward_h, forward_c, backward_h, backward_c = decoders[0](x, initial_state=[forward_h, forward_c, backward_h, backward_c])
        x = tf.keras.layers.Concatenate(name="ConcatenatedTo_SF_-1")([
            x,
            previous_predicted_input
        ])
        x, state_h, state_c = decoders[1](x)
        x = decoders[2](x)
        oup = decoders[3](x)
        all_outputs.append(oup)

        for i in range(1, 10):
            x = tf.keras.layers.Concatenate(name="DecoderInput_" + str(i))([
                tf.keras.layers.RepeatVector(1, name="SecondHalf_SessionRepresentation_" + str(i))(session_representation),
                tf.keras.layers.RepeatVector(1, name="SecondHalf_TF_" + str(i))(self._get_nth_lambda_layer(second_half_tf_transformer, i))
            ])

            x = base_transformer(x)
            x, forward_h, forward_c, backward_h, backward_c = decoders[0](x, initial_state=[forward_h, forward_c, backward_h, backward_c])
            x = tf.keras.layers.Concatenate(name="ConcatenatedTo_SF_" + str(i-1))([
                x,
                oup
            ])
            x, state_h, state_c = decoders[1](x, initial_state=[state_h, state_c])
            x = decoders[2](x)
            oup = decoders[3](x)
            all_outputs.append(oup)

        out_combined = tf.keras.layers.Lambda(lambda x: tf.keras.layers.Concatenate(axis=1)(x), name="Outputs")(all_outputs)

        self.network = tf.keras.Model(inputs=[sf_input, first_half_tf_input, second_half_tf_input, previous_predicted_input],
                                      outputs=[out_combined])
        self.batch_size = batch_size
        self.verbose_each = 10

        self.use_weighted_loss = weighted_loss
        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam()
        self.metric = tf.keras.metrics.BinaryCrossentropy(from_logits=False)

        self.network.compile(
            optimizer=self.optimizer,
            loss=self.mse_loss,
            metrics=[self.metric]
        )
        #os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
        #tf.keras.utils.plot_model(self.network, to_file='encoder_decoder_session_features_prediction.png')

    @staticmethod
    def _repeat_vector(args):
        layer_to_repeat = args[0]
        sequence_layer = args[1]
        return tf.keras.layers.RepeatVector(tf.shape(sequence_layer)[1], name="FeaturesRepeat")(layer_to_repeat)

    @staticmethod
    def _get_nth_lambda_layer(tensor, n):
        return tf.keras.layers.Lambda(lambda x: x[:, n])(tensor)

    @staticmethod
    def _pad_input(features):
        return np.pad(features, [(0, 10 - features.shape[0]), (0, 0)])

    @staticmethod
    def _get_last_session_features(sf_first_half):
        return sf_first_half[-1].reshape((1, SpotifyDataset.SESSION_FEATURES))

    def train(self, set):
        batch_index = 0
        for batch in set.batches(self.batch_size):
            current_batch_size = min(self.batch_size, batch[DatasetDescription.SF_FIRST_HALF].shape[0])
            batch_index += 1
            sfs_first_half = np.zeros((current_batch_size, 10, SpotifyDataset.SESSION_FEATURES), dtype=np.float32)
            tfs_first_half = np.zeros((current_batch_size, 10, SpotifyDataset.TRACK_FEATURES), dtype=np.float32)
            tfs_second_half = np.zeros((current_batch_size, 10, SpotifyDataset.TRACK_FEATURES), dtype=np.float32)
            sfs_second_half = np.zeros((current_batch_size, 10, SpotifyDataset.SESSION_FEATURES), dtype=np.float32)
            last_sfs_first_half = np.zeros((current_batch_size, 1, SpotifyDataset.SESSION_FEATURES), dtype=np.float32)

            for i in range(current_batch_size):
                last_sfs_first_half[i] = self._get_last_session_features(batch[DatasetDescription.SF_FIRST_HALF][i])
                sfs_second_half[i] = self._pad_input(batch[DatasetDescription.SF_SECOND_HALF][i])
                sfs_first_half[i] = self._pad_input(batch[DatasetDescription.SF_FIRST_HALF][i])
                tfs_first_half[i] = self._pad_input(batch[DatasetDescription.TF_FIRST_HALF][i])
                tfs_second_half[i] = self._pad_input(batch[DatasetDescription.TF_SECOND_HALF][i])

            loss, metric = self.train_on_batch([sfs_first_half, tfs_first_half, tfs_second_half, last_sfs_first_half], sfs_second_half)
            if batch_index % self.verbose_each == 0:
                print("--- loss of batch number " + str(batch_index) + ": " + str(loss))
                print("--- metric of batch number " + str(batch_index) + ": " + str(metric))

    def train_on_batch(self, inputs, targets):
        current_batch_size = inputs.shape[0]
        with tf.GradientTape() as tape:
            predicted = self.network(inputs)
            if self.use_weighted_loss:
                pos_weights = np.zeros((current_batch_size, 10, SpotifyDataset.SESSION_FEATURES), dtype=np.float32)
                for i in range(current_batch_size):
                    pos_weights[i] = Importances
                loss = tf.nn.weighted_cross_entropy_with_logits(targets, predicted, pos_weights)
            else:
                loss = self.mse_loss(targets, predicted)

        self.metric.reset_states()
        self.metric.update_state(targets, predicted)
        metric = self.metric.result()

        gradients = tape.gradient(loss, self.network.variables)
        self.optimizer.apply_gradients(zip(gradients, self.network.variables))

        return loss, metric

    def __call__(self, sf_first, sf_second, tf_first, tf_second):
        second_half_real_length = sf_second.shape[0]
        last_sf_first_half = self._get_last_session_features(sf_first).reshape((1, 1, SpotifyDataset.SESSION_FEATURES))
        sf_first = self._pad_input(sf_first).reshape((1, 10, SpotifyDataset.SESSION_FEATURES))
        tf_first = self._pad_input(tf_first).reshape((1, 10, SpotifyDataset.TRACK_FEATURES))
        tf_second = self._pad_input(tf_second).reshape((1, 10, SpotifyDataset.TRACK_FEATURES))
        network_output = self.network([sf_first, tf_first, tf_second, last_sf_first_half]).numpy()
        predictions = []
        assert network_output.shape == (1, 10, SpotifyDataset.SESSION_FEATURES)
        for i in range(second_half_real_length):
            predictions.append(np.around(network_output[0][i][2]))
        return predictions

    def save_model(self, file):
        self.network.save_weights(file)


if __name__ == "__main__":
    import argparse
    from predictor import Predictor

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_folder", default=".." + os.sep + ".." + os.sep + "one_file_train_set", type=str, help="Name of the train log folder.")
    parser.add_argument("--test_folder", default=".." + os.sep + ".." + os.sep + "one_file_test_set", type=str, help="Name of the test log folder.")
    parser.add_argument("--tf_folder", default="." + os.sep + "tf", type=str, help="Name of track features folder")
    parser.add_argument("--episodes", default=1, type=int, help="Number of episodes.")
    parser.add_argument("--batch_size", default=2048, type=int, help="Size of the batch.")
    parser.add_argument("--seed", default=0, type=int, help="Seed to use in numpy and tf.")
    parser.add_argument("--tf_preprocessor", default="MinMaxScaler", type=str, help="Name of the track features preprocessor to use.")
    parser.add_argument("--sf_preprocessor", default="NonePreprocessor", type=str, help="Name of the session features preprocessor to use.")
    parser.add_argument("--result_dir", default="results", type=str, help="Name of the results folder.")
    parser.add_argument("--model_name", default="encoder_decoder_sf", type=str, help="Name of the model to save.")
    parser.add_argument("--weighted_loss", default=True, type=bool, help="Whether to use weighted loss for training.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    model = EncoderDecoderSF(args.batch_size, args.weighted_loss)
    predictor = Predictor(model, args.tf_preprocessor, args.sf_preprocessor)
    predictor.train(args.episodes, args.train_folder, args.tf_folder)
    maa, fpa = predictor.evaluate(args.test_folder, args.tf_folder)

    model.save_model(args.result_dir + os.sep + args.model_name)

    print(str(args))
    print("Encoder-decoder session features prediction model achieved " + str(maa) + " mean average accuracy")
    print("Encoder-decoder session features prediction model achieved " + str(fpa) + " first prediction accuracy")
    print("------------------------------------")
