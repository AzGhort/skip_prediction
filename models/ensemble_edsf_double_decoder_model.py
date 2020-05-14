from models.network_model import NetworkModel
from models.finetuned_edsf_double_decoder_model import FinetunedEDSFDoubleDecoderModel
import numpy as np
from dataset_description import DatasetDescription
from spotify_dataset import SpotifyDataset
import os
import tensorflow as tf


class EnsembleEdsfDoubleDecoderModel(NetworkModel):
    def __init__(self, batch_size, weighted_loss):
        model_a = FinetunedEDSFDoubleDecoderModel(batch_size, 10, None, weighted_loss)
        model_a.network.load_weights(".." + os.sep + "saved_models" + os.sep + "edsf_dd_one_a_1_unweighted" + os.sep + "finetuned_edsf_double_decoder")
        model_b = FinetunedEDSFDoubleDecoderModel(batch_size, 10, None, weighted_loss)
        model_b.network.load_weights(".." + os.sep + "saved_models" + os.sep + "edsf_dd_one_b_1_unweighted" + os.sep + "finetuned_edsf_double_decoder")
        model_c = FinetunedEDSFDoubleDecoderModel(batch_size, 10, None, weighted_loss)
        model_c.network.load_weights(".." + os.sep + "saved_models" + os.sep + "edsf_dd_one_c_1_unweighted" + os.sep + "finetuned_edsf_double_decoder")
        model_d = FinetunedEDSFDoubleDecoderModel(batch_size, 10, None, weighted_loss)
        model_d.network.load_weights(".." + os.sep + "saved_models" + os.sep + "edsf_dd_one_d_1_unweighted" + os.sep + "finetuned_edsf_double_decoder")
        model_e = FinetunedEDSFDoubleDecoderModel(batch_size, 10, None, weighted_loss)
        model_e.network.load_weights(".." + os.sep + "saved_models" + os.sep + "edsf_dd_one_test_1_unweighted" + os.sep + "finetuned_edsf_double_decoder")

        self.batch_size = batch_size
        self.weighted_loss = weighted_loss
        self.models = [model_a.network, model_b.network, model_c.network, model_d.network, model_e.network]

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
        network_outputs = [np.around(model.predict_on_batch(batch_input)) for model in self.models]

        ensembled_outputs = np.zeros((batch_len, 10, 1))
        for session_index in range(batch_len):
            session_outputs = [model_out[session_index] for model_out in network_outputs]
            for track_index in range(10):
                ensembled_outputs[session_index][track_index] = np.around(np.mean([so[track_index] for so in session_outputs]))

        return ensembled_outputs

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
    parser.add_argument("--test_folder", default=".." + os.sep + ".." + os.sep + "mini_test_set", type=str, help="Name of the test log folder.")
    parser.add_argument("--tf_folder", default=".." + os.sep + "tf", type=str, help="Name of track features folder")
    parser.add_argument("--batch_size", default=2048, type=int, help="Size of the batch.")
    parser.add_argument("--tf_preprocessor", default="MinMaxScaler", type=str, help="Name of the track features preprocessor to use.")
    parser.add_argument("--weighted_loss", default=False, type=bool, help="Whether to use weighted loss.")
    args = parser.parse_args()

    model = EnsembleEdsfDoubleDecoderModel(args.batch_size, args.weighted_loss)
    predictor = Predictor(model, args.tf_preprocessor)
    maa, fpa = predictor.evaluate(args.test_folder, args.tf_folder)

    print("Finetuned edsf double decoder prediction model achieved " + str(maa) + " mean average accuracy")
    print("Finetuned edsf double decoder prediction model achieved " + str(fpa) + " first prediction accuracy")
    print("------------------------------------")
