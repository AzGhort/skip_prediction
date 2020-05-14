from models.network_model import NetworkModel
from models.encoder_decoder_sf import EncoderDecoderSF


class FinetunedEncoderDecoderModel(NetworkModel):
    def __init__(self, batch_size, verbose_each, saved_model_file, trainable_decoder=False):
        self.batch_size = batch_size
        self.verbose_each = verbose_each

        ed = EncoderDecoderSF(batch_size, verbose_each)
        self.encoder_level_bidirs = ed.encoder_level_bidirs
        self.encoder_out_states = ed.encoder_out_states
        self.session_representation = ed.session_representation
        self.second_half_tf_transformer = ed.second_half_tf_transformer
        self.pretrained_model = ed.network

        if saved_model_file is not None:
            self.pretrained_model.load_weights(saved_model_file)
        for layer in self.pretrained_model.layers:
            if not trainable_decoder:
                layer.trainable = False
            # only encoder and embedding should be nontrainable
            elif not layer.name.startswith("Decoder"):
                layer.trainable = False
        #self.pretrained_model.summary()

    def save_model(self, file):
        self.network.save_weights(file)
