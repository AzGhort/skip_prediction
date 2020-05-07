from models.network_model import NetworkModel
from models.encoder_decoder_sf import EncoderDecoderSF


class FinetunedEncoderDecoderModel(NetworkModel):
    def __init__(self, batch_size, verbose_each, saved_model_file):
        self.batch_size = batch_size
        self.verbose_each = verbose_each

        ed = EncoderDecoderSF(batch_size, verbose_each)
        self.encoder_level_bidirs = ed.encoder_level_bidirs
        self.pretrained_model = ed.network

        self.pretrained_model.load_weights(saved_model_file)
        for layer in self.pretrained_model.layers:
            layer.trainable = False
        #self.pretrained_model.summary()

    def save_model(self, file):
        self.network.save_weights(file)
