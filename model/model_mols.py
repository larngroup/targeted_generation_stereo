# from keras.models import Sequential
# from keras.layers import LSTM, Dense
# from keras.initializers import RandomNormal
# from model.Smiles_to_tokens import SmilesToTokens
# from utils.utils import Utils
# import tensorflow 

class BaseModel(object):
    def __init__(self, config):
        self.config = config
        self.model = None
        self.token_table = Utils().table

class Model_mols(BaseModel):
    """
    Constructor for the Generator model
    Parameters
    ----------
    Returns
    -------
    This function initializes the architecture for the Generator
    """
    def __init__(self, config):
        super(Model_mols, self).__init__(config)
        self.build_model()
        self.token_table = Utils().table

    def build_model(self):
        input_data = tensorflow.keras.layers.Input(shape=(self.config.paddSize,)) 
        encoder = tensorflow.keras.layers.Embedding(len(self.token_table),self.config.embedding_dim,input_length = self.config.paddSize) (input_data)
        encoder = tensorflow.keras.layers.LSTM(128, dropout=0.1, 
                                               input_shape=(self.config.paddSize,self.config.embedding_dim),
                                               return_sequences=True) (encoder)        
        encoder = tensorflow.keras.layers.LSTM(128, dropout=0.0, 
                                               input_shape=(self.config.paddSize,128),
                                               return_sequences=False) (encoder)  
        
        self.distribution_mean = tensorflow.keras.layers.Dense(self.config.latent_dim, name='mean')(encoder)
        self.distribution_variance = tensorflow.keras.layers.Dense(self.config.latent_dim, name='log_variance')(encoder)
        self.distribution =  [self.distribution_mean, self.distribution_variance]
        latent_encoding = tensorflow.keras.layers.Lambda(self._sample_latent_features)(self.distribution)
        self.encoder_model = tensorflow.keras.Model(input_data, latent_encoding)
        self.encoder_model.summary()
        
        decoder_input = tensorflow.keras.layers.Input(shape=(self.config.latent_dim))
        decoder = tensorflow.keras.layers.RepeatVector(self.config.paddSize)(decoder_input)
        decoder = tensorflow.keras.layers.LSTM(128, dropout=0.0, 
                                               input_shape=(self.config.paddSize,self.config.latent_dim),
                                               return_sequences=True) (decoder)  
        decoder = tensorflow.keras.layers.LSTM(128, dropout=0.1, 
                                               input_shape=(self.config.paddSize,128),
                                               return_sequences=True) (decoder)
        decoder_output = tensorflow.keras.layers.TimeDistributed(tensorflow.keras.layers.Dense(1)) (decoder)
        
                
        self.decoder_model = tensorflow.keras.Model(decoder_input, decoder_output)
        self.decoder_model.summary()
        
        encoded = self.encoder_model(input_data)
        decoded = self.decoder_model(encoded)
        
        self.autoencoder = tensorflow.keras.models.Model(input_data, decoded)
        
    def _sample_latent_features(self,distribution):
        distribution_mean, distribution_variance = distribution
        batch_size = tensorflow.shape(distribution_variance)[0]
        random = tensorflow.keras.backend.random_normal(shape=(batch_size, tensorflow.shape(distribution_variance)[1]))
        return distribution_mean + tensorflow.exp(0.5 * distribution_variance) * random

    def _get_loss(self,distribution_mean, distribution_variance):
        
        def get_reconstruction_loss(y_true, y_pred):
            reconstruction_loss = tensorflow.keras.losses.mse(y_true, y_pred)
            reconstruction_loss_batch = tensorflow.reduce_mean(reconstruction_loss)
            return reconstruction_loss_batch
        
        def get_kl_loss(distribution_mean, distribution_variance):
            kl_loss = 1 + distribution_variance - tensorflow.square(distribution_mean) - tensorflow.exp(distribution_variance)
            kl_loss_batch = tensorflow.reduce_mean(kl_loss)
            return kl_loss_batch*(-0.5)
        
        def total_loss(y_true, y_pred):
            reconstruction_loss_batch = get_reconstruction_loss(y_true, y_pred)
            kl_loss_batch = get_kl_loss(distribution_mean, distribution_variance)
            return reconstruction_loss_batch + kl_loss_batch
        
        return total_loss
