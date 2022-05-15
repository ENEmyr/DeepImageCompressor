import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, Union
from models.Subpixel import Subpixel

class SSDCAE(tf.keras.models.Model):
    def __init__(
            self, 
            filters:Tuple=(16, 16),
            pre_filters:Tuple=(8, 8),
            shape:Tuple[int, int, int]=(128, 128, 3),
            use_prelu:bool=False,
            dropout:Optional[Union[bool, float]]=.2
            ) -> None:
        """
        Subpixel Stacked-Denoising Convolutional Autoencoder
        This module is automatically trained when in model.training is True.
        Args:
            filters: The tuple of filters which is required as an argument for Conv layers
            pre_filters: The tuple of pre-filters which is required as an argument for Conv layers that lay before Downsample layers length of it much equal to `filters`
            shape: The tuple of image shape that use as a sample for the network
            use_prelu: Use PReLU as activation after down/up sample or just use normal ReLU (PReLU will make more parameters)
            dropout: Dropout probability for dropout2d layer
        """
        super(SSDCAE, self).__init__(name='SSDCAE')
        assert len(filters) == len(pre_filters)
        self.filters = filters
        self.pre_filters = pre_filters
        self.shape = shape
        encoder_layers = [ tf.keras.layers.Input(shape=self.shape) ]
        for i, (f, p) in enumerate(zip(self.filters, self.pre_filters)):
            encoder_layers += [
                tf.keras.layers.Conv2D(p, kernel_size=(3, 3), strides=1, padding='same', name=f'Presampling{i+1}'),
                tf.keras.layers.Conv2D(f, kernel_size=(3, 3), strides=2, padding='same', name=f'Downsample{i+1}'),
                tf.keras.layers.BatchNormalization(axis=-1)
            ]
            if use_prelu:
                encoder_layers += [tf.keras.layers.PReLU()]
            else:
                encoder_layers += [tf.keras.layers.Activation('relu')]
            if dropout:
                encoder_layers += [tf.keras.layers.Dropout(dropout)]
        else:
            encoder_layers += [tf.keras.layers.Flatten()]
        self.encoder = tf.keras.Sequential(encoder_layers, name='encoder')
        self.encoder_shape = tf.keras.backend.int_shape(self.encoder.layers[-2].output)
        
        decoder_layers = [
            tf.keras.layers.Reshape(
            self.encoder_shape[1:],
            input_shape=(np.prod(self.encoder_shape[1:]),))
        ]
        for i, f in enumerate(self.filters[::-1]):
            decoder_layers += [
                Subpixel(filters=f, kernel_size=(3,3), r=2, padding='same', name=f'Upsample{len(self.filters)-i}'),
                tf.keras.layers.BatchNormalization(axis=-1)
            ]
            if use_prelu:
                decoder_layers += [tf.keras.layers.PReLU()]
            else:
                decoder_layers += [tf.keras.layers.Activation('relu')]
            if dropout:
                decoder_layers += [tf.keras.layers.Dropout(dropout)]
            if i == len(self.filters)-1:
                decoder_layers += [
                    tf.keras.layers.Conv2DTranspose(self.shape[2], (3, 3), padding='same'),
                    tf.keras.layers.Activation('sigmoid')
                ]
                break
        self.decoder = tf.keras.Sequential(decoder_layers, name='decoder')
        
    def encode_decode_summary(self) -> None:
        self.encoder.summary()
        self.decoder.summary()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)
    
    @tf.function
    def call(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded

    def get_config(self):
        return {
            "name": "SSDCAE",
            "encoder": self.encoder,
            "decoder": self.decoder,
        }
