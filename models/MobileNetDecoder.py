import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, Union
from models.Subpixel import Subpixel

class MobileNetDecoder(tf.keras.models.Model):
    def __init__(
            self, 
            shape:Tuple[int, int, int]=(128, 128, 3),
            use_prelu:bool=False,
            dropout:Optional[Union[bool, float]]=.2
            ) -> None:
        super(MobileNetDecoder, self).__init__()
        self.shape = shape
        basemodel = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=self.shape,
        )
        #basemodel.trainable = False
        self.encoder = tf.keras.Sequential([
            basemodel,
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten()
        ], name='encoder')
        self.encoder_shape = tf.keras.backend.int_shape(self.encoder.layers[-2].output)
        
        decoder_layers = [
            tf.keras.layers.Reshape(
            self.encoder_shape[1:],
            input_shape=(np.prod(self.encoder_shape[1:]),))
        ]
        for i in range(1, int(np.log2(self.shape[1]))-int(np.log2(self.encoder_shape[1]))+1):
            if i == int(np.log2(self.shape[1]))-int(np.log2(self.encoder_shape[1])):
                decoder_layers += [
                    Subpixel(filters=self.shape[2], kernel_size=(3,3), r=2, padding='same'),
                    # Subpixel(filters=128//(2**(i-1)), kernel_size=(3,3), r=2, padding='same'),
                    # tf.keras.layers.Conv2DTranspose(self.shape[2], (3, 3), padding='same'),
                    tf.keras.layers.Activation('sigmoid')
                ]
                break
            decoder_layers += [
                Subpixel(filters=128//(2**i), kernel_size=(3,3), r=2, padding='same'),
                tf.keras.layers.BatchNormalization(axis=-1)
            ]
            if use_prelu:
                decoder_layers += [tf.keras.layers.PReLU()]
            else:
                decoder_layers += [tf.keras.layers.Activation('relu')]
            if dropout:
                decoder_layers += [tf.keras.layers.SpatialDropout2D(dropout)]
        self.decoder = tf.keras.Sequential(decoder_layers, name='decoder')
        
    def encode_decode_summary(self) -> None:
        self.encoder.summary()
        self.decoder.summary()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)
    
    def call(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded
