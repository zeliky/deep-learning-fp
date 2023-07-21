import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Attention, TimeDistributed
from models.options import ModelOptions


class CnnLstmAttentionModel:
    def __init__(self, options: ModelOptions):
        self.options = options

    def get_model(self, batch_size=None):

        if batch_size is None:
            batch_size = self.options.batch_size
        input_shape = (
            self.options.max_sequence_length,
            self.options.image_height,
            self.options.image_width,
            self.options.num_channels)
        inputs = Input(shape=input_shape)

        # create cnn net foreach character image - it will perform as embedding layer for lstm
        # (each letter will get a features values extracted by the cnn)
        image_sequences = tf.unstack(inputs, axis=1)
        cnn_outputs = [self.create_cnn_layers(image) for image in image_sequences]
        cnn_outputs = tf.stack(cnn_outputs, axis=1)


        # Reshape cnn_outputs to have the required shape for Attention layer
        attn_input_shape = (batch_size * self.options.max_sequence_length, -1)
        cnn_outputs = tf.reshape(cnn_outputs, attn_input_shape)

        # LSTM layer with return_sequences=True
        x = LSTM(self.options.lstm_units, return_sequences=True)(cnn_outputs)

        # Reshape x to have the shape (batch_size, max_sequence_length, lstm_units)
        lstm_output_shape = (batch_size, self.options.max_sequence_length, self.options.lstm_units)
        x = tf.reshape(x, lstm_output_shape)

        # Add Attention layer
        x = Attention()([x, x])

        # Dense layer
        x = Dense(64, activation='relu')(x)

        # Softmax layer for classification
        outputs = Dense(self.options.num_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)
        return model

    def create_cnn_layers(self, inputs):
        x = Conv2D(32, (3, 3), activation='relu')(inputs)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        # x = Conv2D(128, (3, 3), activation='relu')(x)
        # x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        return x
