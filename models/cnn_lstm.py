import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Attention, TimeDistributed
from models.options import ModelOptions


class CnnLstmAttentionModel:
    def __init__(self, options: ModelOptions):
        self.options = options

    def get_model(self):
        input_shape = (self.options.max_sequence_length, self.options.image_height, self.options.image_width,
                       self.options.num_channels)
        inputs = Input(shape=input_shape)

        # create CNN layers for each image in the sequence
        image_sequences = tf.unstack(inputs, axis=1)
        # print(f'{image_sequences.shape')
        cnn_outputs = [self.create_cnn_layers(image) for image in image_sequences]
        # print(f'{len(cnn_outputs)}x{cnn_outputs[0]}')

        # going all output into input seq * len(feature vecotr) output of cnn
        # perform as embedding vector (each image is a letter)
        cnn_outputs = tf.stack(cnn_outputs, axis=1)

        # print(f'{cnn_outputs.shape}')
        x = LSTM(self.options.lstm_units, return_sequences=True)(cnn_outputs)

        # attension layer that will help to extreact important letters to classify
        x = Attention()([x, x])

        # Softmax layer for classification
        outputs = Dense(self.options.num_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)
        return model

    def create_cnn_layers(self, inputs):
        x = Conv2D(32, (3, 3), activation='relu')(inputs)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        return x
