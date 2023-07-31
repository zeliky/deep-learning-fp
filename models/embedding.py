import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Layer
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

import numpy as np


class EmbeddingModel:
    def __init__(self,input_shape, embedding_dim):
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim

    def get_model(self):
        input_layer = Input(shape=self.input_shape)

        # Convolutional layers
        x = Conv2D(filters=32, kernel_size=(11, 11), strides=(3, 3), padding='SAME', activation='relu')(input_layer)
        x = MaxPooling2D((3, 3))(x)
        x = Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='SAME', activation='relu')(x)
        x = MaxPooling2D((3, 3))(x)
        x = Conv2D(filters=128, kernel_size=(3, 3),strides=(3,3), padding='SAME', activation='relu')(x)
        x = Conv2D(filters=256, kernel_size=(3, 3), strides=(3,3), padding='SAME', activation='relu')(x)
        x = Conv2D(filters=64, kernel_size=(1,1), strides=(1, 1), activation='relu')(x)
        x = Conv2D(64, (1, 1), activation='relu')(x)

        x = Flatten()(x)
        embeddings = Dense(self.embedding_dim, activation='sigmoid')(x)
        return Model(inputs=input_layer, outputs=embeddings, name="embedding")

