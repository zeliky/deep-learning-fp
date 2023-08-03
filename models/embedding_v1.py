import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D,Lambda, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.utils import Sequence, to_categorical, plot_model
from keras.regularizers import l2
import numpy as np


class EmbeddingModel:

    def get_model(self, input_shape, embedding_dim):
        model = Sequential()
        model.add(Input(shape=input_shape))
        # Convolutional layers
        model.add(Conv2D(input_shape=input_shape,
                         filters=32, kernel_size=(11, 11), padding='SAME', activation='relu',
                         # kernel_regularizer=l2(2e-4),
                         kernel_initializer='he_uniform'

                         ))
        model.add(MaxPooling2D((3, 3)))
        model.add(Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='SAME', activation='relu'))
        model.add(MaxPooling2D((3, 3)))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(3, 3), padding='SAME', activation='relu'))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(3, 3), padding='SAME', activation='relu'))
        model.add(Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), activation='relu'))
        # model.add(Conv2D(64, (1, 1), activation='relu'))
        model.add(Flatten())
        # model.add(Dense(1024, activation='relu',
        # kernel_regularizer=l2(1e-3),
        # kernel_initializer='he_uniform'
        #           ))
        model.add(Dense(embedding_dim, activation='relu',
                        # kernel_regularizer=l2(1e-3),
                        # kernel_initializer='he_uniform'
                        ))
        # model.add(Lambda(lambda x: K.l2_normalize(x,axis=-1)))

        return model



