from constants.constants import *
from models.options import ModelOptions
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Flatten, Dense, Dropout, Activation, \
    BatchNormalization, Add, GlobalAveragePooling2D
from keras.regularizers import l2
from keras import backend as K

class EmbeddingModel:

    def get_model(self, input_shape, embedding_dim):
        model = Sequential()
        model.add(Input(shape=input_shape))
        # Convolutional layers
        model.add(Conv2D(input_shape=input_shape,
                         filters=32, kernel_size=(11, 11), padding='SAME', activation='relu',
                         kernel_regularizer=l2(2e-4),
                         kernel_initializer='he_uniform'
                         ))
        model.add(MaxPooling2D((3, 3)))
        model.add(Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='SAME', activation='relu'))
        model.add(MaxPooling2D((3, 3)))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu'))
        model.add(MaxPooling2D((3, 3)))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu'))
        model.add(MaxPooling2D((3, 3)))

        model.add(Flatten())
        model.add(Dense(embedding_dim, activation='relu',
                        kernel_regularizer=l2(1e-3),
                        kernel_initializer='he_uniform'
                        ))
        model.add(Lambda(lambda x: K.l2_normalize(x,axis=-1)))

        return model