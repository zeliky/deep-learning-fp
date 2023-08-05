from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Flatten, Dense, Dropout, Activation, \
    BatchNormalization, Add, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model, load_model


# based on https://www.kaggle.com/code/songrise/implementing-resnet-18-using-keras

class SimplifiedResnet:
    def res_block(self, X, filters, down_sample=False):
        X_shortcut = X

        # First component of main path
        X = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        # Second component of main path
        X = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(X)
        X = BatchNormalization()(X)

        # Add shortcut value to main path
        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)
        if down_sample:
            # perform down sampling using stride of 2, according to [1].
            self.res_conv = Conv2D(
                self.__channels, strides=2, kernel_size=(1, 1), kernel_initializer='he_normal', padding="same")
            self.res_bn = BatchNormalization()

        return X

    def get_model(self, input_shape, embedding_dim):
        X_input = Input(input_shape)
        X = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(X_input)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)

        # Residual blocks
        X = self.res_block(X, 64)
        X = self.res_block(X, 64)

        # Classifier
        X = GlobalAveragePooling2D()(X)
        X = Dense(embedding_dim, activation='relu', kernel_regularizer=l2(1e-3), kernel_initializer='he_uniform')(X)
        X = Lambda(lambda x: K.l2_normalize(x, axis=1))(X)  # L2 normalization layer for embedding

        # Create model
        return Model(inputs=X_input, outputs=X)
