from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Flatten, Dense, Dropout, Activation, \
    BatchNormalization, Add, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model, load_model


class SimplifiedResnet:
    def res_block(self, x, filters):
        x_shortcut = x

        # First component of main path
        x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # Second component of main path
        x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)

        # Add shortcut value to main path
        x = Add()([x, x_shortcut])
        x = Activation('relu')(x)

        return x

    def get_model(self, num_classes, input_shape):
        x_input = Input(input_shape)
        x = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(x_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        # Residual blocks
        x = self.res_block(x, 64)
        x = self.res_block(x, 64)

        # Classifier
        x = Flatten()(x)
        x = Dense(num_classes, activation='softmax')(x)

        # Create model
        return Model(inputs=x_input, outputs=x)
