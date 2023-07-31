from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input


class CustomCNN:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def get_model(self, options):
        model = Sequential()
        model.add(Input(shape=self.input_shape))

        # Adding Convolutional Layers
        for i in range(options['depth']):
            model.add(Conv2D(filters=options['filters'][i],
                             kernel_size=options['kernel_sizes'][i],
                             strides=options['strides'][i],
                             padding=options['padding'][i]))
            model.add(Activation(options['conv_activation']))
            if options['pooling'][i]:
                model.add(MaxPooling2D(pool_size=options['pool_sizes'][i],
                                       strides=options['pool_strides'][i]))

        model.add(Flatten())  # Flattening the 2D arrays for fully connected layers

        # Adding Fully Connected Layers
        for i in range(options['fc_layers']):
            model.add(Dense(options['fc_units'][i]))
            model.add(Activation(options['fc_activation']))
            model.add(Dropout(options['dropout_rate']))

        # Output Layer
        if not options['skip_classification']:
            model.add(Dense(options['num_classes'], activation='softmax'))
        return model
