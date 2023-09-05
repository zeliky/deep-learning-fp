from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Conv2D, MaxPooling2D, Flatten, Dense, Dropout

class OneLetterClassifierModel:

    def get_model(self, input_shape, num_classes):
        model = Sequential()
        model.add(Input(shape=input_shape))
        # Convolutional layers
        model.add(Conv2D(input_shape=input_shape,
                         filters=8, kernel_size=(3, 3), padding='SAME', activation='relu', name='conv2d-1'))
        model.add(MaxPooling2D((3, 3)))

        model.add(Conv2D(filters=128, kernel_size=(6, 6), padding='SAME', activation='relu', name='conv2d-2'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(filters=256, kernel_size=(4, 4), padding='SAME', activation='relu', name='conv2d-3'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu', name='conv2d-4'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu', name='conv2d-5'))
        model.add(MaxPooling2D((2, 2), name='last-maxpooling'))

        model.add(Flatten(name='last-flatten'))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu', name='dense_512'))
        model.add(Dropout(0.2))
        model.add(Dense(num_classes, activation='softmax'))
        return model
class OneLetterClassifierModel_v5:

    def get_model(self, input_shape, num_classes):
        model = Sequential()
        model.add(Input(shape=input_shape))
        # Convolutional layers
        model.add(Conv2D(input_shape=input_shape,
                         filters=8, kernel_size=(3, 3), padding='SAME', activation='relu', name='conv2d-1'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(filters=8, kernel_size=(3, 3), padding='SAME', activation='relu', name='conv2d-2'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(filters=128, kernel_size=(5, 5), padding='SAME', activation='relu', name='conv2d-3'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu', name='conv2d-4'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(filters=256, kernel_size=(2, 2), padding='SAME', activation='relu', name='conv2d-5'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())
        model.add(Dense(512, activation='relu', name='dense_512'))
        model.add(Dense(num_classes, activation='softmax'))
        return model

class OneLetterClassifierModel_v4:
    def get_model(self, input_shape, num_classes):
        model = Sequential()
        model.add(Input(shape=input_shape))
        # Convolutional layers
        model.add(Conv2D(input_shape=input_shape,
                         filters=2, kernel_size=(6, 6), padding='SAME', activation='relu', name='conv2d-1'))
        model.add(MaxPooling2D(pool_size=(3, 3))),
        model.add(Conv2D(filters=9, kernel_size=(5, 5), padding='SAME', activation='relu', name='conv2d-2'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(filters=200, kernel_size=(5, 5), padding='SAME', activation='relu', name='conv2d-3'))
        model.add(Conv2D(filters=200, kernel_size=(5, 5), padding='SAME', activation='relu', name='conv2d-4'))
        model.add(Conv2D(filters=200, kernel_size=(5, 5), padding='SAME', activation='relu', name='conv2d-5'))
        model.add(MaxPooling2D(pool_size=(3, 3)))

        model.add(
            Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='SAME', activation='relu', name='conv2d-6'))

        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())

        """
        model.add(Dense(1024, activation='relu', name='dense_1024'))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu', name='dense_512'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        """

        model.add(Dense(512, activation='relu', name='dense_512'))
        model.add(Dense(num_classes, activation='softmax'))
        return model


class OneLetterClassifierModel_v3:

    def get_model(self, input_shape, num_classes):
        model = Sequential()
        model.add(Input(shape=input_shape))
        # Convolutional layers
        model.add(Conv2D(input_shape=input_shape,
                         filters=2, kernel_size=(6, 6), padding='SAME', activation='relu', name='conv2d-1'))
        model.add(MaxPooling2D((3, 3)))
        model.add(Conv2D(filters=128, kernel_size=(5, 5), padding='SAME', activation='relu', name='conv2d-2'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(filters=128, kernel_size=(5, 5), padding='SAME', activation='relu', name='conv2d-3'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu', name='conv2d-4'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(filters=256, kernel_size=(2, 2), padding='SAME', activation='relu', name='conv2d-5'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())

        """
        model.add(Dense(1024, activation='relu', name='dense_1024'))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu', name='dense_512'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        """

        model.add(Dense(512, activation='relu', name='dense_512'))
        model.add(Dense(num_classes, activation='softmax'))
        return model


class OneLetterClassifierModel_v2:

    def get_model(self, input_shape, num_classes):
        model = Sequential()
        model.add(Input(shape=input_shape))
        # Convolutional layers
        model.add(Conv2D(input_shape=input_shape,
                         filters=16, kernel_size=(6, 6), padding='SAME', activation='relu',name='conv2d-1'))
        model.add(MaxPooling2D((3,3)))

        model.add(Conv2D(filters=128, kernel_size=(10, 10), strides=(3, 3), padding='SAME', activation='relu', name='conv2d-2'))

        model.add(Conv2D(filters=128, kernel_size=(5, 5), strides=(2, 2), padding='SAME', activation='relu', name='conv2d-3'))

        model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(3, 3), padding='SAME', activation='relu', name='conv2d-4'))

        model.add(Flatten())



        #model.add(Dense(1024, activation='relu', name='dense_1024'))
        model.add(Dense(512, activation='relu', name='dense_512'))
        model.add(Dense(num_classes, activation='softmax'))

        return model

class OneLetterClassifierModel_V1:

    def get_model(self, input_shape, num_classes):
        model = Sequential()
        model.add(Input(shape=input_shape))
        # Convolutional layers
        model.add(Conv2D(input_shape=input_shape,
                         filters=5, kernel_size=(6, 6), padding='SAME', activation='relu'))
        model.add(MaxPooling2D((4, 4)))

        model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=(10, 10), strides=(1, 1), padding='SAME', activation='relu'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu'))
        model.add(Conv2D(filters=32, kernel_size=(10, 10), strides=(1, 1), padding='SAME', activation='relu'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu'))
        model.add(Conv2D(filters=32, kernel_size=(10, 10), strides=(1, 1), padding='SAME', activation='relu'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())

        model.add(Dense(1024, activation='relu', name='dense_1024'))
        model.add(Dense(512, activation='relu', name='dense_512'))
        model.add(Dense(num_classes, activation='softmax'))

        return model