from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Attention, TimeDistributed


def get_model(input_shape, num_classes, num_images_per_set=50):
    input_layer = Input(shape=(num_images_per_set, *input_shape))

    conv1 = TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation='relu'))(input_layer)
    pooling1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)

    conv2 = TimeDistributed(Conv2D(64, kernel_size=(3, 3), activation='relu'))(pooling1)
    pooling2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)

    flatten = TimeDistributed(Flatten())(pooling2)

    lstm = LSTM(64, return_sequences=True)(flatten)
    attention = Attention()([lstm, lstm])

    dense1 = Dense(64, activation='relu')(attention)
    dense2 = Dense(32, activation='relu')(dense1)
    output = Dense(num_classes, activation='softmax')(dense2)

    return Model(inputs=input_layer, outputs=output)


