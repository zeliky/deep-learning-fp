from constants.constants import *
from models.options import ModelOptions
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Flatten, Dense, Dropout, Activation, \
    BatchNormalization, Add, GlobalAveragePooling2D
from keras.regularizers import l2
from keras import backend as K
class EmbeddingModel:
    def get_model(self, saved_path, input_shape, embedding_dim):
        cnn_networkd = load_cnn_network(saved_path, input_shape)

        model = Sequential()
        model.add(cnn_networkd)
        model.add(Conv2D(64, (1, 1), activation='relu'))
        model.add(GlobalAveragePooling2D())
        model.add(Dense(embedding_dim, activation='relu', kernel_regularizer=l2(1e-3), kernel_initializer='he_uniform'))
        model.add(Lambda(lambda x: K.l2_normalize(x, axis=1)))  # L2 normalization layer for embedding
        return model


def load_cnn_network(saved_path, input_shape):
    m = load_model(last_save_path)
    model = Model(m.inputs, m.layers[-3].output)
    model.summary()
    for layer in model.layers:
        layer.trainable = False
    return model


model_options = ModelOptions(
    batch_size=100,
    random_shuffle_amount=1,
)
input_shape = (model_options.image_height, model_options.image_width, 1)
last_save_path = MODEL_CHECKPOINT_PATH + 'incr_model_12_users.h5'
# cnn_network = load_cnn_network(last_save_path,input_shape)
# cnn_network.summary()
embedding = EmbeddingModel().get_model(last_save_path, input_shape, model_options.embedding_dim)
# embedding.summary()
