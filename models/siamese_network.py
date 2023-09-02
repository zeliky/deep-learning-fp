from models.options import ModelOptions
from tensorflow.keras.models import Model
from models.options import ModelOptions
from tensorflow.keras import Input
from tensorflow.keras.layers import Layer,Dense,Lambda
from keras import backend as K
from models.one_letter_classifier_model import OneLetterClassifierModel
from keras.regularizers import l2


# based on  https://pyimagesearch.com/2023/03/06/triplet-loss-with-keras-and-tensorflow/

class SiameseModel:
    def __init__(self, options: ModelOptions, classifier_stored_weights_path):
        self.options = options
        self.alpha = options.alpha
        self.embedding = self.prepare_embedding_model(classifier_stored_weights_path)

    def get_model(self):
        # triplet_input = Input(shape=(self.options.image_height, self.options.image_width, 1), name='triplet_input')

        input_shape = (self.options.image_height, self.options.image_width, 1)
        anchor_input = Input(input_shape, name="anchor_input")
        positive_input = Input(input_shape, name="positive_input")
        negative_input = Input(input_shape, name="negative_input")

        enc_anchor = self.embedding(anchor_input)
        enc_positive = self.embedding(positive_input)
        enc_negative = self.embedding(negative_input)

        loss_layer = TripletLossLayer(alpha=self.alpha, name='triplet_loss_layer')(
            [enc_anchor, enc_positive, enc_negative])

        model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=loss_layer)
        return model

    def get_embedding(self):
        return self.embedding

    def reload_embedding_weights(self, embedding_stored_weights_path):
        self.embedding.load_weights(embedding_stored_weights_path, by_name=True)

    def prepare_embedding_model(self, classifier_stored_weights_path):
        input_shape = (self.options.image_height, self.options.image_width, 1)
        base_model = OneLetterClassifierModel().get_model(num_classes=self.options.num_classes, input_shape=input_shape)

        last_classifier_layer = base_model.get_layer(name="dense_512").output

        x = Dense(self.options.embedding_dim, activation=None,
                  kernel_regularizer=l2(1e-3),
                  kernel_initializer='he_uniform')(last_classifier_layer)

        x = Lambda(lambda v: K.l2_normalize(v, axis=-1))(x)

        new_model = Model(inputs=base_model.input, outputs=x)
        new_model.load_weights(classifier_stored_weights_path, by_name=True)
        for layer in new_model.layers[:-4]:  # Exclude the newly added Dense and Lambda layers
            layer.trainable = False
        new_model.summary()
        return new_model


class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        p_dist = K.sum(K.square(anchor - positive), axis=-1)
        n_dist = K.sum(K.square(anchor - negative), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)

    def cosine_similarity_loss(self, inputs):
        anchor, positive, negative = inputs
        p_sim = K.sum(anchor * positive, axis=-1) / (
                    K.sqrt(K.sum(K.square(anchor), axis=-1)) * K.sqrt(K.sum(K.square(positive), axis=-1)))
        n_sim = K.sum(anchor * negative, axis=-1) / (
                    K.sqrt(K.sum(K.square(anchor), axis=-1)) * K.sqrt(K.sum(K.square(negative), axis=-1)))
        return K.sum(K.maximum(self.alpha - p_sim + n_sim, 0), axis=0)

    def call(self, inputs):
        # loss = self.triplet_loss(inputs)
        loss = self.cosine_similarity_loss(inputs)
        self.add_loss(loss)
        return loss
