import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Layer
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

import numpy as np


class EmbeddingModel:
    def get_model(self, input_shape):
        # Define the tensors for the three input images
        anchor_input = Input(input_shape, name="anchor_input")
        positive_input = Input(input_shape, name="positive_input")
        negative_input = Input(input_shape, name="negative_input")

        # Create the embedding model
        embedding_model = self.embedding_model(input_shape)

        # Generate the embeddings for the anchor, positive and negative images
        encoded_anchor = embedding_model(anchor_input)
        encoded_positive = embedding_model(positive_input)
        encoded_negative = embedding_model(negative_input)

        # TripletLoss Layer
        loss_layer = TripletLossLayer(alpha=0.2, name='triplet_loss_layer')(
            [encoded_anchor, encoded_positive, encoded_negative])

        # Connect the inputs with the outputs
        triplet_net = Model(inputs=[anchor_input, positive_input, negative_input], outputs=loss_layer)

        return triplet_net

    def embedding_model(self, input_shape):
        input_layer = Input(shape=input_shape)

        # Convolutional layers
        x = Conv2D(32, (3, 3), activation='relu')(input_layer)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)

        # Flatten and Dense layers for embeddings
        x = Flatten()(x)
        embeddings = Dense(256, activation='relu')(x)

        emb_model = Model(inputs=input_layer, outputs=embeddings)
        emb_model.compile(optimizer='adam', loss=triplet_loss)
        return emb_model


def triplet_loss(inputs, dist='sqeuclidean', margin='maxplus'):
    """
    Implementation of the triplet loss function.
    """
    anchor, positive, negative = inputs
    positive_distance = K.square(anchor - positive)
    negative_distance = K.square(anchor - negative)
    if dist == 'euclidean':
        positive_distance = K.sqrt(K.sum(positive_distance, axis=-1, keepdims=True))
        negative_distance = K.sqrt(K.sum(negative_distance, axis=-1, keepdims=True))
    elif dist == 'sqeuclidean':
        positive_distance = K.sum(positive_distance, axis=-1, keepdims=True)
        negative_distance = K.sum(negative_distance, axis=-1, keepdims=True)
    loss = positive_distance - negative_distance
    if margin == 'maxplus':
        loss = K.maximum(0.0, 1 + loss)
    elif margin == 'softplus':
        loss = K.log(1 + K.exp(loss))
    return K.mean(loss)


class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        p_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        n_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
        return tf.maximum(p_dist - n_dist + self.alpha, 0.)

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss
