from models.custom_cnn import CustomCNN
from tensorflow.keras.models import Model
import tensorflow as tf
from models.embedding import EmbeddingModel
from models.options import ModelOptions
from tensorflow.keras.metrics import Mean
from tensorflow.keras import Input

# based on  https://pyimagesearch.com/2023/03/06/triplet-loss-with-keras-and-tensorflow/

class SiameseModel(Model):
    def __init__(self, options: ModelOptions):
        super().__init__()
        self.siameseNetwork = SiameseNetwork(options=options).get_model()
        self.margin = options.alpha
        self.loss_tracker = Mean(name="loss")

    def _compute_distance(self, inputs):
        (anchor, positive, negative) = inputs
        # embed the images using the siamese network
        embeddings = self.siameseNetwork((anchor, positive, negative))
        anchor_embedding = embeddings[0]
        positive_embedding = embeddings[1]
        negative_embedding = embeddings[2]
        # calculate the anchor to positive and negative distance
        ap_distance = tf.reduce_sum(
            tf.square(anchor_embedding - positive_embedding), axis=-1
        )
        an_distance = tf.reduce_sum(
            tf.square(anchor_embedding - negative_embedding), axis=-1
        )

        # return the distances
        return ap_distance, an_distance

    def _compute_loss(self, ap_distance, an_distance):
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    def call(self, inputs):
        # compute the distance between the anchor and positive,
        # negative images
        (ap_distance, an_distance) = self._compute_distance(inputs)
        return ap_distance, an_distance

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            # compute the distance between the anchor and positive,
            # negative images
            (ap_distance, an_distance) = self._compute_distance(inputs)
            # calculate the loss of the siamese network
            loss = self._compute_loss(ap_distance, an_distance)
        # compute the gradients and optimize the model
        gradients = tape.gradient(
            loss,
            self.siameseNetwork.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.siameseNetwork.trainable_variables)
        )
        # update the metrics and return the loss
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, inputs):
        # compute the distance between the anchor and positive,
        # negative images
        (ap_distance, an_distance) = self._compute_distance(inputs)
        # calculate the loss of the siamese network
        loss = self._compute_loss(ap_distance, an_distance)

        # update the metrics and return the loss
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker]


class SiameseNetwork:

    def __init__(self, options: ModelOptions):
        self.options = options
        self.input_shape = (options.image_height, options.image_width, 1)
        self.embedding_model = EmbeddingModel(self.input_shape, options.embedding_dim).get_model()

    def get_model(self):
        anchor_input = Input(self.input_shape, name='anchor_input')
        positive_input = Input(self.input_shape, name='positive_input')
        negative_input = Input(self.input_shape, name='negative_input')

        encoded_anchor = self.embedding_model(anchor_input)
        encoded_positive = self.embedding_model(positive_input)
        encoded_negative = self.embedding_model(negative_input)

        outputs = tf.concat([encoded_anchor, encoded_positive, encoded_negative], axis=1)
        siamese_net = Model(inputs=[anchor_input, positive_input, negative_input], outputs=outputs)

        return siamese_net

    def get_embedding_model(self):
        return self.embedding_model


def _reshape_inputs(data):
    x, y = data

    # Reshape the input data
    anchor, positive, negative = tf.unstack(x, axis=1)
    return (anchor, positive, negative)