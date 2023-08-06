from models.options import ModelOptions
from tensorflow.keras.models import Model
from models.options import ModelOptions
from tensorflow.keras import Input
from tensorflow.keras.layers import Layer
from keras import backend as K

# based on  https://pyimagesearch.com/2023/03/06/triplet-loss-with-keras-and-tensorflow/

class SiameseModel:
    def __init__(self, options: ModelOptions, embedding):
        super().__init__()
        self.options = options
        self.alpha = options.alpha
        self.embedding = embedding

    def get_model(self):
      #triplet_input = Input(shape=(self.options.image_height, self.options.image_width, 1), name='triplet_input')

      input_shape = (self.options.image_height, self.options.image_width ,1)
      anchor_input = Input(input_shape, name="anchor_input")
      positive_input = Input(input_shape, name="positive_input")
      negative_input = Input(input_shape, name="negative_input")

      #anchor_input = Lambda(lambda x: x[0])(triplet_input)
      #positive_input = Lambda(lambda x: x[1])(triplet_input)
      #negative_input = Lambda(lambda x: x[2])(triplet_input)

      enc_anchor = self.embedding(anchor_input)
      enc_positive = self.embedding(positive_input)
      enc_negative = self.embedding(negative_input)

      loss_layer = TripletLossLayer(alpha=self.alpha, name='triplet_loss_layer')([enc_anchor, enc_positive, enc_negative])

      model = Model(inputs=[anchor_input,positive_input,negative_input],outputs= loss_layer)
      return model
    def get_embedding(self):
      return self.embedding

class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        p_dist = K.sum(K.square(anchor-positive), axis=-1)
        n_dist = K.sum(K.square(anchor-negative), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss

