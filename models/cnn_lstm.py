from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Input, AdditiveAttention, Flatten
import tensorflow as tf
from models.options import ModelOptions


from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Input, AdditiveAttention, Flatten, Concatenate

class CnnLstmAttentionModel:
    def __init__(self, options: ModelOptions, embedding_model):
        self.model_options = options
        self.embedding_model = embedding_model



    def get_model(self, options):
        input_layer = Input(shape=(self.model_options.max_sequence_length, self.model_options.image_height,
                                   self.model_options.image_width, 1))

        # 2) Use the CustomCNN as the embedding
        self.embedding_model.trainable = False

        # Apply CustomCNN to each image in the sequence
        sequence_embedding = TimeDistributed(self.embedding_model)(input_layer)

        # Flatten the output of the CustomCNN model
        flattened_sequence = TimeDistributed(tf.keras.layers.Flatten())(sequence_embedding)

        # LSTM Layer
        lstm_output = LSTM(options['lstm_units'])(flattened_sequence)

        # Additive Attention Layer
        attention_output = AdditiveAttention()([lstm_output, lstm_output])
        combined_output = Concatenate(axis=-1)([lstm_output, attention_output])

        # Flattening the output for the Dense layer
        attention_output_flat = Flatten()(combined_output)

        dense_output = Dense(options['dense_units'], activation='relu')(attention_output_flat)

        # 5) Classification Layer
        output = Dense(options['num_classes'], activation='softmax')(dense_output)

        # Build the model
        model = Model(inputs=input_layer, outputs=output)

        return model


