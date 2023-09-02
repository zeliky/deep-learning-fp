from tensorflow.keras.layers import Bidirectional, GRU, Attention, Dot, Multiply, Input, MaxPooling2D, Conv2D, Flatten, \
    Dense, TimeDistributed
from tensorflow.keras.models import Sequential, load_model, Model


class SequenceClassificationModel:

    def attention_layer(self, query, values):
        attention_scores = Dot(axes=[2, 2])([query, values])
        attention_scores = Activation('softmax', name='attention_scores')(attention_scores)
        attention_output = Multiply()([attention_scores, values])
        context_vector = Lambda(lambda x: K.sum(x, axis=1), name='context_vector')(attention_output)
        return context_vector

    def get_model(self, options: ModelOptions, letter_classifier, slice_layer):
        input_shape = (options.image_height, options.image_width, 1)

        # Create a new model based on OneLetterClassifierModel architecture
        sequence_input = Input(shape=(options.max_sequence_length,) + input_shape)

        # Slice the letter classifier model up to the specified layer
        letter_classifier = Model(inputs=letter_classifier.input,
                                  outputs=letter_classifier.get_layer(slice_layer).output)

        letter_classifier_timesteps = TimeDistributed(letter_classifier)(sequence_input)
        x = TimeDistributed(Flatten())(letter_classifier_timesteps)
        x = TimeDistributed(Dense(128, activation='relu'))(x)

        # Apply Bidirectional GRU
        bidir_gru = Bidirectional(GRU(128, return_sequences=True, dropout=0.2), merge_mode='sum')(x)

        # Apply Attention mechanism
        context_vector = Attention()([x, bidir_gru])  # Using bidir_gru as values

        # Flatten and apply a dense layer to produce the classification output
        flattened_context = Flatten()(context_vector)
        sequence_output = Dense(options.num_classes, activation='softmax')(flattened_context)

        # Create the final sequence classification model
        final_model = Model(inputs=sequence_input, outputs=sequence_output)

        return final_model
