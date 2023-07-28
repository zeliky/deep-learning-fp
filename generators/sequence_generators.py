import tensorflow as tf
from tensorflow.keras.utils import Sequence, to_categorical
from preprocessing.utils import *
from models.options import ModelOptions
import numpy as np
from constants.constants import *
from generators.collection import DataGeneratorsCollection
from generators.exceptions import FinalStopIteration


class BaseSequenceGenerator(Sequence):
    def __init__(self, user_ids, model_options: ModelOptions, generators: DataGeneratorsCollection):
        self.active_user_ids = [i for i in user_ids]
        self.user_ids = [i for i in user_ids]
        self.generators = generators
        self.num_classes = max(user_ids) + 1
        self.options = model_options

    def next_sequence(self):
        raise 'Not implemented'

    def total_lines(self):
        raise 'Not implemented'

    def get_random_user(self):
        # print(f"users len{len(self.user_ids)}")
        if len(self.active_user_ids):
            return np.random.choice(self.active_user_ids)
        raise FinalStopIteration()

    def __getitem__(self, index):
        labels = []
        sequences = []
        for _ in range(self.options.batch_size):
            try:
                user_id, sequence = self.next_sequence()
                if sequence is None:
                    # print(f"user  {user_id} has finished his dataset ")
                    self.active_user_ids.remove(user_id)
                    continue
                labels.append(user_id)
                sequences.append(sequence)
            except FinalStopIteration:
                print("NO MORE DATA!")
                break

        if len(sequences) == 0:
            return tf.convert_to_tensor([]), tf.convert_to_tensor([])
        padded_sequences = pad_sequences(self.options.max_sequence_length, sequences, self.options.image_height,
                                         self.options.image_width, self.options.num_channels)

        batch_sequences = np.array(padded_sequences)
        batch_labels = np.array(labels)
        print(f"Batch Shape: {batch_sequences.shape}, Label Shape: {batch_labels.shape}")

        return tf.convert_to_tensor(batch_sequences), tf.convert_to_tensor(batch_labels)

    def __len__(self):
        sfl_count = self.generators.random_shuffle_amount if self.generators.random_shuffle_amount > 0 else 1
        samples = self.total_lines() * sfl_count * len(ALLOWED_TYPES)
        epoch_len = samples // self.options.batch_size
        print(f'epoch_len={epoch_len}')
        return max(1, epoch_len)

    def on_epoch_end(self):
        self.generators.reset_generators()
        self.active_user_ids = [i for i in self.user_ids]


class TrainSequenceGenerator(BaseSequenceGenerator):
    def next_sequence(self):
        user_id = self.get_random_user()
        sequence = None
        try:
            sequence = next(self.generators.get_train_generator(user_id))
            return user_id, sequence
        except StopIteration:
            self.generators.deactivate_generator('train', user_id)
        return user_id, None

    def total_lines(self):
        total_lines = 0
        for user_id in self.active_user_ids:
            total_lines += self.generators.train_lines_amount(user_id)
        return total_lines


class ValidationSequenceGenerator(BaseSequenceGenerator):
    def next_sequence(self):
        sequence = None
        user_id = self.get_random_user()
        try:
            sequence = next(self.generators.get_validation_generator(user_id))
            return user_id, sequence
        except StopIteration:
            self.generators.deactivate_generator('valid', user_id)
        return user_id, None

    def total_lines(self):
        total_lines = 0
        for user_id in self.active_user_ids:
            total_lines += self.generators.validation_lines_amount(user_id)
        return total_lines


class TestSequenceGenerator(BaseSequenceGenerator):
    def next_sequence(self):
        sequence = None
        user_id = self.get_random_user()
        try:
            sequence = next(self.generators.get_test_generator(user_id))
            return user_id, sequence
        except StopIteration:
            self.generators.deactivate_generator('test', user_id)
        return user_id, None

    def total_lines(self):
        total_lines = 0
        return len(self.active_user_ids)
