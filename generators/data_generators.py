from tensorflow.keras.utils import Sequence, to_categorical
import numpy as np
from constants.constants import *
from generators.collection import DataGeneratorsCollection
from generators.exceptions import FinalStopIteration


class BaseDataGenerator(Sequence):
    def __init__(self, user_ids, batch_size, generators: DataGeneratorsCollection):
        self.active_user_ids = [i for i in user_ids]
        self.user_ids = [i for i in user_ids]
        self.batch_size = batch_size
        self.generators = generators
        self.num_classes = max(user_ids) + 1

    def next_sequence(self):
        raise 'Not implemented'

    def get_random_user(self):
        # print(f"users len{len(self.user_ids)}")
        if len(self.active_user_ids):
            return np.random.choice(self.active_user_ids)
        raise FinalStopIteration()

    def __getitem__(self, index):
        X_batch, Y_batch = [], []
        samples = 0
        while samples < self.batch_size:
            try:
                user_id, sequence = self.next_sequence()
                if sequence is None:
                    # print(f"user  {user_id} has finished his dataset ")
                    self.active_user_ids.remove(user_id)
                    continue

                # print(f"sequence shape {len(sequence)}")
                X_batch.append(sequence)
                Y_batch.append(user_id)
                samples += 1
            except FinalStopIteration:
                print("NO MORE DATA!")
                break

        return X_batch, to_categorical(Y_batch, num_classes=self.num_classes)

    def __len__(self):
        print(len(self.active_user_ids) * (1 + self.generators.random_shuffle_amount) * len(ALLOWED_TYPES))
        return len(self.active_user_ids) * (1 + self.generators.random_shuffle_amount) * len(ALLOWED_TYPES)

    def on_epoch_end(self):
        self.generators.reset_generators()
        self.active_user_ids = [i for i in self.user_ids]


class TrainDataGenerator(BaseDataGenerator):
    def next_sequence(self):
        user_id = self.get_random_user()
        sequence = None
        try:
            sequence = next(self.generators.get_train_generator(user_id))
        except StopIteration:
            self.generators.deactivate_generator('train', user_id)

        return user_id, sequence


class ValidationDataGenerator(BaseDataGenerator):
    def next_sequence(self):
        sequence = None
        user_id = self.get_random_user()
        try:
            sequence = next(self.generators.get_validation_generator(user_id))
        except StopIteration:
            self.generators.deactivate_generator('valid', user_id)
        return user_id, sequence


class TestDataGenerator(BaseDataGenerator):
    def next_sequence(self):
        sequence = None
        user_id = self.get_random_user()
        try:
            sequence = next(self.generators.get_test_generator(user_id))
        except StopIteration:
            self.generators.deactivate_generator('test', user_id)
        return user_id, sequence
