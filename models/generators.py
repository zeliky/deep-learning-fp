from constants.constants import TRAIN_TYPES, VALIDATE_TYPES
from tensorflow.keras.utils import Sequence, to_categorical
from preprocessing.user_dataset import UserDataset
import numpy as np
from preprocessing.dataset import full_data_set
from constants.constants import *


class DataGeneratorsCollection:
    def __init__(self, user_ids, random_shuffle_amount=0):
        self.random_shuffle_amount = random_shuffle_amount
        self.users_ds = {}

    def get_user_ds(self, user_id):
        if user_id not in self.users_ds:
            uds = UserDataset(user_id)
            uds.warmup()
            self.users_ds[user_id] = uds
        return self.users_ds[user_id]


class CustomDataGen(Sequence):
    def __init__(self, user_ids, batch_size, generators: DataGeneratorsCollection, input_shape):
        self.user_ids = user_ids
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.generators = generators

    def next_sequence(self):
        raise 'Not implemented'

    def __getitem__(self, index):
        X_batch, Y_batch = [], []
        samples = 0
        while samples <= self.batch_size:
            user_id = None
            try:
                user_id, sequence = self.next_sequence()
                X_batch.append(sequence)
                Y_batch.append(user_id)
                samples += 1
            except StopIteration:
                if user_id is not None:
                    self.user_ids.remove(user_id)

        return np.asarray(X_batch), to_categorical(Y_batch, num_classes=max(self.user_ids) + 1)

    def __len__(self):
        print(len(self.user_ids) * (1 + self.random_shuffle_amount) * len(ALLOWED_TYPES))
        return len(self.user_ids) * (1 + self.random_shuffle_amount) * len(ALLOWED_TYPES)

    def on_epoch_end(self):
        pass


class TrainDataGenerator(CustomDataGen):
    def __init__(self, user_ids, batch_size, generators: DataGeneratorsCollection, input_shape):
        super().__init__(user_ids, batch_size, generators, input_shape)

    def next_sequence(self):
        user_id = np.random.choice(self.user_ids)
        uds = self.generators.get_user_ds(user_id)
        return user_id, next(uds.get_train_data(target_size=self.input_shape))


class ValidationDataGenerator(CustomDataGen):
    def __init__(self, user_ids, batch_size, generators: DataGeneratorsCollection, input_shape):
        super().__init__(user_ids, batch_size, generators, input_shape)

    def next_sequence(self):
        user_id = np.random.choice(self.user_ids)
        uds = self.generators.get_user_ds(user_id)
        return user_id,  next(uds.get_validation_data(target_size=self.input_shape))


class TestDataGenerator(Sequence):
    def __init__(self, user_ids, generators: DataGeneratorsCollection, input_shape):
        self.user_ids = user_ids
        self.generators = generators
        self.input_shape = input_shape

    def __getitem__(self, index):
        X_batch, Y_batch = [], []
        samples = 0
        for user_id in self.user_ids:
            uds = self.generators.get_user_ds(user_id)
            X_batch.append( next(uds.get_testing_data(target_size=self.input_shape)))
            Y_batch.append(user_id)
        return np.asarray(X_batch), to_categorical(Y_batch, num_classes=max(self.user_ids) + 1)
