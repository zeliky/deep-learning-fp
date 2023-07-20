from constants.constants import TRAIN_TYPES, VALIDATE_TYPES
from tensorflow.keras.utils import Sequence, to_categorical
from preprocessing.user_dataset import UserDataset
import numpy as np
from preprocessing.dataset import full_data_set
from constants.constants import *


class FinalStopIteration(StopIteration):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class DataGeneratorsCollection:
    def __init__(self, input_shape, random_shuffle_amount=0):
        self.random_shuffle_amount = random_shuffle_amount
        self.input_shape = input_shape
        self.users_ds = {}
        self.active_generators = {
            'train': {},
            'valid': {},
            'test': {}
        }

    def get_user_ds(self, user_id):
        if user_id not in self.users_ds:
            uds = UserDataset(user_id)
            uds.warmup()
            self.users_ds[user_id] = uds
        return self.users_ds[user_id]

    def get_train_generator(self, user_id):
        if user_id not in self.active_generators['train']:
            uds = self.get_user_ds(user_id)
            self.active_generators['train'][user_id] = uds.get_train_data(target_size=self.input_shape)
        return self.active_generators['train'][user_id]

    def get_validation_generator(self, user_id):
        if user_id not in self.active_generators['valid']:
            uds = self.get_user_ds(user_id)
            self.active_generators['valid'][user_id] = uds.get_validation_data(target_size=self.input_shape)
        return self.active_generators['valid'][user_id]

    def get_test_generator(self, user_id):
        if user_id not in self.active_generators['test']:
            uds = self.get_user_ds(user_id)
            self.active_generators[user_id]['test'] = uds.get_testing_data(target_size=self.input_shape)
        return self.active_generators['test'][user_id]

    def deactivate_generator(self, gtype, user_id):
        if user_id in self.active_generators[gtype]:
            del self.active_generators[gtype][user_id]


class CustomDataGen(Sequence):
    def __init__(self, user_ids, batch_size, generators: DataGeneratorsCollection, input_shape):
        self.user_ids = user_ids
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.generators = generators
        self.num_classes = max(user_ids) + 1

    def next_sequence(self):
        raise 'Not implemented'

    def get_random_user(self):
        #print(f"users len{len(self.user_ids)}")
        if len(self.user_ids):
            return np.random.choice(self.user_ids)
        raise FinalStopIteration()

    def __getitem__(self, index):
        X_batch, Y_batch = [], []
        samples = 0
        while samples <= self.batch_size:
            try:
                user_id, sequence = self.next_sequence()
                if sequence is None:
                    #print(f"user  {user_id} has finished his dataset ")
                    self.user_ids.remove(user_id)
                    continue

                #print(f"sequence shape {len(sequence)}")
                X_batch.append(sequence)
                Y_batch.append(user_id)
                samples += 1
            except FinalStopIteration:
                print("NO MORE DATA!")
                break

        return X_batch, to_categorical(Y_batch, num_classes=self.num_classes)

    def __len__(self):
        print(len(self.user_ids) * (1 + self.random_shuffle_amount) * len(ALLOWED_TYPES))
        return len(self.user_ids) * (1 + self.random_shuffle_amount) * len(ALLOWED_TYPES)

    def on_epoch_end(self):
        pass


class TrainDataGenerator(CustomDataGen):
    def next_sequence(self):
        user_id = self.get_random_user()
        sequence = None
        try:
            sequence = next(self.generators.get_train_generator(user_id))
        except StopIteration:
            self.generators.deactivate_generator('train', user_id)

        return user_id, sequence


class ValidationDataGenerator(CustomDataGen):
    def next_sequence(self):
        sequence = None
        user_id = self.get_random_user()
        try:
            sequence = next(self.generators.get_validation_generator(user_id))
        except StopIteration:
            self.generators.deactivate_generator('test', user_id)
        return user_id, sequence


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
            X_batch.append(next(uds.get_testing_data(target_size=self.input_shape)))
            Y_batch.append(user_id)
            self.generators.deactivate_generator('test', user_id)
        return np.asarray(X_batch), to_categorical(Y_batch, num_classes=max(self.user_ids) + 1)
