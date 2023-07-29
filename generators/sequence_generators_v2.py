import tensorflow as tf
from tensorflow.keras.utils import Sequence, to_categorical
from preprocessing.utils import *
from models.options import ModelOptions
import numpy as np
from constants.constants import *
from preprocessing.user_dataset import UserDataset


class SequenceGenerator(Sequence):

    def __init__(self, mode, user_ids, model_options: ModelOptions):
        self.options = model_options
        self.user_ids = [i for i in user_ids]
        self.id_to_class = {user_id: i for i, user_id in enumerate(user_ids)}
        self.input_shape = (model_options.image_height, model_options.image_width)
        self.usage_stats = {}
        self.num_classes = model_options.num_classes
        self.users_ds = {}
        self.generators = {}
        self.mode = mode

    def on_epoch_end(self):
        self.generators = {}

    def incr_usage(self, user_id):
        if user_id not in self.usage_stats:
            self.usage_stats[user_id] = 0
        self.usage_stats[user_id] += 1

    def max_usages_per_user(self):
        lines = 50
        return lines * self.options.random_shuffle_amount

    def get_user_ds(self, user_id):
        if user_id not in self.users_ds:
            uds = UserDataset(user_id)
            uds.warmup()
            self.users_ds[user_id] = uds
        self.incr_usage(user_id)
        return self.users_ds[user_id]

    def get_sequence_generator(self, user_id):
        if user_id not in self.generators:
            #print(f"new generator for {user_id}")
            uds = self.get_user_ds(user_id)
            self.generators[user_id] = uds.random_line_generator(mode=self.mode,
                                                                 max_sequence_length=self.options.max_sequence_length,
                                                                 target_size=self.input_shape)
        return self.generators[user_id]

    def __len__(self):
        lines = 20
        users = len(self.user_ids)
        random_shuffle_amount = self.options.random_shuffle_amount
        types = len(ALLOWED_TYPES)

        total_batches = (types * lines * users * random_shuffle_amount) // self.options.batch_size
        return total_batches

    def __getitem__(self, index):
        batch, labels = [], []
        for s in range(self.options.batch_size):
            user_id = random.choice(self.user_ids)
            sequence = next(self.get_sequence_generator(user_id))
            batch.append(sequence)
            labels.append(to_categorical(self.id_to_class[user_id], num_classes=self.options.num_classes))

        return np.asarray(batch), np.asarray(labels)
