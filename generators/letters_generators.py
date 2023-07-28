from models.options import ModelOptions
from preprocessing.user_dataset import UserDataset
from tensorflow.keras.utils import Sequence,to_categorical
import tensorflow as tf
import random
import numpy as np


class BaseLetterGenerator(Sequence):
    def __init__(self, mode, user_ids, options: ModelOptions):
        self.options = options
        self.user_ids = [i for i in user_ids]
        self.id_to_class = {user_id: i for i, user_id in enumerate(user_ids)}
        self.input_shape = (options.image_height, options.image_width)
        self.random_shuffle_amount = options.random_shuffle_amount
        self.users_ds = {}
        self.generators = {}
        self.mode = mode

    def __len__(self):
        return self.options.max_embedding_samples

    def on_epoch_end(self):
        self.generators = {}

    def reset_generators(self):
        self.generators = {}

    def get_user_ds(self, user_id):
        if user_id not in self.users_ds:
            uds = UserDataset(user_id)
            uds.warmup()
            self.users_ds[user_id] = uds
        return self.users_ds[user_id]

    def get_letters_generator(self, user_id, is_anchor=False):
        key = f"anc{user_id}" if is_anchor else str(user_id)
        if key not in self.generators:
            print(f"new generator for {user_id} anchor{is_anchor}")
            uds = self.get_user_ds(user_id)
            self.generators[key] = uds.random_letters_generator(mode=self.mode, target_size=self.input_shape,
                                                                  original_only=is_anchor, random_shuffle_amount=self.random_shuffle_amount)
        return self.generators[key]


class LettersGenerator(BaseLetterGenerator):
    def __init__(self, mode, user_ids, options: ModelOptions):
        super().__init__(mode, user_ids, options)

    def __len__(self):
        lines = 20
        users = len(self.user_ids)
        letters = 50
        random_shuffle_amount = self.options.random_shuffle_amount
        types = len(ALLOWED_TYPES)

        total_batches = (types * lines * users * letters* random_shuffle_amount) // self.options.batch_size
        #print(f"LettersGenerator __len__ {total_batches}")
        return total_batches

    def __getitem__(self, index):
        batch, labels = [], []
        users_count = len(self.user_ids)
        for s in range(self.options.batch_size):
            user_id = random.choice(self.user_ids)
            letter = next(self.get_letters_generator(user_id))
            batch.append(letter)
            labels.append(to_categorical(self.id_to_class[user_id], num_classes=users_count))
            # labels.append(self.id_to_class[user_id]+1)

        if len(batch) == 0:
            batch = np.zeros((self.options.batch_size, self.options.image_height, self.options.image_width, 1))
            labels = np.zeros((self.options.batch_size,))
        #print(f"LettersGenerator batch: {len(batch)}")
        return np.asarray(batch), np.asarray(labels)


class TripletsGenerator(BaseLetterGenerator):
    def __init__(self, mode, user_ids, options: ModelOptions):
        super().__init__(mode, user_ids, options)

    def __getitem__(self, index):
        batch, labels = [], []
        for _ in range(self.options.batch_size):
            positive_user, negative_user_id = random.sample(self.user_ids, 2)
            for triplet in self.get_triplets(positive_user, negative_user_id):
                if triplet is None:
                    positive_user, negative_user_id = random.sample(self.user_ids, 2)
                    continue
                batch.append(triplet)
                labels.append(positive_user)

        return np.asarray(batch), np.asarray(labels)

    def get_triplets(self, positive_user, negative_user_id):
        anchor_generator = self.get_letters_generator(positive_user, True)
        positive_generator = self.get_letters_generator(positive_user, False)
        negative_generator = self.get_letters_generator(negative_user_id, False)
        anc_letter = next(anchor_generator)
        positive_letter = next(positive_generator)
        negative_letter = next(negative_generator)
        if anc_letter is not None and positive_letter is not None and negative_letter is not None:
            yield [anc_letter, positive_letter, negative_letter]