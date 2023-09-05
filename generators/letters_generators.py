from constants.constants import *
from models.options import ModelOptions
from preprocessing.user_dataset import UserDataset
from tensorflow.keras.utils import Sequence, to_categorical
import tensorflow as tf
import random
import numpy as np
import gc, pickle
from preprocessing.dataset import DataSet, full_data_set


class BaseLetterGenerator(Sequence):
    MAX_USERS_PER_CHUNK = 200

    def __init__(self, mode, user_ids, options: ModelOptions, load_types):
        self.options = options
        self.all_user_ids = [i for i in user_ids]
        self.id_to_class = {user_id: i for i, user_id in enumerate(user_ids)}
        print(f'id_to_class:{self.id_to_class}')
        self.user_ids = []
        self.input_shape = (options.image_height, options.image_width)
        self.random_shuffle_amount = options.random_shuffle_amount
        self.users_ds = {}
        self.generators = {}

        # generator settings.
        self.mode = mode
        self.load_types = load_types
        self.train_split = 0.8
        self.shuffle = True

        self.select_users_chunk()

    def __len__(self):
        return self.options.max_embedding_samples // self.options.batch_size

    def select_users_chunk(self):
        chunk_size = min(len(self.all_user_ids), self.MAX_USERS_PER_CHUNK)
        print(f"select_users_chunk {chunk_size} out of {len(self.all_user_ids)} ")
        self.user_ids = random.sample(self.all_user_ids, chunk_size)
        print(f"selected user_ids {self.user_ids}")

    def on_epoch_end(self):
        global full_data_set
        # full_data_set = DataSet()
        self.select_users_chunk()
        self.generators = {}
        print('running GarbageCollector...')
        gc.collect()

    def reset_generators(self):
        self.generators = {}

    def set_user_ids(self, user_ids):
        print(f"updating user_ids set to {user_ids}")
        self.user_ids = user_ids

    def get_user_ds(self, user_id):
        if user_id not in self.users_ds:
            uds = UserDataset(user_id)
            uds.warmup(self.load_types, self.train_split, self.shuffle)
            self.users_ds[user_id] = uds
        return self.users_ds[user_id]
    def remove_user_ds(self,user_id):
        if user_id in self.users_ds:
            del self.users_ds[user_id]

    def get_letters_generator(self, user_id, is_anchor=False):
        key = f"anc{user_id}" if is_anchor else str(user_id)
        if key not in self.generators:
            # print(f"new generator for {user_id} anchor{is_anchor}")
            uds = self.get_user_ds(user_id)
            if self.mode == MODE_TEST:
                self.generators[key] = uds.valid_letters_generator(mode=self.mode, target_size=self.input_shape,
                                                                   allowed_types=self.load_types)
            else:
                self.generators[key] = uds.random_letters_generator(mode=self.mode, target_size=self.input_shape,
                                                                    allowed_types=self.load_types,
                                                                    random_shuffle_amount=self.random_shuffle_amount)
        return self.generators[key]

    def set_train_split(self, train_split):
        self.train_split = train_split

    def set_shuffle(self, shuffle):
        self.shuffle = shuffle

    def get_allowed_types(self, mode=None):
        if mode is None:
            mode = self.mode

        if mode == MODE_TRAIN:
            allowed_types = TRAIN_TYPES
        elif mode == MODE_TEST or mode == MODE_VALIDATION:
            allowed_types = VALIDATE_TYPES
        else:
            allowed_types = TRAIN_TYPES

        return allowed_types

class LettersGenerator(BaseLetterGenerator):
    def __init__(self, mode, user_ids, options: ModelOptions, total_users):
        allowed_types = self.get_allowed_types(mode)

        super().__init__(mode, user_ids, options, allowed_types)
        self.total_users = total_users

    def __len__(self):
        if self.mode == MODE_TEST or self.mode==MODE_VALIDATION:
            total_batches =  self._exact_len() // self.options.batch_size
        else:
          #lines = 20
          lines = 15
          users = len(self.user_ids)
          #letters_per_line = self.options.max_sequence_length
          letters_per_line =30
          types = len(self.load_types)
          random_shuffle_amount = 5
          total_batches = (types * lines * users * letters_per_line* random_shuffle_amount) // self.options.batch_size
          # print(f"LettersGenerator __len__ {total_batches}")


        return total_batches+1 if total_batches  else 1

    def __getitem__(self, index):
        batch, labels = [], []
        possible_items = self.options.batch_size if self.mode != MODE_TEST else min(self.options.batch_size,
                                                                                    self._exact_len())
        uids = [i for i in self.user_ids]
        for s in range(possible_items):
            if len(uids) == 0:
                break
            user_id = random.choice(uids)
            # print(f"__getitem__{user_id}")
            try:
                letter, _, _, _ = next(self.get_letters_generator(user_id))
                batch.append(letter)
                labels.append(to_categorical(self.id_to_class[user_id], num_classes=self.total_users))
            except StopIteration:
                uids.remove(user_id)

        if len(batch) == 0:
            batch = np.zeros((self.options.batch_size, self.options.image_height, self.options.image_width, 1))
            labels = np.zeros((self.options.batch_size,))
        # print(f"LettersGenerator:{self.mode}__getitem__ batch: {len(batch)},{len(labels)}")
        return np.asarray(batch), np.asarray(labels)

    def _exact_len(self):
        total = 0
        for user_id in self.user_ids:
            uds = self.get_user_ds(user_id)
            total += uds.count_possible_options(self.mode)
        print(total)
        return total


class TripletsGenerator(BaseLetterGenerator):
    def __init__(self, mode, user_ids, options: ModelOptions):
        super().__init__(mode, user_ids, options, EMBEDDING_TYPES)
        self.set_train_split(1)  # it should iterate all rows
        self.set_shuffle(False)

    def __len__(self):
        lines = 20
        users = len(self.user_ids)
        letters_per_line = self.options.max_sequence_length
        random_shuffle_amount = 5

        total_batches = (lines * users * letters_per_line * random_shuffle_amount) // self.options.batch_size
        # print(f"LettersGenerator __len__ {total_batches}")
        return total_batches

    def __getitem__(self, index):
        anchors, positives, negatives = [], [], []
        for _ in range(self.options.batch_size):
            # print(f"__getitem__ {2} out of {len(self.user_ids)} ")
            positive_user, negative_user_id = random.sample(self.user_ids, 2)
            # print(f"TripletsGenerator __getitem__ {positive_user} {negative_user_id} " )
            for triplet in self.get_triplets(positive_user, negative_user_id):
                if triplet is None:
                    # print(f"__getitem__ {2} out of {len(self.user_ids)} ")
                    positive_user, negative_user_id = random.sample(self.user_ids, 2)
                    continue
                anchor, positive, negative = triplet
                anchors.append(anchor)
                positives.append(positive)
                negatives.append(negative)

        batch = [np.asarray(anchors), np.asarray(positives), np.asarray(negatives)]
        return batch, []

    def get_triplets(self, positive_user, negative_user_id):
        anc_letter, img_path, line_idx, split_index = next(self.get_letters_generator(positive_user, True))
        uds = self.get_user_ds(positive_user)
        filtered = list(filter(lambda im_type: im_type != img_path, self.load_types))
        if len(filtered) > 0:
            img_path = random.choice(filtered)
        positive_letter = uds.get_letter(img_path, line_idx, split_index, self.input_shape)
        negative_letter, _, _, _ = next(self.get_letters_generator(negative_user_id, False))
        if anc_letter is not None and positive_letter is not None and negative_letter is not None:
            yield anc_letter, positive_letter, negative_letter


class SequenceGenerator(BaseLetterGenerator):
    def __init__(self, mode, user_ids, options: ModelOptions):
        allowed_types = self.get_allowed_types(mode)
        super().__init__(mode, user_ids, options, allowed_types)
        self.set_train_split(0.8)
        self.set_shuffle(True)

    def on_epoch_end(self):
        self.reset_generators()

    def get_sequence_generator(self, user_id):
        if user_id not in self.generators:
            # print(f"new generator for {user_id}")
            uds = self.get_user_ds(user_id)
            if self.mode != MODE_TEST:
                self.generators[user_id] = uds.random_line_generator(mode=self.mode,
                                                                     max_sequence_length=self.options.max_sequence_length,
                                                                     target_size=self.input_shape,
                                                                     allowed_types=self.load_types)
            else:
                # replace to valid_line_generator
                self.generators[user_id] = uds.valid_line_generator(mode=self.mode,
                                                                    max_sequence_length=self.options.max_sequence_length,
                                                                    target_size=self.input_shape,
                                                                    allowed_types=self.load_types)
        return self.generators[user_id]

    def __len__(self):
        if self.mode == MODE_TEST:
            total_batches = self._exact_len() // self.options.batch_size
        else:
            lines = 20
            users = len(self.user_ids)
            random_shuffle_amount = self.options.random_shuffle_amount
            types = len(ALLOWED_TYPES)
            total_batches = (types * lines * users * random_shuffle_amount) // self.options.batch_size
        return total_batches + 1 if total_batches else 1

    def __getitem__(self, index):
        batch, labels = [], []
        possible_items = self.options.batch_size if self.mode != MODE_TEST else min(self.options.batch_size,
                                                                                    self._exact_len())
        uids = [i for i in self.user_ids]
        for s in range(possible_items):
            if len(uids) == 0:
                break
            user_id = random.choice(uids)
            try:
                sequence = next(self.get_sequence_generator(user_id))
                batch.append(sequence)
                labels.append(to_categorical(self.id_to_class[user_id], num_classes=self.options.num_classes))
            except StopIteration:
                uids.remove(user_id)
                self.remove_user_ds(user_id)
        return np.asarray(batch), np.asarray(labels)

    def _exact_len(self):
        total = 0
        for user_id in self.user_ids:
            uds = self.get_user_ds(user_id)
            total += uds.count_possible_lines(self.mode)
        return total
