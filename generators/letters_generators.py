from constants.constants import *
from models.options import ModelOptions
from preprocessing.user_dataset import UserDataset
from tensorflow.keras.utils import Sequence, to_categorical
import tensorflow as tf
import random
import numpy as np
import gc
from preprocessing.dataset import DataSet, full_data_set

class BaseLetterGenerator(Sequence):
    MAX_USERS_PER_CHUNK = 10

    def __init__(self, mode, user_ids, options: ModelOptions, load_types):
        self.options = options
        self.all_user_ids = [i for i in user_ids]
        self.id_to_class = {user_id: i for i, user_id in enumerate(user_ids)}
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
        full_data_set = DataSet()
        self.select_users_chunk()
        self.generators = {}
        print('running GarbageCollector...')
        gc.collect()

    def get_user_ds(self, user_id):
        if user_id not in self.users_ds:
            uds = UserDataset(user_id)
            uds.warmup(self.load_types, self.train_split, self.shuffle)
            self.users_ds[user_id] = uds
        return self.users_ds[user_id]

    def get_letters_generator(self, user_id, is_anchor=False):
        key = f"anc{user_id}" if is_anchor else str(user_id)
        if key not in self.generators:
            # print(f"new generator for {user_id} anchor{is_anchor}")
            uds = self.get_user_ds(user_id)
            self.generators[key] = uds.random_letters_generator(mode=self.mode, target_size=self.input_shape,
                                                                allowed_types=self.load_types,
                                                                random_shuffle_amount=self.random_shuffle_amount)
        return self.generators[key]

    def set_train_split(self, train_split):
        self.train_split = train_split

    def set_shuffle(self, shuffle):
        self.shuffle = shuffle


class LettersGenerator(BaseLetterGenerator):
    def __init__(self, mode, user_ids, options: ModelOptions, total_users):
        super().__init__(mode, user_ids, options, EMBEDDING_TYPES)
        self.total_users = total_users

    def __len__(self):
        lines = 20
        # lines=10
        users = len(self.user_ids)
        letters_per_line = self.options.max_sequence_length
        # letters =30
        random_shuffle_amount = self.options.random_shuffle_amount
        types = len(self.load_types)

        total_batches = (types * lines * users * letters_per_line * random_shuffle_amount) // self.options.batch_size
        # print(f"LettersGenerator __len__ {total_batches}")
        return total_batches

    def __getitem__(self, index):
        batch, labels = [], []
        users_count = len(self.user_ids)
        for s in range(self.options.batch_size):
            user_id = random.choice(self.user_ids)
            # print(f"__getitem__{user_id}")
            letter, _, _, _ = next(self.get_letters_generator(user_id))
            batch.append(letter)
            labels.append(to_categorical(self.id_to_class[user_id], num_classes=self.total_users))
            # print(labels)

        if len(batch) == 0:
            batch = np.zeros((self.options.batch_size, self.options.image_height, self.options.image_width, 1))
            labels = np.zeros((self.options.batch_size,))
        # print(f"LettersGenerator batch: {len(batch)}")
        return np.asarray(batch), np.asarray(labels)


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
            #print(f"__getitem__ {2} out of {len(self.user_ids)} ")
            positive_user, negative_user_id = random.sample(self.user_ids, 2)
            # print(f"TripletsGenerator __getitem__ {positive_user} {negative_user_id} " )
            for triplet in self.get_triplets(positive_user, negative_user_id):
                if triplet is None:
                    #print(f"__getitem__ {2} out of {len(self.user_ids)} ")
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
        if len(filtered)>0:
            img_path = random.choice(filtered)
        positive_letter = uds.get_letter(img_path, line_idx, split_index, self.input_shape)
        negative_letter, _, _, _ = next(self.get_letters_generator(negative_user_id, False))
        if anc_letter is not None and positive_letter is not None and negative_letter is not None:
            yield anc_letter, positive_letter, negative_letter