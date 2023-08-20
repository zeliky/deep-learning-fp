from constants.constants import *
from utils import *
import pickle, random, math
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import numpy as np


class TrainPlan:
    def __init__(self):
        self.mode = None
        self.train_split = None
        self.shuffle = None
        self.user_ids = None
        self.train_index = {}
        self.validation_index = {}

    def set_options(self, mode, user_ids, target_size, train_split=0.8, shuffle=True):
        self.mode = mode
        self.target_size = target_size
        self.train_split = train_split
        self.shuffle = shuffle
        self.user_ids = user_ids
        self.train_index = {}
        self.validation_index = {}

    def prepare_index(self):
        for user_id in self.user_ids:
            path = f"{SUB_IMAGES_PATH}{user_id}/index"
            with open(path, 'rb') as u_pickle:
                index = pickle.load(u_pickle)
                lines = list(index.keys())
                if self.shuffle:
                    random.shuffle(lines)

                split_idx = int(len(lines) * self.train_split)
                train, validation = (lines[0:split_idx], lines[split_idx:])

                self.train_index[user_id] = {}
                for line_idx in train:
                    self.train_index[user_id][line_idx] = index[line_idx]

                self.validation_index[user_id] = {}
                for line_idx in validation:
                    self.validation_index[user_id][line_idx] = index[line_idx]

    def save_index(self):
        path = f"{SUB_IMAGES_PATH}/train_plan_{int(self.train_split * 100)}-{math.ceil((1 - self.train_split) * 100)}{'_shuffled' if self.shuffle else ''}"
        with open(path, 'wb') as g_pickle:
            data = {
                MODE_TRAIN: self.train_index,
                MODE_VALIDATION: self.train_index,
            }
            pickle.dump(data, g_pickle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_train_plan(self, plan_name):
        path = f"{SUB_IMAGES_PATH}/{plan_name}"
        with open(path, 'rb') as g_pickle:
            index = pickle.load(g_pickle)
            self.train_index = index[MODE_TRAIN]
            self.validation_index = index[MODE_VALIDATION]
            self.user_ids = list(self.train_index.keys())

    def get_triplet_paths(self):
        index = self.train_index if self.mode == MODE_TRAIN else self.validation_index
        types = list(TYPES_TO_PATH.keys())
        selected_types = random.sample(types, 3)
        positive_user, negative_user = random.sample(self.user_ids, 2)
        pos_line = random.choice(list(index[positive_user].keys()))
        neg_line = random.choice(list(index[negative_user].keys()))
        pos_char = random.choice(list(index[positive_user][pos_line].keys()))
        neg_char = random.choice(list(index[negative_user][neg_line].keys()))
        return [
            f"{positive_user}/{selected_types[0]}/{pos_line}/{pos_char}.jpg",
            f"{positive_user}/{selected_types[1]}/{pos_line}/{pos_char}.jpg",
            f"{negative_user}/{selected_types[2]}/{neg_line}/{neg_char}.jpg",
        ]

    def get_sequence_paths(self, max_length):
        index = self.train_index if self.mode == MODE_TRAIN else self.validation_index

        types = list(TYPES_TO_PATH.keys())
        selected_type = random.choice(types)
        selected_user = random.choice(self.user_ids)
        sequence_length = random.randint(int(0.5 * max_length), max_length)
        sequence = []
        for i in range(0, sequence_length):
            line = random.choice(list(index[selected_user].keys()))
            char = random.choice(list(index[selected_user][line].keys()))
            sequence.append(f"{selected_user}/{selected_type}/{line}/{char}.jpg")
        return selected_user, sequence

    def load_files(self, paths, target_size, data_augmentation):
        images = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            for result in executor.map(_prepare_image, paths, (target_size, data_augmentation)):
                images.append(result)
        return images


def _prepare_image(path, target_size, data_augmentation):
    image_path = f"{SUB_IMAGES_PATH}{path}"
    im = Image.open(image_path)
    if im.mode != 'L':
        im = im.convert(mode='L')
    image_array = np.asarray(im.getchannel(0))
    return create_thumbnail(image_array, target_size, data_augmentation)
