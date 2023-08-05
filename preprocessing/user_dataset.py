import random
from concurrent.futures import ThreadPoolExecutor
from preprocessing.utils import *
from preprocessing.dataset import full_data_set

import cv2


class UserDataset:
    def __init__(self, user_id):
        self.user_id = user_id
        self.train_lines = []
        self.validation_lines = []
        self.all_lines = []
        self.normalized_lines = {}
        self.test_line = None
        self.split_points = {}
        self.min_width = 20
        self.min_colored_pixels = 500 * 255

    def warmup(self, load_types, train_split=0.8, shuffle=True):
        e = ThreadPoolExecutor(max_workers=len(load_types))
        futures = [e.submit(full_data_set.load_image, t, self.user_id) for t in load_types]
        results = [f.result() for f in futures]
        self.split_dataset(train_split, shuffle)

    def split_dataset(self, train_split, shuffle):
        bw_image = full_data_set.load_image(LINES_REMOVED_BW_IMAGES, self.user_id)
        self.train_lines, self.validation_lines = select_train_validation_lines(bw_image, train_split, shuffle)
        self.test_line = bw_image.get_test_line_idx()
        self.all_lines = sorted(self.train_lines + self.validation_lines)

    def get_letters(self, img_path, line_idx, target_size):
        split_points = self._get_characters_split_points(line_idx)
        line = self._get_normalized_line(img_path, line_idx)
        for (x, y, w, h) in split_points:
            img = line[:, x:x + w]
            # print(f"get_letter shape {img.shape}")
            thumbnail = create_thumbnail(img, target_size)
            np_im = np.array(thumbnail, dtype=np.float32) / 255
            np_img = np_im.reshape(target_size[0], target_size[1], 1)
            yield np_img
        return

    def get_line_as_sequence(self, img_path, line_idx, max_sequence_length, target_size):
        sequence = []
        line = self._get_normalized_line(img_path, line_idx)
        split_points = self._get_characters_split_points(line_idx)
        for (x, y, w, h) in split_points:
            img = line[:, x:x + w]
            thumbnail = create_thumbnail(img, target_size)
            np_im = np.array(thumbnail, dtype=np.float32) / 255
            np_img = np_im.reshape(target_size[0], target_size[1], 1)
            sequence.append(np_img)
        return pad_sequence(max_sequence_length, sequence, target_size[0], target_size[1], 1)

    def random_line_generator(self, mode, max_sequence_length, target_size, sample_from_lines_amount=None,
                              sequence_length=None, original_only=False):
        while True:
            types = ALLOWED_TYPES if not original_only else [ORIGINAL_IMAGES]
            img_path = random.choice(types)
            lines = self._get_lines_ids_set(mode)
            if sample_from_lines_amount is None:
                sample_from_lines_amount = random.randint(1, len(lines) - 1)
            #print(f"random_line_generator {sample_from_lines_amount} out of {len(lines)} ")
            selected_lines = random.sample(lines, min(sample_from_lines_amount,len(lines)))
            if sequence_length is None:
                sequence_length = random.randint(int(0.3 * max_sequence_length), max_sequence_length)
            sequence = []
            for _ in range(sequence_length):
                line_idx = random.choice(selected_lines)
                line = self._get_normalized_line(img_path, line_idx)
                split_points = self._get_characters_split_points(line_idx)
                (x, y, w, h) = random.choice(split_points)
                img = line[:, x:x + w]
                thumbnail = create_thumbnail(img, target_size)
                np_im = np.array(thumbnail, dtype=np.float32) / 255
                np_img = np_im.reshape(target_size[0], target_size[1], 1)
                sequence.append(np_img)
            yield pad_sequence(max_sequence_length, sequence, target_size[0], target_size[1], 1)

    def random_letters_generator(self, mode, target_size, random_shuffle_amount=1, allowed_types=ALLOWED_TYPES):
        while True:
            img_path = random.choice(allowed_types)
            lines = self._get_lines_ids_set(mode)
            line_idx = random.choice(lines)
            line = self._get_normalized_line(img_path,line_idx)
            split_points = self._get_characters_split_points(line_idx)
            split_index = random.randint(0, len(split_points) - 1)
            (x, y, w, h) = split_points[split_index]

            img = line[:, x:x + w]
            thumbnails = [create_thumbnail(img, target_size) for _ in range(random_shuffle_amount)]
            for i, thumbnail in enumerate(thumbnails):
                np_im = np.array(thumbnail, dtype=np.float32) / 255
                np_img = np_im.reshape(target_size[0], target_size[1], 1)
                # print(f"{img_path}: u:{self.user_id} l:{line_idx} x:{x}-{x+w} rand:{i}")
                yield np_img, img_path, line_idx, split_index

    def get_letter(self, img_path, line_idx, split_index, target_size):
        line = self._get_normalized_line(img_path, line_idx)
        split_points = self._get_characters_split_points(line_idx)
        (x, y, w, h) = split_points[split_index]
        img = line[:, x:x + w]
        thumbnail = create_thumbnail(img, target_size)
        np_im = np.array(thumbnail, dtype=np.float32) / 255
        return np_im.reshape(target_size[0], target_size[1], 1)

    def _get_lines_ids_set(self, mode):
        if mode == MODE_TRAIN:
            return self.train_lines
        elif mode == MODE_VALIDATION:
            return self.validation_lines
        return [self.test_line]

    def _get_characters_split_points(self, idx):
        if idx in self.split_points:
            return self.split_points[idx]
        line = self._get_normalized_line(LINES_REMOVED_BW_IMAGES, idx)
        binary = np.where(line > 30, 1, 0).astype('uint8')
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        dilation = cv2.dilate(binary, rect_kernel, iterations=1)
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        split_points = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > self.min_width:
                sub_img = line[:, x:x + w]
                # print(f"_get_characters_split_points line \t {idx}\t{x}\t{x+w}\t{sub_img.sum()}")
                if sub_img.sum() > self.min_colored_pixels:
                    split_points.append((x, y, w, h))

        self.split_points[idx] = sorted(split_points, key=lambda tup: tup[0])
        # print( self.split_points[idx])
        return self.split_points[idx]

    def _get_normalized_line(self, img_path, line_idx):
        if img_path not in self.normalized_lines:
            self.normalized_lines[img_path] = {}
        if line_idx not in self.normalized_lines:
            user_file = full_data_set.load_image(img_path, self.user_id)
            self.normalized_lines[img_path][line_idx] = normalized_line(user_file.get_line(line_idx))
        return self.normalized_lines[img_path][line_idx]

    ############# ______________________ NOT NEEEDED ANY MORE ! ______________________ #############
    def DEP_get_characters_split_points(self, idx):
        if idx in self.split_points:
            return self.split_points[idx]

        img = full_data_set.load_image(LINES_REMOVED_BW_IMAGES, self.user_id)
        line = normalized_line(img.get_line(idx))

        sum_vector = np.sum(line, axis=0)
        threshold = 0.05 * np.mean(sum_vector)
        split_points = []
        section, last_split = None, None

        for idx, val in enumerate(sum_vector):
            value_type = 0 if val < threshold else 1
            if section is None:
                section = value_type
                last_split = idx
            elif section != value_type and idx - last_split > 8:
                split_points.append(idx)
                last_split = idx
                section = value_type
        self.split_points[idx] = split_points
        return split_points

    def DEP_document_letters_generator(self, mode, target_size, original_only=False):
        types = ALLOWED_TYPES if original_only else [ORIGINAL_IMAGES]
        lines = []
        if mode == MODE_TRAIN:
            lines = self.train_lines
        elif mode == MODE_VALIDATION:
            lines = self.validation_lines
        for img_path in types:
            for line_idx in lines:
                for np_img in self.get_letters(img_path, line_idx, target_size):
                    print(f"{img_path}: u{self.user_id} l:{line_idx}")
                    yield np_img
        return

    def get_train_data(self, target_size):
        print(f"train lines {self.train_lines}")
        for img_path in ALLOWED_TYPES:
            print(img_path)
            for line_idx in self.train_lines:
                # print(f"get_train_data user:{self.user_id} -  line {img_path}::{line_idx}")
                sequence = self.get_line_as_sequence(img_path, line_idx, target_size)
                yield sequence
        return

    def get_validation_data(self, target_size):
        print(f"validation lines {self.validation_lines}")
        for line_idx in self.validation_lines:
            # print(f"get_validation_data line {ORIGINAL_IMAGES}::{line_idx}")
            yield self.get_line_as_sequence(ORIGINAL_IMAGES, line_idx, target_size)
        return

    def get_testing_data(self, target_size):
        print(f"testing line {self.test_line}")
        # print(f"get_testing_data user:{self.user_id} line {ORIGINAL_IMAGES}::{self.test_line}")
        yield self.get_line_as_sequence(ORIGINAL_IMAGES, self.test_line, target_size)
        return
