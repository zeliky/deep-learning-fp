import random
from concurrent.futures import ThreadPoolExecutor
from preprocessing.utils import *
from preprocessing.dataset import full_data_set
import os, pickle
import cv2


class UserDataset:
    ftypes = {'original': ORIGINAL_IMAGES,
              'rotated':ROTATED_IMAGES,
              'lines_removed_bw':LINES_REMOVED_BW_IMAGES,
              'lines_removed':LINES_REMOVED_IMAGES
              }
    def __init__(self, user_id):
        self.user_id = user_id
        self.train_lines = []
        self.validation_lines = []
        self.all_lines = []
        self.normalized_lines = {}
        self.test_line = None
        self.split_points = {}
        self.min_width = 10
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

    def sub_images_base_path(self,ftype, line ):
      path = f"{SUB_IMAGES_PATH}{self.user_id}/{ftype}/{line}/"
      if not os.path.exists(path):
        os.makedirs(path)
        print(f"create {path}")
      return path


    def store_sub_lines(self, target_size):
        self.warmup(self.ftypes.values(), train_split=1, shuffle = False)
        index = {}
        for ftype, img_path in self.ftypes.items():
          print(ftype)
          print(img_path)
          for line_idx in self.all_lines:
            index[line_idx] = {}
            base_dir = self.sub_images_base_path(ftype, line_idx)
            line = self._get_normalized_line(img_path, line_idx)
            split_points = self._get_characters_split_points(line_idx)
            for ch_id, (x, y, w, h) in enumerate(split_points):
               index[line_idx][ch_id] = True
               img_array = line[:, x:x + w]
               img = Image.fromarray(img_array)
               store_path = f'{base_dir}{ch_id}.jpg'
               print(store_path)
               img.save(store_path)



        index_path = f"{SUB_IMAGES_PATH}{self.user_id}/index"
        with open(index_path, 'wb') as handle:
          pickle.dump(index, handle, protocol=pickle.HIGHEST_PROTOCOL)

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

    def valid_line_generator(self, mode, max_sequence_length, target_size, allowed_types=ALLOWED_TYPES):
        img_path = random.choice(allowed_types)
        lines = self._get_lines_ids_set(mode)
        while True:
            for line_idx in lines:
                line = self._get_normalized_line(img_path, line_idx)
                split_points = self._get_characters_split_points(line_idx)
                sequence = []
                seq_len = 0

                for (x, y, w, h) in split_points:
                    if len(split_points) ==0:
                      #print(f"no split points for user {self.user_id} line:{line_idx}")
                      continue
                    img = line[:, x:x + w]
                    thumbnail = create_thumbnail(img, target_size, data_augmentation=False)
                    np_im = np.array(thumbnail, dtype=np.float32) / 255
                    np_img = np_im.reshape(target_size[0], target_size[1], 1)
                    sequence.append(np_img)
                    seq_len += 1
                    if seq_len == max_sequence_length:
                        break
                yield pad_sequence(max_sequence_length, sequence, target_size[0], target_size[1], 1)

    def random_line_generator(self, mode, max_sequence_length, target_size, allowed_types=ALLOWED_TYPES):
        while True:
            #print(f"random_line_generator types: {allowed_types}")
            img_path = random.choice(allowed_types)
            lines = self._get_lines_ids_set(mode)

            sequence_length = random.randint(int(0.7 * max_sequence_length), max_sequence_length)
            sequence = []
            for _ in range(sequence_length):
                line_idx = random.choice(lines)
                line = self._get_normalized_line(img_path, line_idx)
                split_points = self._get_characters_split_points(line_idx)
                if len(split_points) ==0:
                  #print(f"no split points for user {self.user_id} line:{line_idx}")
                  continue
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
            line = self._get_normalized_line(img_path, line_idx)
            split_points = self._get_characters_split_points(line_idx)
            points_amount = len(split_points)
            if points_amount <= 2:  # skip lines with too few examples
                continue
            split_index = random.randint(0, points_amount - 1)
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
                    split_points.append((x-5, y, w+5, h))

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
