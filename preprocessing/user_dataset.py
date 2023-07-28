from concurrent.futures import ThreadPoolExecutor
from preprocessing.utils import *
from preprocessing.dataset import full_data_set


# from cv2 import cv2

class UserDataset:
    def __init__(self, user_id):
        self.user_id = user_id
        self.train_lines = []
        self.validation_lines = []
        self.all_lines = []
        self.test_line = None
        self.split_points = {}
        self.min_width = 20

    def warmup(self):
        e = ThreadPoolExecutor(max_workers=len(ALLOWED_TYPES))
        futures = [e.submit(full_data_set.load_image, t, self.user_id) for t in ALLOWED_TYPES]
        results = [f.result() for f in futures]
        self.split_dataset()

    def split_dataset(self, train_split=0.8):
        bw_image = full_data_set.load_image(LINES_REMOVED_BW_IMAGES, self.user_id)
        self.train_lines, self.validation_lines = select_train_validation_lines(bw_image)
        self.test_line = bw_image.get_test_line_idx()
        self.all_lines = sorted(self.train_lines + self.validation_lines)

    def get_train_data(self, target_size):
        print(f"train lines {self.train_lines}")
        for img_path in ALLOWED_TYPES:
            print(img_path)
            for line_idx in self.train_lines:
                # print(f"get_train_data user:{self.user_id} -  line {img_path}::{line_idx}")
                sequence = self.get_line_sequence(img_path, line_idx, target_size)
                yield sequence
        return

    def get_validation_data(self, target_size):
        print(f"validation lines {self.validation_lines}")
        for line_idx in self.validation_lines:
            # print(f"get_validation_data line {ORIGINAL_IMAGES}::{line_idx}")
            yield self.get_line_sequence(ORIGINAL_IMAGES, line_idx, target_size)
        return

    def get_testing_data(self, target_size):
        print(f"testing line {self.test_line}")
        # print(f"get_testing_data user:{self.user_id} line {ORIGINAL_IMAGES}::{self.test_line}")
        yield self.get_line_sequence(ORIGINAL_IMAGES, self.test_line, target_size)
        return

    def get_letters(self, img_path, line_idx, target_size):

        split_points = self._get_characters_split_points(line_idx)
        user_file = full_data_set.load_image(img_path, self.user_id)
        line = normalized_line(user_file.get_line(line_idx))
        # print(f"split_points {line_idx}: {split_points}")
        prev_width, prev_x = None, None
        for (x, y, w, h) in split_points:
            if prev_x is not None:
                x = prev_x
                w += prev_width
                # print(f"get_letters:width after join {w}")
                if w > self.min_width:
                    prev_width, prev_x = None, None

            if w < self.min_width:
                # print(f"get_letters:width {w}")
                prev_x = x
                prev_width = w
                continue

            img = line[:, x:x + w]
            # print(f"get_letter shape {img.shape}")
            thumbnail = create_thumbnail(img, target_size)
            np_im = np.array(thumbnail, dtype=np.float32) / 255
            np_img = np_im.reshape(target_size[0], target_size[1], 1)
            yield np_img
        return

    def get_line_sequence(self, img_path, line_idx, target_size):
        sequence = []
        for np_img in self.get_letters(img_path, line_idx, target_size):
            sequence.append(np_img)
        return sequence

    def DEP_document_letters_generator(self, mode, target_size, original_only=False):
        types = ALLOWED_TYPES if original_only else [ORIGINAL_IMAGES]
        lines =[]
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

    def random_letters_generator(self, mode, target_size, random_shuffle_amount=1, original_only=False):
      while True:
        types = ALLOWED_TYPES if not original_only else [ORIGINAL_IMAGES]

        lines =[]
        if mode == MODE_TRAIN:
            lines = self.train_lines
        elif mode == MODE_VALIDATION:
            lines = self.validation_lines
        img_path = random.choice(types)
        user_file = full_data_set.load_image(img_path, self.user_id)
        line_idx = random.choice(lines)
        line = normalized_line(user_file.get_line(line_idx))
        split_points = self._get_characters_split_points(line_idx)
        (x, y, w, h) = random.choice(split_points)

        img = line[:, x:x + w]
        thumbnails = [create_thumbnail(img, target_size) for _ in range(random_shuffle_amount)]
        for i, thumbnail in enumerate(thumbnails):
          np_im = np.array(thumbnail, dtype=np.float32) / 255
          np_img = np_im.reshape(target_size[0], target_size[1], 1)
          #print(f"{img_path}: u:{self.user_id} l:{line_idx} x:{x}-{x+w} rand:{i}")
          yield np_img


    def _get_characters_split_points(self, idx):
        if idx in self.split_points:
            return self.split_points[idx]
        img = full_data_set.load_image(LINES_REMOVED_BW_IMAGES, self.user_id)
        line = normalized_line(img.get_line(idx))
        binary = np.where(line > 30, 1, 0).astype('uint8')
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        dilation = cv2.dilate(binary, rect_kernel, iterations=1)
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        split_points = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > self.min_width:
                split_points.append((x, y, w, h))

        self.split_points[idx] = sorted(split_points, key=lambda tup: tup[0])
        # print( self.split_points[idx])
        return self.split_points[idx]

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
