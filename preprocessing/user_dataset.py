from concurrent.futures import ThreadPoolExecutor
from preprocessing.utils import *
from preprocessing.dataset import full_data_set


class UserDataset:
    def __init__(self, user_id):
        self.user_id = user_id
        self.train_lines = []
        self.validation_lines = []
        self.test_line = None
        self.split_points = {}

    def warmup(self):
        e = ThreadPoolExecutor(max_workers=len(ALLOWED_TYPES))
        futures = [e.submit(full_data_set.load_image, t, self.user_id) for t in ALLOWED_TYPES]
        results = [f.result() for f in futures]
        self.split_dataset()

    def split_dataset(self, train_split=0.8):
        bw_image = full_data_set.load_image(LINES_REMOVED_BW_IMAGES, self.user_id)
        self.train_lines, self.validation_lines = select_train_validation_lines(bw_image)
        self.test_line = bw_image.get_test_line_idx()

    def get_train_data(self, target_size):
        print(f"train lines {self.train_lines}")
        for img_path in ALLOWED_TYPES:
            print(img_path)
            for line_idx in self.train_lines:
                # print(f"get_train_data user:{self.user_id} -  line {img_path}::{line_idx}")
                sequence = self.get_letters(img_path, line_idx, target_size)
                yield sequence
        return

    def get_validation_data(self, target_size):
        print(f"validation lines {self.validation_lines}")
        for line_idx in self.validation_lines:
            # print(f"get_validation_data line {ORIGINAL_IMAGES}::{line_idx}")
            yield self.get_letters(ORIGINAL_IMAGES, line_idx, target_size)
        return

    def get_testing_data(self, target_size):
        print(f"testing line {self.test_line}")
        # print(f"get_testing_data user:{self.user_id} line {ORIGINAL_IMAGES}::{self.test_line}")
        yield self.get_letters(ORIGINAL_IMAGES, self.test_line, target_size)
        return

    def get_letters(self, img_path, line_idx, target_size):
        split_points = self._get_characters_split_points(line_idx)
        user_file = full_data_set.load_image(img_path, self.user_id)
        line = normalized_line(user_file.get_line(line_idx))
        # print(f'line {img_path}::{line_idx}')
        # show_line(line)

        sub_images = split_array(line, split_points)
        sequence = []
        for img in sub_images:
            # add bounding lines on right and left sides of the letter (should be kept to estimate the original size
            # compared to line height)
            img[:, 0] = 100
            img[:, -1] = 100

            thumbnail = create_thumbnail(img, target_size)
            np_im = np.array(thumbnail, dtype=np.float32) / 255
            np_img = np_im.reshape(target_size[0], target_size[1], 1)
            sequence.append(np_img)
        return sequence

    def build_split_index(self):
        for idx in self.train_lines:
            self.split_points['train'][idx] = self._get_characters_split_points(idx)
        for idx in self.validation_lines:
            self.split_points['validation'][idx] = self._get_characters_split_points(idx)

        idx = self.test_line
        self.split_points['test'][idx] = self._get_characters_split_points(idx)

    def _get_characters_split_points(self, idx):
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
