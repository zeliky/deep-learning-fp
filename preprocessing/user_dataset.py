from concurrent.futures import ThreadPoolExecutor
from preprocessing.utils import *
from preprocessing.dataset import full_data_set


class UserDataset:
    def __init__(self, user_id):
        self.user_id = user_id
        self.train_lines = []
        self.validation_lines = []
        self.test_line = []
        self.split_points = {}

    def warmup(self):
        e = ThreadPoolExecutor(max_workers=len(ALLOWED_TYPES))
        futures = [e.submit(full_data_set.load_image, t, self.user_id) for t in ALLOWED_TYPES]
        results = [f.result() for f in futures]

    def split_dataset(self, train_split=0.8):
        bw_image = full_data_set.load_image(LINES_REMOVED_BW_IMAGES, self.user_id)
        self.train_lines, self.validation_lines = select_train_validation_lines(bw_image)

    def get_letters(self, img_path, line_idx, normalized, target_size):
        split_points = self._get_characters_split_points(line_idx)
        user_file = full_data_set.load_image(img_path, self.user_id)
        line = normalized_line(user_file.get_line(line_idx))
        sub_images = split_array(line, split_points)
        for img in sub_images:
            # add bounding lines on right and left sides of the letter (should be kept to estimate the original size
            # compared to line height)
            img[:, 0] = 120
            img[:, -1] = 120
            thumbnail = create_thumbnail(sub_images, target_size)
            # thumbnail.show()
            np_img = np.asarray(thumbnail)
            yield normalized_line(np_img) if normalized else np_img

    def get_train_data(self, normalized=True, target_size=(25, 25)):
        for img_type in TRAIN_TYPES:
            if len(self.train_lines) == 0:
                self.split_dataset()

            for line_idx in self.train_lines:
                self.get_letters(img_type, line_idx, normalized, target_size)

    def get_testing_data(self, random_shuffle_amount=3, chunks=16, normalized=True):
        for img_type in ALLOWED_TYPES:
            img = full_data_set.load_image(img_type, self.user_id)
            base_line = img.get_testing_line()
            yield normalized_line(base_line) if normalized else base_line
            for i in range(random_shuffle_amount):
                sps = split_and_shuffle_array(base_line, chunks)
                yield normalized_line(sps) if normalized else sps

    def get_validation_data(self, random_shuffle_amount=10, chunks=32, normalized=True):
        for img_type in ALLOWED_TYPES:
            img = full_data_set.load_image(img_type, self.user_id)
            for idx in self.validation_lines:
                base_line = img.get_line(idx)
                # print("line {}".format(idx))
                yield normalized_line(base_line) if normalized else base_line
                for i in range(random_shuffle_amount):
                    # print("line {} -shuffle {} ".format(idx, i))
                    sps = split_and_shuffle_array(base_line, chunks)
                    yield normalized_line(sps) if normalized else sps

    def build_split_index(self):
        for idx in self.train_lines:
            self.split_points['train'][idx] = self._get_characters_split_points(idx)
        for idx in self.validation_lines:
            self.split_points['validation'][idx] = self._get_characters_split_points(idx)

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
            elif section != value_type and idx - last_split > 10:
                split_points.append(idx)
                last_split = idx
                section = value_type
        self.split_points[idx] = split_points
        return split_points


"""

    def get_train_data(self, random_shuffle_amount=10, normalized=True):
        split_points = {}
        for img_type in TRAIN_TYPES:
            img = full_data_set.load_image(img_type, self.user_id)
            for idx in self.train_lines:
                if idx not in split_points:
                    split_points[idx] = self._get_characters_split_points(idx)
                base_line = img.get_line(idx)
                # print("line {}".format(idx))
                yield normalized_line(base_line) if normalized else base_line
                for i in range(random_shuffle_amount):
                    # print("line {} -shuffle {} ".format(idx, i))
                    sps = split_and_shuffle_array(base_line, split_points[idx])
                    yield normalized_line(sps) if normalized else sps
                    
                    
  def line_generator(self, line_indexes, random_shuffle_amount=10, chunks=32,normalized = True):
    for img_type in ALLOWED_TYPES:
      img = ds.load_image(img_type, self.user_id)
      for idx in line_indexes:
        base_line = img.get_line(idx)
        #print("line {}".format(idx))
        yield normalized_line(base_line) if normalized else base_line
        for i in range(random_shuffle_amount):
          #print("line {} -shuffle {} ".format(idx, i))
          sps = split_and_shuffle_array(base_line,chunks)
          yield normalized_line(sps) if normalized else sps

def split_and_shuffle_array(A, num_chunks):
  num_rows, num_cols = A.shape
  chunk_size, remaining_cols = divmod(num_cols, num_chunks)
  chunks = np.split(A[:, :-remaining_cols], num_chunks,axis=1)
  if remaining_cols:
    chunks.append( A[:, -remaining_cols:])

  np.random.shuffle(chunks)
  shuffled_array = np.concatenate(chunks,axis=1)
  return shuffled_array
"""
