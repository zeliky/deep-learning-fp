from concurrent.futures import ThreadPoolExecutor
from preprocessing.utils import *
from preprocessing.dataset import ds


class UserDataset:
    def __init__(self, user_id):
        self.user_id = user_id
        self.train_idx = []
        self.validation_idx = []
        self.test_idx = []
        self.split_points = {
            'train': {},
            'validation': {}
        }

    def line_generator(self, run_type, img_type):
        img = ds.load_image(img_type, self.user_id)
        if run_type == 'train':
            lines_idx = self.train_idx
        elif run_type == 'validation':
            lines_idx = self.validation_idx

        for i in range(10):
            rand_line = random.randint(0, len(lines_idx) - 1)
            line = img.get_line(rand_line)
            rand_section = random.randint(1, len(self.split_points[run_type]) - 2)

            start_idx = self.split_points[run_type][line][rand_section]
            end_idx = self.split_points[run_type][line][rand_section + 1]
            print(f"{start_idx}-{end_idx}")
            chunk = line[:, start_idx:end_idx]
            show_line(chunk)

    def warmup(self):
        e = ThreadPoolExecutor(max_workers=len(ALLOWED_TYPES))
        futures = [e.submit(ds.load_image, t, self.user_id) for t in ALLOWED_TYPES]
        results = [f.result() for f in futures]

    def split_dataset(self, train_split=0.8):
        bw_image = ds.load_image(LINES_REMOVED_BW_IMAGES, self.user_id)
        self.train_idx, self.validation_idx = select_train_validation_lines(bw_image)

    def get_testing_data(self, random_shuffle_amount=3, chunks=16, normalized=True):
        for img_type in ALLOWED_TYPES:
            img = ds.load_image(img_type, self.user_id)
            base_line = img.get_testing_line()
            yield normalized_line(base_line) if normalized else base_line
            for i in range(random_shuffle_amount):
                sps = split_and_shuffle_array(base_line, chunks)
                yield normalized_line(sps) if normalized else sps

    def get_train_data(self, random_shuffle_amount=10, normalized=True):
        split_points = {}
        for img_type in TRAIN_TYPES:
            img = ds.load_image(img_type, self.user_id)
            for idx in self.train_idx:
                if idx not in split_points:
                    split_points[idx] = self.find_split_points(idx)
                base_line = img.get_line(idx)
                # print("line {}".format(idx))
                yield normalized_line(base_line) if normalized else base_line
                for i in range(random_shuffle_amount):
                    # print("line {} -shuffle {} ".format(idx, i))
                    sps = split_and_shuffle_array(base_line, split_points[idx])
                    yield normalized_line(sps) if normalized else sps

    def get_validation_data(self, random_shuffle_amount=10, chunks=32, normalized=True):
        for img_type in ALLOWED_TYPES:
            img = ds.load_image(img_type, self.user_id)
            for idx in self.validation_idx:
                base_line = img.get_line(idx)
                # print("line {}".format(idx))
                yield normalized_line(base_line) if normalized else base_line
                for i in range(random_shuffle_amount):
                    # print("line {} -shuffle {} ".format(idx, i))
                    sps = split_and_shuffle_array(base_line, chunks)
                    yield normalized_line(sps) if normalized else sps

    def build_split_index(self):
        for idx in self.train_idx:
            self.split_points['train'][idx] = self.find_split_points(idx)
        for idx in self.validation_idx:
            self.split_points['validation'][idx] = self.find_split_points(idx)

    def find_split_points(self, idx):
        img = ds.load_image(LINES_REMOVED_BW_IMAGES, self.user_id)
        line = normalized_line(img.get_line(idx))

        sum_vector = np.sum(line, axis=0)
        threshold = 0.05 * np.mean(sum_vector)
        split_points = []
        section, las_split = None, None

        for idx, val in enumerate(sum_vector):
            value_type = 0 if val < threshold else 1
            if section is None:
                section = value_type
                last_split = idx
            elif section != value_type and idx - last_split > 10:
                split_points.append(idx)
                last_split = idx
                section = value_type
        return split_points


"""
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
