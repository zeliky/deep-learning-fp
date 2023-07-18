from constants.constants import *
import random
import numpy as np
from PIL import Image


def normalized_line(line_data):
    desired_shape = LINE_SHAPE
    # normalized_data =  (255 - line_data) / 255.0
    normalized_data = (255 - line_data)
    pad_rows = max(0, desired_shape[0] - normalized_data.shape[0])
    pad_cols = max(0, desired_shape[1] - normalized_data.shape[1])

    padded_array = np.pad(normalized_data, ((0, pad_rows), (0, pad_cols)), mode='constant')
    return padded_array


def select_train_validation_lines(user_image, train_split=0.8):
    rows = []
    for i, line in enumerate(user_image.get_all_lines()):
        idx = i + 1
        if not user_image.is_test_line(idx) and not is_empty_line(line):
            rows.append(idx)

    random.shuffle(rows)
    split_idx = int(len(rows) * train_split)
    return rows[0:split_idx], rows[split_idx:]


def show_line(line_data):
    image = Image.fromarray(line_data.astype(np.uint8))
    image.show()


def is_empty_line(line_data, threshold=4000):
    values = line_data.flatten()
    sum = values[values < 50].sum()
    # print("is_empty_line {}".format(sum))
    return sum < threshold


def split_and_shuffle_array(A, split_points):
    chunks = []
    start_idx = 0
    for end_idx in split_points:
        chunk = A[:, start_idx:end_idx]
        chunks.append(chunk)
        start_idx = end_idx
    last_chunk = A[:, start_idx:]
    chunks.append(last_chunk)
    np.random.shuffle(chunks)
    shuffled_array = np.concatenate(chunks, axis=1)
    return shuffled_array
