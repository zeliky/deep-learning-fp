from constants.constants import *
import matplotlib.pyplot as plt
import random, math
import numpy as np
from PIL import Image


def image_dots(img_data, threshold=50):
    height, width = img_data.shape
    for i in range(height):
        line = ''
        for j in range(width):
            if img_data[i, j] < threshold:
                line += ' '
            else:
                line += '.'
        print(line)


def show_line(line_data):
    plt.clf()
    plt.figure(figsize=(20, 5))
    plt.axis('off')
    plt.imshow(line_data.astype(np.uint8), cmap='gray')
    plt.show()


def is_empty_line(line_data, threshold=5000):
    values = line_data.flatten()
    sum = values[values < 50].sum()
    # print("is_empty_line {}".format(sum))
    return sum < threshold


def normalized_line(line_data):
    desired_shape = LINE_SHAPE
    # normalized_data =  (255 - line_data) / 255.0
    normalized_data = (255 - line_data)
    pad_rows = max(0, desired_shape[0] - normalized_data.shape[0])
    pad_cols = max(0, desired_shape[1] - normalized_data.shape[1])

    padded_array = np.pad(normalized_data, ((0, pad_rows), (0, pad_cols)), mode='constant')
    return padded_array


def select_train_validation_lines(user_image, train_split, shuffle):
    rows = []
    for i, line in enumerate(user_image.get_all_lines()):
        idx = i + 1
        if not user_image.is_test_line(idx) and not is_empty_line(line):
            rows.append(idx)
    if shuffle:
        random.shuffle(rows)
    split_idx = int(len(rows) * train_split)
    # print((rows[0:split_idx], rows[split_idx:]))
    return (rows[0:split_idx], rows[split_idx:])


def split_and_shuffle_array(arr, split_points):
    chunks = split_array(arr, split_points)
    np.random.shuffle(chunks)
    shuffled_array = np.concatenate(chunks, axis=1)
    return shuffled_array


def split_array(arr, split_points):
    chunks = []
    start_idx = 0
    for end_idx in split_points:
        chunk = arr[:, start_idx:end_idx]
        chunks.append(chunk)
        start_idx = end_idx
    last_chunk = arr[:, start_idx:]
    chunks.append(last_chunk)
    return chunks


def create_thumbnail(image_array, target_size, data_augmentation=True):
    height, width = image_array.shape
    target_height, target_width = target_size
    org_image = Image.fromarray(image_array)

    if data_augmentation:
        random_scale_w = random.uniform(0.85, 1.15)
        random_scale_h = random.uniform(0.85, 1.15)
        random_rotate = random.randint(-10, 10)
        org_image_rs = org_image.resize((int(width * random_scale_w), int(height * random_scale_h)), Image.NEAREST)
        org_image_ro = org_image_rs.rotate(random_rotate, Image.NEAREST, expand=True)
    else:
        org_image_ro = org_image

    canvas = Image.new("L", (height, height), 0)
    if width < height:
        left = (height - width) // 2
        top = 0
        canvas.paste(org_image_ro, (left, top))
    else:
        scale_factor = height / width
        s_width = round(scale_factor * width)
        s_height = round(scale_factor * height)
        resized_image = org_image_ro.resize((s_width, s_height), Image.NEAREST)
        left = (height - s_width) // 2
        top = 0
        canvas.paste(resized_image, (left, top))
        del resized_image

    if data_augmentation:
        del org_image_rs
        del org_image_ro
    del org_image

    thumbnail = canvas.resize((target_width, target_height), Image.NEAREST)
    return thumbnail


def show_strip(the_images):
    l = len(the_images)
    plt.clf()
    fig, axs = plt.subplots(1, l, figsize=(15, 25))
    for i in range(0, l):
        img = the_images[i]
        axs[i].imshow(img, cmap='gray')
        axs[i].axis('off')
    plt.show()


def show_sequence(the_images):
    l = len(the_images)
    dim = math.ceil(math.sqrt(l))
    plt.clf()
    fig, axs = plt.subplots(dim, dim, figsize=(10, 10))
    k = 0
    for i in range(0, dim):
        for j in range(0, dim):
            img = the_images[k]
            axs[i, j].imshow(img, cmap='gray')
            axs[i, j].axis('off')
            k += 1
            if k == l:
                plt.show()
                return


def show_triplet(triplets):
    plt.clf()
    fig, axs = plt.subplots(1, 3, figsize=(5, 5))

    for k in range(0, 3):
        img = triplets[k]
        axs[k].imshow(img, cmap='gray')
        axs[k].axis('off')

    plt.show()


def pad_sequence(max_sequence_length, sequence, image_height, image_width, num_channels):
    sequence = np.asarray(sequence)
    padding_size = max_sequence_length - len(sequence)
    if padding_size > 0:
        padding_shape = (padding_size, image_height, image_width, num_channels)
        padding_images = np.zeros(padding_shape)
        padded_sequence = np.concatenate([sequence, padding_images], axis=0)
    else:
        padded_sequence = np.asarray(sequence)
    return padded_sequence


def pad_sequences(max_length, sequences, image_height, image_width, num_channels):
    # Pad sequences to have the same length (pad with zero images)
    padded_sequences = []
    for sequence in sequences:
        seq_len = len(sequence)
        if seq_len == 0:
            continue
        if seq_len > max_length:
            sequence = sequence[:max_length]
            seq_len = len(sequence)

        num_padding = max_length - seq_len
        if num_padding > 0:
            sequence = np.concatenate(
                [sequence, np.zeros((num_padding, image_height, image_width, num_channels))])
        padded_sequences.append(sequence)
    return np.array(padded_sequences)

