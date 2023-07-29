from generators.sequence_generators_v2 import SequenceGenerator
from preprocessing.dataset import DataSet
from preprocessing.utils import *
from constants.constants import *
from models.options import ModelOptions

user_ids = [19]
num_classes = max(user_ids) + 1

num_epochs = 3

model_options = ModelOptions(
    num_classes=len(user_ids) ,
    batch_size=100,
    image_height=50,
    image_width=50,
    num_channels=1,
    max_sequence_length=50,
    random_shuffle_amount=1
)


def display_sequences():
    train_gen = SequenceGenerator(MODE_TRAIN, user_ids, model_options)
    valid_gen = SequenceGenerator(MODE_VALIDATION, user_ids, model_options)

    train_gen.on_epoch_end()
    for i, (batch_x, labels) in enumerate(train_gen):
        for id, sequence in enumerate(batch_x):
            print(f"line sequence from user {labels[id]}")
            if id == 5:
                break
            show_sequence(sequence * 255)

        # break
    train_gen.on_epoch_end()


full_data_set = DataSet()
display_sequences()
