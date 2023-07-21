from preprocessing.user_dataset import UserDataset
from models.options import ModelOptions


class DataGeneratorsCollection:
    def __init__(self, options: ModelOptions):
        self.active_generators = None
        self.random_shuffle_amount = options.random_shuffle_amount
        self.input_shape = (options.image_height,options.image_width)
        self.users_ds = {}
        self.reset_generators()

    def reset_generators(self):
        self.active_generators = {
            'train': {},
            'valid': {},
            'test': {}
        }

    def get_user_ds(self, user_id):
        if user_id not in self.users_ds:
            uds = UserDataset(user_id)
            uds.warmup()
            self.users_ds[user_id] = uds
        return self.users_ds[user_id]

    def train_lines_amount(self, user_id):
        uds = self.get_user_ds(user_id)
        return len(uds.train_lines)

    def validation_lines_amount(self, user_id):
        uds = self.get_user_ds(user_id)
        return len(uds.validation_lines)

    def get_train_generator(self, user_id):
        if user_id not in self.active_generators['train']:
            uds = self.get_user_ds(user_id)
            self.active_generators['train'][user_id] = uds.get_train_data(target_size=self.input_shape)
        return self.active_generators['train'][user_id]

    def get_validation_generator(self, user_id):
        if user_id not in self.active_generators['valid']:
            uds = self.get_user_ds(user_id)
            self.active_generators['valid'][user_id] = uds.get_validation_data(target_size=self.input_shape)
        return self.active_generators['valid'][user_id]

    def get_test_generator(self, user_id):
        if user_id not in self.active_generators['test']:
            uds = self.get_user_ds(user_id)
            self.active_generators['test'][user_id] = uds.get_testing_data(target_size=self.input_shape)
        return self.active_generators['test'][user_id]

    def deactivate_generator(self, gtype, user_id):
        if user_id in self.active_generators[gtype]:
            del self.active_generators[gtype][user_id]
