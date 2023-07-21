class ModelOptions:
    def __init__(self, **kwargs):
        self.num_classes = kwargs.get('num_classes', 10)
        self.batch_size = kwargs.get('batch_size', 100)
        self.image_height = kwargs.get('image_height', 50)
        self.image_width = kwargs.get('image_width', 50)
        self.num_channels = kwargs.get('num_channels', 1)
        self.max_sequence_length = kwargs.get('max_sequence_length', 32)
        self.random_shuffle_amount = kwargs.get('random_shuffle_amount', 0)
        self.lstm_units = kwargs.get('lstm_units', 5)

    def __repr__(self):
        return str(self.__dict__)