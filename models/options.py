class ModelOptions:
    def __init__(self, **kwargs):
        self.num_classes = kwargs.get('num_classes', 200)
        self.batch_size = kwargs.get('batch_size', 100)
        self.image_height = kwargs.get('image_height', 150)
        self.image_width = kwargs.get('image_width', 150)
        self.num_channels = kwargs.get('num_channels', 1)
        self.max_sequence_length = kwargs.get('max_sequence_length', 30)
        self.random_shuffle_amount = kwargs.get('random_shuffle_amount', 1)
        #self.lstm_units = kwargs.get('lstm_units', 5)
        self.max_embedding_samples = kwargs.get('max_embedding_samples', 5)
        self.alpha = kwargs.get('alpha', 0.2)
        self.embedding_dim = kwargs.get('embedding_dim', 64)

    def __repr__(self):
        return str(self.__dict__)