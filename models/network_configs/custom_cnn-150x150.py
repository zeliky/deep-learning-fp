layers_options = {
    'depth': 5,  # number of convolutional layers
    'filters': [96, 256, 384, 384, 256],  # number of filters for each conv layer
    'kernel_sizes': [(11, 11), (5, 5), (3, 3), (3, 3), (3, 3)],  # filter sizes
    'strides': [(4, 4), (1, 1), (1, 1), (1, 1), (1, 1)],  # strides for each conv layer
    'padding': ['valid', 'same', 'same', 'same', 'same'],  # padding for each conv layer
    'conv_activation': 'relu',  # activation function for the convolutional layers
    'pooling': [True, True, False, False, True],  # whether to include a pooling layer after each conv layer
    'pool_sizes': [(3, 3), (3, 3), None, None, (3, 3)],  # sizes of the pooling filters
    'pool_strides': [(2, 2), (2, 2), None, None, (2, 2)],  # strides for each pooling layer
    'fc_layers': 2,  # number of fully connected layers
    'fc_units': [150, model_options.num_classes],  # number of units in each fully connected layer
    'fc_activation': 'relu',  # activation function for the fully connected layers
    'dropout_rate': 0.1,  # dropout rate
    'num_classes': model_options.num_classes   # number of classes in the output layer
}