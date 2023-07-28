layers_options = {
    'depth': 5,
    'filters': [96, 256, 384, 384, 256],
    'kernel_sizes': [(4, 4), (3, 3), (3, 3), (3, 3), (3, 3)],
    'strides': [(2, 2), (1, 1), (1, 1), (1, 1), (1, 1)],
    'padding': ['valid', 'same', 'same', 'same', 'same'],
    'conv_activation': 'relu',
    'pooling': [True, True, False, False, True],
    'pool_sizes': [(2, 2), (2, 2), None, None, (2, 2)],
    'pool_strides': [(2, 2), (2, 2), None, None, (2, 2)],
    'fc_layers': 3,
    'fc_units': [1024, 1024, model_options.num_classes],
    'fc_activation': 'relu',
    'dropout_rate': 0.1,
    'num_classes': model_options.num_classes
}