layers_options = {
    'depth':5,  # number of convolutional layers
    'filters': [32,64,128,256,64],  # number of filters for each conv layer
    'kernel_sizes': [(10,10),(5,5), (3,3), (3,3), (1,1 )],  # filter sizes
    'strides': [(3, 3),(3, 3),(1, 1),(3, 3),(1, 1)],  # strides for each conv layer
    'padding': [ 'same','same','same','same','same' ],  # padding for each conv layer
    'conv_activation': 'relu',  # activation function for the convolutional layers
    'pooling': [True,True,True,False,False],  # whether to include a pooling layer after each conv layer
    'pool_sizes': [(3, 3),(3, 3),(3, 3),(3, 3)],  # sizes of the pooling filters
    'pool_strides': [(2, 2),(2, 2),(2, 2),(2, 2),(2, 2)],  # strides for each pooling layer
    'fc_layers': 1,  # number of fully connected layers
    'fc_units': [256],  # number of units in each fully connected layer
    'fc_activation': 'relu',  # activation function for the fully connected layers
    'dropout_rate': 0.2,  # dropout rate
    'num_classes':model_options.num_classes   # number of classes in the output layer
}