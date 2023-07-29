from constants.constants import *
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from generators.letters_generators import LettersGenerator
from models.options import ModelOptions
from models.custom_cnn import CustomCNN
from keras.optimizers import Adam
from keras.models import load_model

filepath = MODEL_CHECKPOINT_PATH + "model-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

num_epochs = 100
user_ids = [i for i in range(1, 10)]

model_options = ModelOptions(
    num_classes=len(user_ids),
    batch_size=1000,
    image_height=150,
    image_width=150,
    num_channels=1,
    max_sequence_length=40,
    random_shuffle_amount=32
)

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
    'fc_layers': 3,  # number of fully connected layers
    'fc_units': [1024, 512, model_options.num_classes],  # number of units in each fully connected layer
    'fc_activation': 'relu',  # activation function for the fully connected layers
    'dropout_rate': 0.1,  # dropout rate
    'num_classes': model_options.num_classes  # number of classes in the output layer
}

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# If a GPU is available, the TensorFlow should default to it
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

input_shape = (model_options.image_height, model_options.image_width, 1)
num_classes = len(user_ids)
train_gen = LettersGenerator(MODE_TRAIN, user_ids, model_options)
valid_gen = LettersGenerator(MODE_VALIDATION, user_ids, model_options)

sm = CustomCNN(input_shape)
model = sm.get_model(layers_options)

####################### IF NEED TO LOAD PREVIOUS MODEL ############################
filepath = MODEL_CHECKPOINT_PATH + 'model-30-0.46.hdf5'

if filepath is not None:
    opt = Adam(learning_rate=1e-5)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
else:
    loaded_model = load_model(filepath)

model.summary()
history = model.fit(train_gen, epochs=num_epochs, batch_size=model_options.batch_size,
                    validation_data=valid_gen, verbose=1, callbacks=callbacks_list)
