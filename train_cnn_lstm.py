from constants.constants import *
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from generators.sequence_generators_v2 import SequenceGenerator
from models.options import ModelOptions
from models.cnn_lstm import CnnLstmAttentionModel
from keras.optimizers import Adam
from keras.models import load_model

filepath = MODEL_CHECKPOINT_PATH + "cnnlstm-{epoch:02d}-{val_accuracy:.2f}.hdf5"
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
    'lstm_units': 128,
    'dense_units': 64,
    'num_classes': model_options.num_classes
}



print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# If a GPU is available, the TensorFlow should default to it
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

input_shape = (model_options.image_height, model_options.image_width, 1)
num_classes = len(user_ids)
train_gen = SequenceGenerator(MODE_TRAIN, user_ids, model_options)
valid_gen = SequenceGenerator(MODE_VALIDATION, user_ids, model_options)

embedding_path = MODEL_CHECKPOINT_PATH + 'model-15-0.46.hdf5'
embedding_model = load_model(embedding_path)


sm = CnnLstmAttentionModel(model_options, embedding_model, 'dense_7')
model = sm.get_model(layers_options)

####################### IF NEED TO LOAD PREVIOUS MODEL ############################


if filepath is not None:
    opt = Adam(learning_rate=1e-5)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
else:
    loaded_model = load_model(filepath)

model.summary()
#history = model.fit(train_gen, epochs=num_epochs, batch_size=model_options.batch_size,
#                    validation_data=valid_gen, verbose=1, callbacks=callbacks_list)
