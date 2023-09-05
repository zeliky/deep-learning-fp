from constants.constants import *
from models.options import ModelOptions
from models.one_letter_classifier_model import OneLetterClassifierModel
from preprocessing.dataset import DataSet
from generators.letters_generators import LettersGenerator

from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

preload_weights_file = ''
#set it to continue training
#preload_weights_file = 'ft-one-letter-classifier-model-01-1.01.hdf5'

model_options = ModelOptions()
input_shape = (model_options.image_height, model_options.image_width, 1)

model = OneLetterClassifierModel().get_model(num_classes=model_options.num_classes , input_shape=input_shape)
model.summary()

full_data_set = DataSet()

user_ids = range(0,10)
input_shape = (model_options.image_height, model_options.image_width, 1)

model = OneLetterClassifierModel().get_model(num_classes=model_options.num_classes , input_shape=input_shape)


filepath = MODEL_CHECKPOINT_PATH + "ft-one-etter-lassifier-model-{epoch:02d}-{loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, mode='min')
callbacks_list = [checkpoint]


valid_gen = LettersGenerator(MODE_VALIDATION, user_ids, model_options, model_options.num_classes)
train_gen = LettersGenerator(MODE_TRAIN, user_ids, model_options, model_options.num_classes)


num_epochs = 5
opt =  Adam(learning_rate=1e-3)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

if preload_weights_file is not None:
    model.load_weights(MODEL_CHECKPOINT_PATH + preload_weights_file)

history = model.fit(train_gen, epochs=num_epochs, batch_size=model_options.batch_size,
                    validation_data=valid_gen, verbose=1, callbacks=callbacks_list)
