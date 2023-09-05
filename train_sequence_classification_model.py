from constants.constants import *
from models.options import ModelOptions
from models.one_letter_classifier_model import SequenceClassificationModel
from preprocessing.dataset import DataSet
from generators.letters_generators import SequenceGenerator
from tensorflow.keras.models import Sequential, Model,load_model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


user_ids = range(0,80)
letter_model = load_model(MODEL_CHECKPOINT_PATH+'ft-one-letter-classifier-model-02-1.79.hdf5')
#preload_weights_file = ''
#set it to continue training
preload_weights_file = 'backup-dont-delete/FINAL-200-users-sequence-classifier.hdf5'

model_options = ModelOptions()

print(model_options)
model_options.max_sequence_length = 30
model_options.batch_size = 10

model = SequenceClassificationModel().get_model(model_options , letter_model, slice_layer='last-flatten')
model = load_model(MODEL_CHECKPOINT_PATH+'backup-dont-delete/FINAL-200-users-sequence-classifier.hdf5')


filepath = MODEL_CHECKPOINT_PATH + "sequence-classifier-model-{epoch:02d}-{loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, mode='min')
callbacks_list = [checkpoint]


valid_gen = SequenceGenerator(MODE_VALIDATION, user_ids, model_options)
train_gen = SequenceGenerator(MODE_TRAIN, user_ids, model_options)


num_epochs = 4
opt =  Adam(learning_rate=1e-3)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(train_gen, epochs=num_epochs, batch_size=model_options.batch_size,
                    validation_data=valid_gen, verbose=1, callbacks=callbacks_list)
