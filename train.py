from generators.sequence_generators import TrainSequenceGenerator, DataGeneratorsCollection, ValidationSequenceGenerator, \
    TestSequenceGenerator
from models.network_configs import ModelOptions
from preprocessing.utils import *
from models.cnn_lstm import CnnLstmAttentionModel as Model
from models.simple_cnn import SimpleCnnModel as Model
from tensorflow import expand_dims

input_shape = (model_options.image_height, model_options.image_width, 1)
user_ids=range(0,50)
#user_ids=range(0,10)
model_options = ModelOptions()
print(model_options)

model = OneLetterClassifierModel().get_model(num_classes=model_options.num_classes , input_shape=input_shape)


filepath = MODEL_CHECKPOINT_PATH + "ft-one-etter-lassifier-model-{epoch:02d}-{loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, mode='min')
callbacks_list = [checkpoint]


valid_gen = LettersGenerator(MODE_VALIDATION, user_ids, model_options, model_options.num_classes )
train_gen = LettersGenerator(MODE_TRAIN, user_ids, model_options, model_options.num_classes)


num_epochs = 5
opt =  Adam(learning_rate=1e-3)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(train_gen, epochs=num_epochs, batch_size=model_options.batch_size,
                    validation_data=valid_gen, verbose=1, callbacks=callbacks_list)






#..............continue training

filepath = MODEL_CHECKPOINT_PATH + "ft-one-letter-classifier-model-{epoch:02d}-{loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, mode='min')
callbacks_list = [checkpoint]


user_ids=[3,4,8]

print(model_options)
valid_gen = LettersGenerator(MODE_VALIDATION, user_ids, model_options, model_options.num_classes )
train_gen = LettersGenerator(MODE_TRAIN, user_ids, model_options, model_options.num_classes)

opt =  Adam(learning_rate=1e-4)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

num_epochs=3
model.load_weights(MODEL_CHECKPOINT_PATH+'ft-one-etter-lassifier-model-04-0.09.hdf5')
history = model.fit(train_gen, epochs=num_epochs,batch_size=model_options.batch_size,
                   validation_data=valid_gen, verbose=1,callbacks=callbacks_list)