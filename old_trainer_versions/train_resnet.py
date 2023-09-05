from constants.constants import *
from models.options import ModelOptions
from models.resnet import SimplifiedResnet
from generators.letters_generators import LettersGenerator
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

def train_model(model, num_epochs, train_gen, valid_gen, callbacks_list):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(train_gen, epochs=num_epochs, batch_size=model_options.batch_size,
              validation_data=valid_gen, verbose=1, callbacks=callbacks_list)
    return model


model_options = ModelOptions(
    batch_size=100,
    random_shuffle_amount=1,
)

filepath = MODEL_CHECKPOINT_PATH + "incr_model_{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, mode='max')
callbacks_list = [checkpoint]

max_classes = 20
# num_epocs = 2
num_epocs = 1
input_shape = (model_options.image_height, model_options.image_width, 1)

last_save_path = MODEL_CHECKPOINT_PATH + 'incr_model_12_users.h5'
# last_save_path = None
for i in range(13, max_classes):
    model = SimplifiedResnet().get_model(num_classes=max_classes, input_shape=input_shape)
    model.summary()
    if last_save_path is not None:
        print(f'loading weights from {last_save_path}')
        old_model = load_model(last_save_path)
        old_weights = old_model.get_weights()
        model.set_weights(old_weights)
        print(f'doe loading weights')

    user_ids = [i for i in range(0, i)]
    total_users = len(user_ids)
    print(f"start traing on {total_users} users: ({user_ids}) using {num_epocs} epocs")
    train_gen = LettersGenerator(MODE_TRAIN, user_ids, model_options, max_classes)
    valid_gen = LettersGenerator(MODE_VALIDATION, user_ids, model_options, max_classes)
    model = train_model(model, num_epocs, train_gen, valid_gen, callbacks_list)
    last_save_path = MODEL_CHECKPOINT_PATH + f'incr_model_{i}_users.h5'
    print(f"saving weights in {last_save_path}")
    model.save(last_save_path)
    # num_epocs += 2
