from generators.data_generators import TrainDataGenerator, DataGeneratorsCollection, ValidationDataGenerator, \
    TestDataGenerator
from models.options import ModelOptions
from preprocessing.utils import *
from models.cnn_lstm import CnnLstmAttentionModel as Model

user_ids = [1, 5]
num_classes = max(user_ids) + 1

num_epochs = 3

model_options = ModelOptions(
    num_classes=max(user_ids) + 1,
    batch_size=100,
    image_height=50,
    image_width=50,
    num_channels=1,
    max_sequence_length=32,
    random_shuffle_amount=0
)

data_generator_collection = DataGeneratorsCollection(options=model_options)

train_gen = TrainDataGenerator(user_ids, model_options, data_generator_collection)
valid_gen = ValidationDataGenerator(user_ids, model_options, data_generator_collection)
test_gen = TestDataGenerator(user_ids, model_options, data_generator_collection)
classifier = Model(model_options)
model = classifier.get_model()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(train_gen, epochs=num_epochs, batch_size=model_options.batch_size,
                    validation_data=valid_gen, verbose=1)


def sample_inputs(gen):
    train_gen.on_epoch_end()
    for i, (batch_x, labels) in enumerate(gen):
        print(batch_x.shape)

        for id, sequence in enumerate(batch_x):
            print(f"line sequence from user {np.argmax(labels[id])}")
            # if id == 1:
            #    show_sequence(sequence)
            print(sequence.shape)
    train_gen.on_epoch_end()


"""
#model = get_model(num_classes=num_classes, input_shape = (225, 4965,3) )
model = get_model(num_classes=num_classes, input_shape = (25, 25,1) )
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


history = model.fit (train_gen, epochs=num_epochs, batch_size=batch_size,
                    validation_data=valid_gen ,verbose=1)
"""
"""
for b in range(5):
  X_batch, Y_batch = traingen.__getitem__()
  X_batch.shape
  #for i in range(100,130):
  #  print(Y_batch[i])
  #  show_line(X_batch[i])
  print(Y_batch)
"""
