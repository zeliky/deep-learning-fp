from models.generators import CustomDataGen
from models.generators import TrainDataGenerator, DataGeneratorsCollection, ValidationDataGenerator
from models.cnn_lstm import get_model
user_ids = [1,5]
num_classes = max(user_ids)+1
batch_size = 50
random_shuffle_amount = 10
num_epochs=3
input_shape = (50,50)


data_generator_collection = DataGeneratorsCollection(user_ids, random_shuffle_amount=0)

train_gen = TrainDataGenerator(user_ids, batch_size, data_generator_collection, input_shape)
valid_gen = ValidationDataGenerator(user_ids, batch_size, data_generator_collection, input_shape)


train_gen.__getitem__(1)
valid_gen.__getitem__(1)

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