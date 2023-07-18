from models.generators import CustomDataGen
from models.cnn_lstm import get_model
user_ids = [1,5]
num_classes = max(user_ids)+1
batch_size = 500
random_shuffle_amount = 10
num_epochs=3


train_gen = CustomDataGen(user_ids, batch_size, random_shuffle_amount, CustomDataGen.TYPE_TRAIN)
valid_gen = CustomDataGen(user_ids, batch_size, random_shuffle_amount, CustomDataGen.TYPE_TRAIN)



#model = get_model(num_classes=num_classes, input_shape = (225, 4965,3) )
model = get_model(num_classes=num_classes, input_shape = (25, 25,3) )
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

"""
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