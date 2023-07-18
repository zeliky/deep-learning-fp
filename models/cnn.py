def get_model(input_shape, num_classes):
  model = Sequential()
  model.add(Conv2D(6, (5, 5),activation='relu', padding='same', input_shape=input_shape))
  model.add(MaxPooling2D((2, 2)))
  model.add(Conv2D(4, (3,3), activation='relu'))
  model.add(MaxPooling2D((2, 2)))
  model.add(Flatten())
  model.add(Dropout(0.1))
  model.add(Dense(9, activation='relu'))
  model.add(Dropout(0.1))
  model.add(Dense(num_classes, activation='softmax'))
  return model