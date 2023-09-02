user_ids = range(0,50)
letter_model = load_model(MODEL_CHECKPOINT_PATH+'ft-one-etter-lassifier-model-04-0.09.hdf5')
model_options = ModelOptions()

print(model_options)
model_options.max_sequence_length = 20
model_options.batch_size = 10

model = SequenceClassificationModel().get_model(model_options , letter_model, slice_layer='last-maxpooling')


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
