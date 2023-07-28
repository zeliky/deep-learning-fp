def display_sequences():
  data_generator_collection = DataGeneratorsCollection(options=model_options)

  train_gen = TrainSequenceGenerator(user_ids, model_options, data_generator_collection)
  valid_gen = ValidationSequenceGenerator(user_ids, model_options, data_generator_collection)
  test_gen = TestSequenceGenerator(user_ids, model_options, data_generator_collection)

  train_gen.on_epoch_end()
  for i, (batch_x, labels) in enumerate(train_gen):
    for id, sequence in enumerate(batch_x):
      print(f"line sequence from user {labels[id]}")
      if id ==5:
        break
      show_sequence(sequence*255)

    #break
  train_gen.on_epoch_end()

full_data_set = DataSet()
display_sequences()