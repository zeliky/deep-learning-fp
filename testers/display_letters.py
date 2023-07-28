def display_letters():

  train_gen = LettersGenerator(MODE_TRAIN, user_ids, model_options)
  valid_gen = LettersGenerator(MODE_VALIDATION, user_ids, model_options)

  train_gen.on_epoch_end()
  for i, (batch_x, labels) in enumerate(train_gen):
    #pass
    show_sequence(batch_x)
    if i==10:
     break
    #for id, letter in enumerate(batch_x):
    #  print(labels[id])
    #  #image_dots(letter.squeeze()*255)
    #  show_line(letter.squeeze())
    #break


    #break
  train_gen.on_epoch_end()

full_data_set = DataSet()
display_letters()