def display_triplets():

  train_gen = TripletsGenerator(MODE_TRAIN, user_ids, model_options)
  valid_gen = TripletsGenerator(MODE_VALIDATION, user_ids, model_options)

  train_gen.on_epoch_end()
  for i, batch_x in enumerate(train_gen):
    anchors, possitives, negatives = batch_x[0]
    for i in range(len(anchors)):
        show_triplet([anchors[i], possitives[i],negatives[i]])

    if i==5:
      break
    break
    #for id, letter in enumerate(batch_x):
    #  print(labels[id])
    #  #image_dots(letter.squeeze()*255)
    #  show_line(letter.squeeze())
    #break


    #break
  train_gen.on_epoch_end()

#full_data_set = DataSet()
display_triplets()