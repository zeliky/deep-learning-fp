def display_triplets():
  user_ids = [1,5,15,14,46,25,]
  train_gen = TripletsGenerator(user_ids, model_options)
  valid_gen = TripletsGenerator(user_ids, model_options)

  train_gen.on_epoch_end()
  for i, (batch_x, labels) in enumerate(train_gen):
    for id, triplet in enumerate(batch_x):
      show_triplet(triplet)




    #break
  train_gen.on_epoch_end()

#full_data_set = DataSet()
display_triplets()