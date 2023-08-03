from constants.constants import *
from generators.letters_generators import LettersGenerator
from preprocessing.utils import show_sequence
def display_letters():
  train_gen = LettersGenerator(MODE_TRAIN, user_ids, model_options, len(user_ids))
  valid_gen = LettersGenerator(MODE_VALIDATION, user_ids, model_options, len(user_ids))

  train_gen.on_epoch_end()
  for i, (batch_x, labels) in enumerate(train_gen):
    #print(labels)
    #pass
    show_sequence(batch_x)
    if i==5:
      break
    #for id, letter in enumerate(batch_x):
    #  print(labels[id])
    #  image_dots(letter.squeeze()*255)
    #  if i==10:
    #    break

    #  show_line(letter.squeeze())
    #break

  train_gen.on_epoch_end()

#full_data_set = DataSet()
display_letters()