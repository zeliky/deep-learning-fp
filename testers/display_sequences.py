from generators.sequence_generators_v2 import SequenceGenerator
from preprocessing.dataset import DataSet
from preprocessing.utils import *
from constants.constants import *
from models.options import ModelOptions

def display_squences():
  valid_gen = SequenceGenerator(MODE_VALIDATION, user_ids, model_options)
  train_gen = SequenceGenerator(MODE_TRAIN, user_ids, model_options, valid_gen.set_user_ids)

  for b, (batch_x, labels) in enumerate(valid_gen):
    #print(batch_x.shape)
    print(labels)
    #for i in range(len(batch_x)):
    #  print(batch_x[i].shape)
    #  pass
      #print(f"line from user: {labels[i]}")
      #show_sequence(batch_x[i])
      #if i==5:
      #  break
      #for id, letter in enumerate(batch_x):
      #  print(labels[id])
      #  image_dots(letter.squeeze()*255)
      #  if i==10:
      #    break

      #  show_line(letter.squeeze())
    #break

  train_gen.on_epoch_end()
user_ids = range(0,4)
model_options = ModelOptions()
#full_data_set = DataSet()
display_squences()
