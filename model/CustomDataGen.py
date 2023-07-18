from constants import *

class CustomDataGen(Sequence):
  TYPE_TRAIN = 'train'
  TYPE_VALIDATE = 'validate'
  TYPE_TEST = 'test'

  def __init__(self, user_ids, batch_size, random_shuffle_amount, generator_type):
    self.user_ids = user_ids
    self.random_shuffle_amount = random_shuffle_amount
    self.batch_size = batch_size
    self.users_ds = {}
    self.generator_type = generator_type
    self.users_count = len(self.user_ids)
    self.cursor = 0


  def __getitem__(self, index):
    if self.generator_type == self.TYPE_TRAIN:
        allowed_types = TRAIN_TYPES
    elif self.generator_type == self.TYPE_VALIDATE:
       allowed_types = VALIDATE_TYPES

    X_batch , Y_batch = [], []
    samples = 0
    while samples<=self.batch_size :
      ds.reset()
      user_id = self.user_ids[self.cursor]
      uds = UserDataset(user_id)
      uds.warmup()
      user_samples = 0
      uds.split_dataset()
      for l in uds.get_train_data( random_shuffle_amount=random_shuffle_amount, normalized = True):
        h,w  = l.shape
        l.reshape(h,w,1)
        X_batch.append(l)
        Y_batch.append(user_id)
        samples+=1

      self.cursor = (self.cursor+1 ) % self.users_count

    return np.asarray(X_batch), to_categorical(Y_batch, num_classes = max(self.user_ids)+1)

  def __len__(self):
    print(len(self.user_ids) * self.random_shuffle_amount * len(TRAIN_TYPES))
    return len(self.user_ids) * self.random_shuffle_amount * len(TRAIN_TYPES)

  def on_epoch_end(self):
    pass