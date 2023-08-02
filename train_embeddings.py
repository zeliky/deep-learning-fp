from constants.constants import *
from keras.callbacks import ModelCheckpoint
from models.options import ModelOptions
from keras.optimizers import Adam
from preprocessing.dataset import DataSet
from generators.letters_generators import TripletsGenerator
from models.embedding import EmbeddingModel
from models.siamese_network import SiameseModel
import random
from tensorflow.keras.utils import Sequence, to_categorical, plot_model

full_data_set = DataSet()
filepath = MODEL_CHECKPOINT_PATH + "siamese-model-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

num_epochs=3
#user_ids =[i for i in range(1,3)]
user_ids = [random.randint(1,40),random.randint(41,80)]
model_options = ModelOptions(
    num_classes=len(user_ids),
    batch_size=100,
    random_shuffle_amount=1,
    alpha = 0.2,
    embedding_dim=512
)

num_classes = model_options.num_classes
train_gen = TripletsGenerator(MODE_TRAIN, user_ids, model_options)
valid_gen = TripletsGenerator(MODE_VALIDATION, user_ids, model_options)

sm_network = SiameseModel(model_options)
model = sm_network.get_model()
opt =  Adam(learning_rate=1e-3)
model.compile(optimizer=opt, loss=None)
model.summary()

plot_model(model,show_shapes=True, show_layer_names=True)
history = model.fit(train_gen, epochs=num_epochs,batch_size=model_options.batch_size,
                   validation_data=valid_gen, verbose=1)