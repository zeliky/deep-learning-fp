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

from models.embedding import embedding

full_data_set = DataSet()
filepath = MODEL_CHECKPOINT_PATH + "siamese-model-{epoch:02d}-{loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

num_epochs=20
user_ids =[i for i in range(0,200)]

model_options = ModelOptions(
    num_classes=len(user_ids),
    batch_size=100,
    random_shuffle_amount=1,
    alpha = 0.2,
    embedding_dim=128
)

num_classes = model_options.num_classes
train_gen = TripletsGenerator(MODE_TRAIN, user_ids, model_options)
valid_gen = TripletsGenerator(MODE_VALIDATION, user_ids, model_options)
input_shape = (model_options.image_height,model_options.image_width ,1)
embedding_network = EmbeddingModel().get_model(embedding_dim=model_options.embedding_dim , input_shape=input_shape)
sm_network = SiameseModel(model_options, embedding_network)
model = sm_network.get_model()
opt =  Adam(learning_rate=1e-3)
model.compile(optimizer=opt, loss=None)
model.summary()


plot_model(model,show_shapes=True, show_layer_names=True)
history = model.fit(train_gen, epochs=num_epochs,batch_size=model_options.batch_size,
#                   validation_data=valid_gen, verbose=1)
                    verbose=1, callbacks=callbacks_list)





# model.load_weights(MODEL_CHECKPOINT_PATH+'siamese-model-02-0.44.hdf5')