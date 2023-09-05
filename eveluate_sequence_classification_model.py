from sklearn import metrics
from preprocessing.utils import *
from concurrent.futures import ThreadPoolExecutor
from preprocessing.dataset import DataSet
from models.options import ModelOptions
import pickle
from tensorflow.keras.utils import Sequence, to_categorical, plot_model
from tensorflow.keras.models import Sequential, Model,load_model

full_data_set = DataSet()
model_options= ModelOptions()


class TesingData:
    def __init__(self):
        self.user_ids = []
        self.testing_data = {}

    def prepare_test_lines(self, user_ids):
        self.user_ids = user_ids
        e = ThreadPoolExecutor(max_workers=10)
        futures = [e.submit(full_data_set.get_testing_strip, user_id) for user_id in self.user_ids]
        results = [f.result() for f in futures]

        testing_data = {}
        for user_id, line, split_points in results:
            self.testing_data[user_id] = {
                'line': line,
                'split_points': split_points
            }

    def dump_data(self, output_path):
        with open(MODEL_CHECKPOINT_PATH + output_path, 'wb') as f:
            pickle.dump(self.testing_data, f)

    def load_test_lines(self, stored_path):
        with open(MODEL_CHECKPOINT_PATH + stored_path, 'rb') as f:
            self.testing_data = pickle.load(f)

    def line_generator(self, max_sequence_length, target_size):
        img_path = ORIGINAL_IMAGES

        for user_id in self.testing_data:
            line = self.testing_data[user_id]['line']
            split_points = self.testing_data[user_id]['split_points']
            sequence = []
            if len(split_points) == 0:
                print(f"no split points for user {user_id} ")
                continue
            seq_len = 0
            if len(split_points) > max_sequence_length:
                split_points = random.sample(split_points, max_sequence_length)

            for (x, y, w, h) in split_points:
                if seq_len == max_sequence_length:
                    yield user_id, pad_sequence(max_sequence_length, sequence, target_size[0], target_size[1], 1)
                    sequence = []

                img = line[:, x:x + w]
                thumbnail = create_thumbnail(img, target_size, data_augmentation=False)
                np_im = np.array(thumbnail, dtype=np.float32) / 255
                np_img = np_im.reshape(target_size[0], target_size[1], 1)
                sequence.append(np_img)
                seq_len += 1
            if seq_len > max_sequence_length * 0.7:
                yield user_id, pad_sequence(max_sequence_length, sequence, target_size[0], target_size[1], 1)

    def batch_generator(self, max_batch_size, max_sequence_length, target_size):
        batch = []
        labels = []
        for label, sequence in self.line_generator(max_sequence_length=model_options.max_sequence_length,
                                                   target_size=input_shape):
            batch.append(sequence)
            labels.append(to_categorical(label, num_classes=len(self.testing_data.keys())))
            if len(batch) == max_batch_size:
                yield np.asarray(batch), np.asarray(labels)
                batch = []
                labels = []

        if len(batch) > 0:
            yield np.asarray(batch), np.asarray(labels)




#for dumping data
user_ids = range(0, 200)
td = TesingData()
td.prepare_test_lines(user_ids)
td.dump_data('testing_strips.pkl')






def show_confusion_matrix_v1(all_labels, all_predictions):
    print("Confusion matrix:\n")
    cmatrix = metrics.confusion_matrix(all_labels, all_predictions)
    for i, r in enumerate(cmatrix):
        if i == 0:
            line = ''.ljust(5)
            for user in range(0, len(r)):
                line += str(user).ljust(3)
            print(line)

        line = str(i).ljust(5)
        for c in r:
            if c < 1:
                c = ' '.ljust(3)
            line += str(c).ljust(3)
        print(line)


def show_confusion_matrix(all_labels, all_predictions, num_classes):
    cmatrix = metrics.confusion_matrix(all_labels, all_predictions)

    cmap = plt.cm.get_cmap('Blues')
    plt.figure(figsize=(10, 8))
    plt.imshow(cmatrix, cmap=cmap)

    cbar = plt.colorbar()

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Classes')
    plt.ylabel('True Classes')
    plt.xticks(range(num_classes), rotation=90)
    plt.yticks(range(num_classes))

    plt.tight_layout()
    plt.show()


model_options.max_sequence_length = 30
model_options.batch_size = 10

td = TesingData()
td.load_test_lines('testing_strips.pkl')
print(td.testing_data.keys())

model = load_model(MODEL_CHECKPOINT_PATH + 'backup-dont-delete/FINAL-200-users-sequence-classifier.hdf5')
model.summary()

input_shape = (model_options.image_height, model_options.image_width)
testing_gen = td.batch_generator(max_batch_size=model_options.max_sequence_length,
                                 max_sequence_length=model_options.max_sequence_length, target_size=input_shape)

all_predictions, all_labels = [], []
batch_x = []
for b, (batch_x, labels) in enumerate(testing_gen):
    predict_values = model.predict(batch_x)
    all_labels += np.argmax(labels, axis=1).tolist()
    all_predictions += np.argmax(predict_values, axis=1).tolist()

all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)
accuracy = sum(all_predictions == all_labels) / len(all_labels)
print(f"Accuracy: {accuracy:.4f}")

print("Logistic regression using  features cross validation:\n%s\n" % (
    metrics.classification_report(all_labels, all_predictions)))

show_confusion_matrix(all_labels, all_predictions, 200)

