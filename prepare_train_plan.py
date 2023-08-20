from constants.constants import *
from preprocessing.train_plan import TrainPlan


def build_training_plans(ids):
    # classification index
    tp = TrainPlan()
    tp.set_options(MODE_TRAIN, ids, 0.8, True)
    tp.prepare_index()
    tp.save_index()

    # embedding index
    tp = TrainPlan()
    tp.set_options(MODE_TRAIN, ids, 1, False)
    tp.prepare_index()
    tp.save_index()

def test_triplets(train_plan, count=20):
    tp = TrainPlan()
    tp.load_train_plan(train_plan)
    for i, triplet in enumerate(tp.triplets_paths_generator()):
        print(triplet)
        if i == count:
            break


def test_sequences(train_plan, count=20, max_sequence_length=10):
    tp = TrainPlan()
    tp.load_train_plan(train_plan)
    for i, sequence in enumerate(tp.sequence_paths_generator(max_sequence_length)):
        print(sequence)
        if i == count:
            break


user_ids = range(0, 10)
#embedding_train_plan = 'train_plan_100-0'
#build_training_plans(user_ids)
#test_triplets(embedding_train_plan)


classification_train_plan = 'train_plan_80-20_shuffled'
build_training_plans(user_ids)
test_sequences(classification_train_plan)


#
# tp.set_options(MODE_TRAIN, user_ids, 0.8, True)
# tp.prepare_index()
# tp.save_index()

# embedding_train_plan = 'train_plan_100-0'
# tp.load_train_plan(embedding_train_plan)
# print(tp.user_ids)
# print(tp.train_index)
# for i, triplet in enumerate(tp.triplets_paths_generator()):
#    print(triplet)
#    if i == 20:
#        break


# classification_train_plan = 'train_plan_80-20_shuffled'
