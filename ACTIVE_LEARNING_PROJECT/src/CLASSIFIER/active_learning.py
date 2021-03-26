from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
#Active learning
from modAL.uncertainty import uncertainty_sampling, margin_sampling, entropy_sampling, classifier_uncertainty #https://www.sciencedirect.com/science/article/pii/S0020025516313949
from modAL.batch import ranked_batch #https://www.sciencedirect.com/science/article/pii/S0020025516313949
from modAL.batch import select_cold_start_instance #https://www.sciencedirect.com/science/article/pii/S0020025516313949
from modAL.batch import select_instance #https://www.sciencedirect.com/science/article/pii/S0020025516313949
from modAL.batch import uncertainty_batch_sampling

from modAL.models import ActiveLearner, Committee
from ALMa import ActiveLearningManager

import matplotlib as mpl
import matplotlib.pyplot as plt

from functools import partial
import time
import pandas as pd
import random
import copy
import warnings
from tensorflow.keras import regularizers
# Seed value (can actually be different for each attribution step)
seed_value= 0

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
#tf.random.set_seed(seed_value) # tensorflow 2.x
tf.set_random_seed(seed_value) # tensorflow 1.x

from keras.layers import (concatenate, Flatten, AveragePooling1D,add, Concatenate,
                          Reshape, Conv1D, Dense, MaxPool1D,MaxPooling1D,Activation,
                          Dropout, GlobalAveragePooling1D, GlobalMaxPooling1D)

from keras.utils import to_categorical
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras import (Input, Model)

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

RANDOM_STATE_SEED = 1
np.random.seed(RANDOM_STATE_SEED)

class ResidualConv1D:
    """
    ***ResidualConv1D for use with best performing classifier***
    """

    def __init__(self, filters, kernel_size, pool=False):
        self.pool = pool
        self.kernel_size = kernel_size
        self.params = {
            "padding": "same",
            "kernel_initializer": "he_uniform",
            "strides": 1,
            "filters": filters,
        }

    def build(self, x):

        res = x
        if self.pool:
            x = MaxPooling1D(1, padding="same")(x)
            res = Conv1D(kernel_size=1, **self.params)(res)

        out = Conv1D(kernel_size=1, **self.params)(x)

#         out = BatchNormalization(momentum=0.9)(out)
        out = Activation("relu")(out)
        out = Conv1D(kernel_size=self.kernel_size, **self.params)(out)

#         out = BatchNormalization(momentum=0.9)(out)
        out = Activation("relu")(out)
        out = Conv1D(kernel_size=self.kernel_size, **self.params)(out)

        out = add([res, out])

        return out

    def __call__(self, x):
        return self.build(x)

def make_pretty_summary_plot(performance_history,N_QUERIES):
    with plt.style.context("seaborn-white"):
        fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)

        ax.plot(performance_history)
        ax.scatter(range(len(performance_history)), performance_history, s=13)

        ax.xaxis.set_major_locator(
            mpl.ticker.MaxNLocator(nbins=N_QUERIES + 3, integer=True)
        )
        ax.xaxis.grid(True)

        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10))
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
        ax.set_ylim(bottom=0, top=1)
        ax.yaxis.grid(True, linestyle="--", alpha=1 / 2)

        ax.set_title("Incremental classification accuracy")
        ax.set_xlabel("Query iteration")
        ax.set_ylabel("Classification Accuracy")

        plt.show()

def split_train_validation(num_split, X_training, y_training):
    X_train = X_training[:num_split]
    y_train = y_training[:num_split]
    X_validation = X_training[num_split:]
    y_validation = y_training[num_split:]
    return X_train, y_train, X_validation, y_validation

def active_learning_scores(manager, num_train, X_train, y_train, X_testing, y_testing, learner, num_classes,learning_type, X_validation, y_validation, csv, name):
    if learning_type == 'Atis':
        BATCH_SIZE_1 = 500
    elif learning_type == 'new_dataset':
        BATCH_SIZE_1 = 3000
    else:
        BATCH_SIZE_1 = 2000
    #validation
    log_cols_validation=["Classifier", "Accuracy", "Recall", 'Precision', 'f1_score']
    log_validation = pd.DataFrame(columns=log_cols_validation)
    scores_acc_val = []
    scores_recall_val = []
    scores_precision_val = []
    scores_f1_score_val = []
    performance_history_val = []
    #test
    scores_acc = []
    scores_recall = []
    scores_precision = []
    scores_f1_score = []
    performance_history = []
    
    print(num_train,BATCH_SIZE_1)
    N_QUERIES = num_train//BATCH_SIZE_1
    print(N_QUERIES)
    for i in range(N_QUERIES):
        print(i)
        if manager.unlabeld.size == 0:
            break
        for index in range(1):
            indices_to_label, query_instance = learner.query(manager.unlabeld)
            labels = []  # Hold a list of the new labels
            i = 0
            for ix in indices_to_label:
                #print('aea2', i)
                #i+=1
                """
                Here is the tricky part that the manager solves. The indicies are indexed with respect to unlabeled data
                but we want to work with them with respect to the original data. The manager makes this almost transparent
                """
                # Map the index that is with respect to unlabeled data back to an index with respect to the whole dataset
                original_ix = manager.get_original_index_from_unlabeled_index(ix)
                #print(original_ix)
                #print(manager.sources[original_ix]) #Show the original data so we can decide what to label
                # Ahora podemos buscar la etiqueta en el conjunto original de etiquetas sin ningún tipo de contabilidad
                # Now we can lookup the label in the original set of labels without any bookkeeping
                y = y_train[original_ix]
                #print(y)
                # We create a Label instance, a tuple of index and label
                # The index should be with respect to the unlabeled data, the add_labels function will automatically
                # calculate the offsets
                label = (ix, y)
                #print(label)
                # append the labels to a list
                labels.append(label)
                #print(labels)
                #print('aea33')
            # Insert them all at once.
            manager.add_labels(labels)
            # Note that if you need to add labels with indicies that repsect the original dataset you can do
            # manager.add_labels(labels,offset_to_unlabeled=False)
        #######################################
        learner.teach(manager.labeled, manager.labels)
        #######################################
        # VALIDATION
        model_accuracy_val = learner.score(X_validation, y_validation)
        performance_history_val.append(learner.score(X_validation, y_validation))
        acc_val, recall_val, precision_val, f1_score_val = active_learning_metrics(X_validation, y_validation, learner)
        scores_acc_val.append(acc_val)
        scores_recall_val.append(recall_val)
        scores_precision_val.append(precision_val)
        scores_f1_score_val.append(f1_score_val)
        print('validation', performance_history_val)
        
        # TEST
        performance_history.append(learner.score(X_testing, y_testing))
        model_accuracy = learner.score(X_testing, y_testing)
        acc, recall, precision, f1_score = active_learning_metrics(X_testing, y_testing, learner)
        scores_acc.append(acc)
        scores_recall.append(recall)
        scores_precision.append(precision)
        scores_f1_score.append(f1_score)
        print('testing', performance_history)
    
    log_entry_val = pd.DataFrame([[name, convert_to_percent(scores_acc_val), convert_to_percent(scores_recall_val), convert_to_percent(scores_precision_val), convert_to_percent(scores_f1_score_val)]], columns=log_cols_validation)
    log = log_validation.append(log_entry_val)
    log.to_csv(csv + '_validation' + '.csv')    
    print('sale')
    # Finnaly make a nice plot
    #make_pretty_summary_plot(performance_history, N_QUERIES)
    return scores_acc, scores_recall, scores_precision, scores_f1_score     

def active_learning_metrics(X_testing, y_testing, learner):
    predictions = learner.predict(X_testing)    
    print('****Results****')    
    acc = accuracy_score(y_testing, predictions)
    print("Accuracy: {:.4%}".format(acc))
    
    recall = recall_score(y_testing, predictions, average="macro")
    print("Recall: {:.4%}".format(recall))
    
    precision = precision_score(y_testing, predictions, average="macro")
    print("Precision: {:.4%}".format(precision))

    f1_scor = f1_score(y_testing, predictions, average="macro")
    print("f1_score: {:.4%}".format(f1_scor))
        
    return acc, recall, precision, f1_scor

def convert_to_percent(array):
    return [i*100 for i in array]

def scores_active_learning_save(manager, name, csv, log_cols, log, num_train, X_train, y_train, X_testing, y_testing, learner, num_classes, learning_type, X_validation, y_validation):
    scores_acc, scores_recall, scores_precision, scores_f1_score = active_learning_scores(manager, num_train,X_train, y_train, X_testing, y_testing, learner, num_classes, learning_type, X_validation, y_validation,csv, name)
    log_entry = pd.DataFrame([[name, convert_to_percent(scores_acc), convert_to_percent(scores_recall), convert_to_percent(scores_precision), convert_to_percent(scores_f1_score)]], columns=log_cols)
    log = log.append(log_entry)
    log.to_csv(csv + '_test' + '.csv')

def prepare_manager(X_train):
    manager = ActiveLearningManager(X_train)
    return manager

def prepare_learner(query,learning_type):
    if learning_type == 'Atis':
        BATCH_SIZE_1 = 500
    elif learning_type == 'new_dataset':
        BATCH_SIZE_1 = 3000
    else:
        BATCH_SIZE_1 = 2000
    preset_batch = partial(query, n_instances=BATCH_SIZE_1)
    return preset_batch

def ranked_batch_al(name, csv, log_cols, log, clf, X_train, y_train, num_train, X_testing, y_testing, num_classes, learning_type, X_val= None, Y_val= None):
    if learning_type == 'Atis':
        BATCH_SIZE_1 = 500
    elif learning_type == 'new_dataset':
        BATCH_SIZE_1 = 3000
    else:
        BATCH_SIZE_1 = 2000
    N_QUERIES = num_train//BATCH_SIZE_1
    num_split = N_QUERIES * BATCH_SIZE_1
    if learning_type == 'new_dataset':
        X_validation, y_validation = X_val, Y_val
    else:
        X_train, y_train, X_validation, y_validation = split_train_validation(num_split, X_train, y_train)
    query_strats = [uncertainty_sampling, uncertainty_batch_sampling]
    namess = ['uncertainty_sampling', 'batch_sampling']
    for query, namess in zip(query_strats,namess):
        manager = prepare_manager(X_train)
        preset_batch = prepare_learner(query, learning_type)
        learner = ActiveLearner(estimator=clf,query_strategy=preset_batch)
        scores_active_learning_save(manager, name, csv + '_' + namess, log_cols, log, num_train, X_train, y_train, X_testing, y_testing, learner, num_classes, learning_type, X_validation, y_validation)

def calculate_active_learning_to_architecture_cnn(X_train, Y_train, X_test, Y_test, csv, num_classes, num_train, learning_type, X_val= None, Y_val= None):
    print('entrooo')
    start_time = time.time()
    log_cols = ["Classifier", "Accuracy", "Recall", 'Precision', 'f1_score']
    log = pd.DataFrame(columns=log_cols)
    name = 'architecture_cnn'
    Active_Learning_architecure_cnn(name, csv + name, log_cols, log, X_train, Y_train, num_train, X_test, Y_test, num_classes, learning_type, X_val= None, Y_val= None)
    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


def Active_Learning_architecure_cnn(name, csv, log_cols, log, X_train, y_train, num_train, X_testing, y_testing, num_classes, learning_type, X_val= None, Y_val= None):
    if learning_type == 'Atis':
        X_validation, y_validation = X_testing, y_testing
        n_initial = 500
    elif learning_type == 'new_dataset':
        X_validation, y_validation = X_val, Y_val
        n_initial = 1000
    else:
        n_initial = 2000
        N_QUERIES = num_train//n_initial
        num_split = N_QUERIES * n_initial        
        X_train, y_train, X_validation, y_validation = split_train_validation(num_split, X_train, y_train)
    #
    new_label_y_test = np.array(y_testing)
    new_label_y_validation = np.array(y_validation)
    #to categorical label
    y_train = to_categorical(y_train)
    y_testing = to_categorical(y_testing)
    y_validation = to_categorical(y_validation)
    #model
    tCNN = KerasClassifier(build_fn = Build_model, news_input_shape= X_train, num_classes=num_classes , verbose=0, epochs=50, batch_size=32)
    
    # assemble initial data
    initial_idx = np.random.choice(range(len(X_train)), size=n_initial, replace=False)
    X_initial = X_train[initial_idx]
    y_initial = y_train[initial_idx]

    # generate the pool
    # remove the initial data from the training dataset
    """
    Training the ActiveLearner
    """
    # Pre-set our batch sampling to retrieve 3 samples at a time.
    BATCH_SIZE_CNN = n_initial

    query_strats = [uncertainty_sampling, uncertainty_batch_sampling]
    for q_s in query_strats:
        print('q_s', q_s)
        print('q_s', q_s)
        print('q_s', q_s)
        # initialize Data
        X_pool = np.delete(X_train, initial_idx, axis=0)
        y_pool = np.delete(y_train, initial_idx, axis=0)
        performance_history = []
        scores_acc = []
        scores_recall = []
        scores_precision = []
        scores_f1_score = []
        
        # initialize ActiveLearner
        preset_batch = partial(q_s, n_instances=BATCH_SIZE_CNN)
        learner = ActiveLearner(estimator = tCNN, X_training = X_initial, y_training = y_initial, verbose = 1,query_strategy=preset_batch)

        # Pool-based sampling
        N_QUERIES = num_train//BATCH_SIZE_CNN
        unqueried_score = learner.score(X_testing, y_testing)
        acc, recall, precision, f1_scor = active_learning_metrics(X_testing, new_label_y_test, learner)
        scores_acc.append(acc)
        scores_recall.append(recall)
        scores_precision.append(precision)
        scores_f1_score.append(f1_scor)
        performance_history = [unqueried_score]
        print(performance_history)
        print('N_QUERIES', N_QUERIES)

        #validation
        log_cols_validation=["Classifier", "Accuracy", "Recall", 'Precision', 'f1_score']
        log_validation = pd.DataFrame(columns=log_cols_validation)
        scores_acc_val = []
        scores_recall_val = []
        scores_precision_val = []
        scores_f1_score_val = []
        performance_history_val = []

        for index in range(N_QUERIES - 1):
            query_index, query_instance = learner.query(X_pool,n_instances=BATCH_SIZE_CNN)

            # Teach our ActiveLearner model the record it has requested.
            X, y = X_pool[query_index], y_pool[query_index]
            learner.teach(X=X, y=y)

            # Remove the queried instance from the unlabeled pool.
            X_pool = np.delete(X_pool, query_index, axis=0)
            y_pool = np.delete(y_pool, query_index, axis=0)

            # Calculate and report our model's accuracy.
            model_accuracy = learner.score(X_testing, y_testing)
            print('Accuracy after query {n}: {acc:0.4f}'.format(n=index + 1, acc=model_accuracy))

            # Save our model's performance for plotting.
            #TEST
            performance_history.append(learner.score(X_testing, new_label_y_test))
            acc, recall, precision, f1_scor = active_learning_metrics(X_testing, new_label_y_test, learner)
            scores_acc.append(acc)
            scores_recall.append(recall)
            scores_precision.append(precision)
            scores_f1_score.append(f1_scor)
            print('testing', performance_history)
    
            # VALIDATION
            performance_history_val.append(learner.score(X_validation, new_label_y_validation))
            acc_val, recall_val, precision_val, f1_score_val = active_learning_metrics(X_validation, new_label_y_validation, learner)
            scores_acc_val.append(acc_val)
            scores_recall_val.append(recall_val)
            scores_precision_val.append(precision_val)
            scores_f1_score_val.append(f1_score_val)
            print('validation', performance_history_val)

        #TEST SAVE
        log_entry = pd.DataFrame([[name, convert_to_percent(scores_acc), convert_to_percent(scores_recall), convert_to_percent(scores_precision), convert_to_percent(scores_f1_score)]], columns=log_cols)
        log = log.append(log_entry)
        log.to_csv(csv + '_' + str(q_s) + '.csv')    
        print('sale')

        #VALIDATION SAVE
        log_entry_val = pd.DataFrame([[name, convert_to_percent(scores_acc_val), convert_to_percent(scores_recall_val), convert_to_percent(scores_precision_val), convert_to_percent(scores_f1_score_val)]], columns=log_cols_validation)
        log = log_validation.append(log_entry_val)
        log.to_csv(csv + + '_' + str(q_s) + '_validation' + '.csv')    
        print('sale')

def Build_model(news_input_shape, num_classes):
    model = Sequential()
    #model.add(AveragePooling1D(pool_size=3, strides=1, input_shape=(news_input_shape.shape[1], 1))
    
    model.add(Conv1D(filters=1, kernel_size=1, name='gaa', input_shape=(news_input_shape.shape[1], 1)))
    model.add(Conv1D(filters=4, kernel_size=4, name='0_conv_conc1'))
    model.add(Conv1D(filters=5, kernel_size=3, name='0_conv_conc2'))
    model.add(Conv1D(filters=1, kernel_size=1, name='0_conv_conc3'))
    model.add(Flatten())

    model.add(Dense(units=320, activation='relu', name='dense_layer_200'))
    model.add(Reshape(target_shape=(320,1)))
    model.add(Conv1D(filters=3, kernel_size=3, name='1_conv_conc'))
    model.add(Conv1D(filters=3, kernel_size=3, name='2_conv_conc'))
    model.add(MaxPooling1D(pool_size=3, strides=1, name='first_avgPool'))
    model.add(Conv1D(filters=3, kernel_size=3, name='2_conv_sconc'))
    model.add(MaxPooling1D(pool_size=3, strides=1, name='first_avgsssPsool'))
    model.add(Flatten())

    model.add(Dropout(rate=0.3))
    model.add(Dense(units=180, activation='relu', name="1_dense_layer"))
    model.add(Reshape(target_shape=(180, 1)))
    model.add(MaxPooling1D(pool_size=3, strides=1, name='first_avgssPosol'))
    model.add(Conv1D(filters=3, kernel_size=2))
    model.add(Conv1D(filters=5, kernel_size=5))
    model.add(MaxPooling1D(pool_size=3, strides=1, name='first_avgPosol1'))
    model.add(Flatten())

    model.add(Dense(units=120, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
    model.add(Dense(units=num_classes, activation='relu', name="2_dense_layer", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                        bias_regularizer=regularizers.l2(1e-4),
                        activity_regularizer=regularizers.l2(1e-5)))
    #model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])

    model.summary()
    return model


def plot_scores(scores):
    # Plot our performance over time.
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)
    
    ax.plot(scores)
    ax.scatter(range(len(scores)), scores, s=13)
    
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10))
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
    
    ax.set_ylim(bottom=0, top=1)
    ax.grid(True)
    
    ax.set_title('Incremental classification accuracy')
    ax.set_xlabel('Query iteration')
    ax.set_ylabel('Classification Accuracy')