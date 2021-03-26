#Sklearn
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import pyplot
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from datetime import timedelta
import time
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV

from functools import partial
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from nltk.classify.scikitlearn import SklearnClassifier
from tensorflow.keras import regularizers

import pandas as pd
import numpy as np

from active_learning import ranked_batch_al, calculate_active_learning_to_architecture_cnn
from keras import (Input, Model)

import warnings
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import (concatenate, Flatten, AveragePooling1D,add, Concatenate,
                          Reshape, Conv1D, Dense, MaxPool1D,MaxPooling1D,Activation,
                          Dropout, GlobalAveragePooling1D, GlobalMaxPooling1D)
RANDOM_SEED = 42
def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    warnings.filterwarnings("ignore")
    fxn()

classifiers = [
    KNeighborsClassifier(3),
    CalibratedClassifierCV(svm.LinearSVC(C=1.0, max_iter = 100,loss="hinge")),
    #RandomForestClassifier(max_depth=50),
    MLPClassifier(max_iter=100, activation='relu', hidden_layer_sizes=(100,), solver='adam'),
    LogisticRegression()
    #svm.SVC(kernel='rbf',C=1.0, probability=True),
    ]

def randomforest_grid_search(dataset, labels):
    tuned_parameters = [{'n_estimators': [200, 500], 'max_features': ['auto', 'sqrt', 'log2'], 'max_depth' : [4,5,6,7,8],'criterion' :['gini', 'entropy']}]
    clf = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=tuned_parameters, cv= 3)
    clf.fit(dataset, labels)
    return (clf.best_params_['n_estimators'], clf.best_params_['max_features'], 
    clf.best_params_['max_depth'], clf.best_params_['criterion'])

def linearSVM_grid_search(dataset, labels):
    C_s = 10.0 ** np.arange(-1, 3)
    tuned_parameters = [{'C': C_s}]
    clf = GridSearchCV(svm.LinearSVC(C=1), tuned_parameters, cv=3)
    clf.fit(dataset, labels)
    return clf.best_params_['C']

def svm_grid_search(dataset, labels):
    C_s = 10.0 ** np.arange(-1, 3)
    gammas = 10.0 ** np.arange(-1, 3)
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': gammas,'C': C_s}]
    clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=3)
    clf.fit(dataset, labels)
    return (clf.best_params_['C'], clf.best_params_['gamma'])

def choose_classify(clf, i, X_train, y_train):
    if i == 0:
        name = clf
        print(name)
        c = linearSVM_grid_search(X_train, y_train)
        clf = CalibratedClassifierCV(svm.LinearSVC(C=c))
    return clf, name

def Active_Learning(X_training, y_training, X_testing, y_testing, csv, num_classes, num_train, learning_type, X_val= None, Y_val= None):
    log_cols=["Classifier", "Accuracy", "Recall", 'Precision', 'f1_score']
    log = pd.DataFrame(columns=log_cols)
    for i, clf in enumerate(classifiers):
        start_time = time.time()
        name = clf.__class__.__name__
        print("="*30)
        print(name)
        print("="*30)
        print("="*30)
        print('Active_learning')
        print("="*30)
        print("="*30)
        if learning_type == 'new_dataset':
            ranked_batch_al(name, csv + '_' + name, log_cols, log, clf, X_training, y_training, num_train, X_testing, y_testing, num_classes, learning_type, X_val, Y_val)
        else:
            ranked_batch_al(name, csv + '_' + name, log_cols, log, clf, X_training, y_training, num_train, X_testing, y_testing, num_classes, learning_type, X_val= None, Y_val= None)
        
        end_time = time.time()
        time_dif = end_time - start_time
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    print("="*30)
    print("="*30)
    print("="*30)
    X_training = X_training.reshape((X_training.shape[0], X_training.shape[1], 1))
    X_testing = X_testing.reshape((X_testing.shape[0], X_testing.shape[1], 1))
    if learning_type == 'new_dataset':
        X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    name = 'Architecture_CNN___new_version'
    print("="*30)
    print("="*30)
    print(name)
    print("="*30)
    print("="*30)
    if learning_type == 'new_dataset':
        calculate_active_learning_to_architecture_cnn(X_training, y_training, X_testing, y_testing, csv + name, num_classes, num_train, learning_type, X_val, Y_val)
    else:
        calculate_active_learning_to_architecture_cnn(X_training, y_training, X_testing, y_testing, csv + name, num_classes, num_train, learning_type, X_val= None, Y_val= None)


from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def Supervised_Learning(X_training, y_training, X_testing, y_testing, csv, num_classes, num_train, learning_type, embedding_type, X_val=None, Y_val=None):
    #testing
    log_cols=["Classifier", "Accuracy", "Recall", 'Precision', 'f1_score']
    log = pd.DataFrame(columns=log_cols)
    #Validation
    log_cols_val=["Classifier", "Accuracy", "Recall", 'Precision', 'f1_score']
    log_val = pd.DataFrame(columns=log_cols_val)
    for i, clf in enumerate(classifiers):
        start_time = time.time()
        name = clf.__class__.__name__
        print("="*30)
        print(name)
        print("="*30)
        print("="*30)
        print('Supervised Learning')
        print("="*30)
        print("="*30)
        clf.fit(X_training, y_training)
        #testing
        predicted = clf.predict(X_testing)
        acc, recall, precision, f1_scor = accuracy_metrics(y_testing, predicted)
        log_entry = pd.DataFrame([[name, convert_to_percent(acc), convert_to_percent(recall), convert_to_percent(precision), convert_to_percent(f1_scor)]], columns=log_cols)
        log = log.append(log_entry)
        log.to_csv(csv + '_' + name + '.csv') 

        #testing
        predicted_val = clf.predict(X_val)
        acc_val, recall_val, precision_val, f1_scor_val = accuracy_metrics(Y_val, predicted_val)
        log_entry_val = pd.DataFrame([[name, convert_to_percent(acc_val), convert_to_percent(recall_val), convert_to_percent(precision_val), convert_to_percent(f1_scor_val)]], columns=log_cols_val)
        log_val = log_val.append(log_entry_val)
        log_val.to_csv(csv + '_' + name + '_val' + '.csv') 

    #training
    X_training = X_training.reshape((X_training.shape[0], X_training.shape[1], 1))
    y_training = to_categorical(y_training)
    #testing
    X_testing = X_testing.reshape((X_testing.shape[0], X_testing.shape[1], 1))
    y_testing = to_categorical(y_testing)
    tCNN = TrueCNN(X_training, num_classes).model
    #evaluation
    print('history --- ')
    if embedding_type == 'new_dataset':
        X_valing = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
        y_valining = to_categorical(Y_val)
        history = tCNN.fit(x=X_training, y=y_training, batch_size=32, epochs=5, validation_data=(X_valing, y_valining), shuffle=True)
        #plot_training_history(history)
    else:
        history = tCNN.fit(x=X_training, y=y_training, batch_size=32, epochs=5, validation_data=(X_testing, y_testing), shuffle=True)
    print('Evaluate')
    print('Test')
    print(tCNN.evaluate([X_testing], y_testing))
    # evaluate the model
    print('Train')
    print(tCNN.evaluate([X_training], y_training, verbose=0))
    print('validation')
    print(tCNN.evaluate([X_valing], y_valining, verbose=0))
    #val_mse = tCNN.evaluate(X_valing, y_valining, verbose=0)
    '''
    print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))
    print(history.history.keys())
    # plot loss during training
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    # plot mse during training
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    '''

def convert_to_percent(array):
    return array*100

def accuracy_metrics(y_testing, prediction):   
    acc = accuracy_score(y_testing, prediction)
    print("Accuracy: {:.4%}".format(acc))
    
    recall = recall_score(y_testing, prediction, average="macro")
    print("Recall: {:.4%}".format(recall))
    
    precision = precision_score(y_testing, prediction, average="macro")
    print("Precision: {:.4%}".format(precision))

    f1_scor = f1_score(y_testing, prediction, average="macro")
    print("f1_score: {:.4%}".format(f1_scor)) 
    return acc, recall, precision, f1_scor

class TrueCNN:
    def __init__(self, news_input_shape, num_classes):
        print('init')
        self.model = self.archi_inicial_cnn(news_input_shape, num_classes)
    
    def archi_inicial_cnn(self, news_input_shape, num_classes):
        np.random.seed(RANDOM_SEED)
        print('Building CNN model archi_inicial_cnn...')

        news = Input(shape=(news_input_shape.shape[1], 1), name='news_input')
        first_layer = Conv1D(filters=1, kernel_size=1 )(news)
        first_layer = Conv1D(filters=4, kernel_size=4)(first_layer)
        first_layer = Conv1D(filters=5, kernel_size=3 )(first_layer)
        first_layer = Conv1D(filters=1, kernel_size=1 )(first_layer)
        first_layer = Flatten()(first_layer)

        output = Dense(units=320, activation='relu', name='dense_layer_200')(first_layer)
        output = Reshape(target_shape=(320,1))(output)
        output = Conv1D(filters=3, kernel_size=3)(output)
        output = Conv1D(filters=3, kernel_size=3)(output)
        output = MaxPooling1D(pool_size=3, strides=1, name='first_avsgPool')(output)
        output = Conv1D(filters=3, kernel_size=3)(output)
        output = MaxPooling1D(pool_size=3, strides=1, name='first_avgPool')(output)
        output = Flatten()(output)
        
        output = Dropout(rate=0.3)(output)
        output = Dense(units=180, activation='relu',bias_regularizer=regularizers.l2(1e-4), name="1_dense_layer")(output)
        output = Reshape(target_shape=(180, 1))(output)
        output = MaxPooling1D(pool_size=3, strides=1, name='first_avgPool1')(output)
        output = Conv1D(filters=3, kernel_size=2)(output)
        output = Conv1D(filters=5, kernel_size=5)(output)
        output = MaxPooling1D(pool_size=3, strides=1, name='first_avgPoossl1')(output)
        output = Flatten()(output)
        
        output = Dense(units=120, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(output)
        
        output = Dense(units=num_classes, activation='relu', name="2_dense_layer", 
                        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                        bias_regularizer=regularizers.l2(1e-4),
                        activity_regularizer=regularizers.l2(1e-5))(output)

        model = Model(inputs=news, outputs=output)
        #model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']) //https://stackoverflow.com/questions/42081257/why-binary-crossentropy-and-categorical-crossentropy-give-different-performances
        model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['acc',f1_m,precision_m, recall_m])
        model.summary()
        return model
