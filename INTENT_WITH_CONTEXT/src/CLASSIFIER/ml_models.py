import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn import svm
from datetime import timedelta
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

from plot.plot_cm import plot_confusion_matrix
from models.cnn_archi import TrueCNN

EPOCHS = 5
BATCH_SIZE = 32

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    warnings.filterwarnings("ignore")
    fxn()

#Hyperparameters
cv = 5    

#—————— Estimator with LinearSVC
estimator_LinearSVC = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, 
                      C=1.0, multi_class='ovr', fit_intercept=True,
                      intercept_scaling=1, class_weight='balanced', verbose=0, random_state=None, max_iter=1000)

#—————— GridSearch with parameters
paramGrid_LinearSVC = {"C": [0.01, 0.05, 0.1, 0.5, 1, 5, 10],
             "loss":['hinge', 'squared_hinge'],
             "max_iter": [500, 1000, 5000],
             "tol": [0.00001, 0.0001, 0.001, 0.01, 0.1]}

#—————— Estimator with RandomForestClassifier
estimator_RandomForestClassifier = { "n_estimators": [30, 90] }

#—————— Estimator with MLPClassifier
parameters_MLPClassifier = [{
        'hidden_layer_sizes': [(100, 50)],
        'activation': ['relu'],
        'solver': ['sgd', 'adam'],
        'learning_rate_init': [.01],
        'learning_rate': ['adaptive'],
        'max_iter': [1000]
    }]

#—————— Estimator with n_neighbors
grid_knn = {'n_neighbors':[1,2,3,4,5,6,7,8,9]}

#—————— Estimator with LogisticRegression
parameters_lr = [{'C': [1,1.5,2]}]        

classifiers = {
    'smv' : GridSearchCV(estimator=estimator_LinearSVC, param_grid=paramGrid_LinearSVC, cv=cv),
    'MLP' : GridSearchCV(estimator=MLPClassifier(), param_grid=parameters_MLPClassifier, cv=cv),
    'Knn' : GridSearchCV(estimator=KNeighborsClassifier(), param_grid=grid_knn, cv=cv),    
    'LR' : GridSearchCV(estimator=LogisticRegression(), param_grid=parameters_lr,  cv=cv),
    'RF' : GridSearchCV(estimator=RandomForestClassifier(), param_grid=estimator_RandomForestClassifier, cv=cv)
}

def compute_classification(X_train, y_train, X_test, y_test, csv, path, folder, X_dual_train = None, X_dual_test = None):
    csv = pd.read_csv(csv)
    path = path + folder
    create_folder(path)
    path = path + '/'
    map_label = dict(enumerate(csv.intent.factorize()[1]))
    log_cols=["Classifier", "Accuracy", "Recall", 'Precision', 'f1_score']
    log = pd.DataFrame(columns=log_cols)
    print('llego')
    compute_classification_architecture(X_train, y_train, X_test, y_test, csv, len(map_label), path, folder, map_label, X_dual_train = X_dual_train, X_dual_test = X_dual_test)
    if X_dual_train is None:
        pass
    else:
        X_train = np.concatenate([X_train, X_dual_train])
        X_test = np.concatenate([X_test, X_dual_test])
    for name in classifiers:
        start_time = time.time()
        print("="*30)
        print(name)
        print("="*30)
        clf = classifiers[name]
        clf.fit(X_train, y_train)
        #testing
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose()
        df.to_csv(path + 'Single_BERT' + '_' + folder + '_' + name + '_' + 'report'+ '.csv')
        acc, recall, precision, f1_scor = accuracy_metrics(y_test, y_pred)
        log_entry = pd.DataFrame([[name, convert_to_percent(acc), convert_to_percent(recall), convert_to_percent(precision), convert_to_percent(f1_scor)]], columns=log_cols)
        log = log.append(log_entry)
        log.to_csv(path + 'Single_BERT' + '_' + folder + '_' + name + '.csv')
        #plot_confusion_matrix(y_test, y_pred, map_label, path, folder)
        end_time = time.time()
        time_dif = end_time - start_time
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    print('clases')
    print(len(map_label))
    print(map_label)
    

def compute_classification_architecture(X_train, y_train, X_test, y_test, csv, num_classes, path, folder, map_label, X_dual_train = None, X_dual_test = None):
    #training
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    y_train = to_categorical(y_train)
    #testing
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    y_test = to_categorical(y_test)
    if X_dual_train is None:
        tCNN = TrueCNN(X_train, num_classes, query_answer=None).model
        history = tCNN.fit(x=X_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, y_test), shuffle=True)
        print('Evaluate')
        print('Test')
        print(tCNN.evaluate([X_test], y_test))
        # evaluate the model
        print('Train')
        print(tCNN.evaluate([X_train], y_train, verbose=0))
        pred_test = np.argmax(tCNN.predict(X_test), axis=1)
        pred_train = np.argmax(tCNN.predict(X_train), axis=1)
    else:
        tCNN = TrueCNN(X_train, num_classes, query_answer=X_dual_train).model
        history = tCNN.fit(x=[X_train, X_dual_train], y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=([X_test, X_dual_test], y_test), shuffle=True)
        print('Evaluate')
        print('Test')
        print(tCNN.evaluate([X_test, X_dual_test], y_test))
        # evaluate the model
        print('Train')
        print(tCNN.evaluate([X_train, X_dual_train], y_train, verbose=0))
        ##predict
        pred_test = np.argmax(tCNN.predict([X_test, X_dual_test]), axis=1)
        
        pred_train = np.argmax(tCNN.predict([X_train, X_dual_train]), axis=1)
    y_test = np.argmax(y_test, axis=1)
    y_train = np.argmax(y_train, axis=1)

    print('Predict')
    print('Test')
    print(y_test)
    print(pred_test)
    print(classification_report([map_label[i] for i in y_test], [map_label[i] for i in pred_test]))
    print('Train')
    print(y_train)
    print(pred_train)
    print(classification_report([map_label[i] for i in y_train], [map_label[i] for i in pred_train]))

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

def create_folder(path):
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)


def convert_to_percent(array):
    return array*100

def accuracy_metrics(y_test, prediction):   
    acc = accuracy_score(y_test, prediction)
    print("Accuracy: {:.4%}".format(acc))
    
    recall = recall_score(y_test, prediction, average="macro")
    print("Recall: {:.4%}".format(recall))
    
    precision = precision_score(y_test, prediction, average="macro")
    print("Precision: {:.4%}".format(precision))

    f1_scor = f1_score(y_test, prediction, average="macro")
    print("f1_score: {:.4%}".format(f1_scor)) 
    return acc, recall, precision, f1_scor
