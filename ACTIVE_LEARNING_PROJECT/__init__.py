# import python modules defined by BERT
#ber uncased in atis, no use bert cased
import argparse
import time
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import timedelta
from CLASSIFIER.classifiers import Active_Learning, Supervised_Learning
#from classifiers_bert import Active_Learning_bert_multi
import pandas as pd
# modify the default parameters of np.load
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

def calculate_active_learning(X_train, Y_train, X_test, Y_test, csv, num_classes, num_train, learning_type, X_val= None, Y_val= None):
    Active_Learning(X_train, Y_train, X_test, Y_test, csv, num_classes, num_train, learning_type, X_val, Y_val)

def calculate_Supervised_learning(X_train, Y_train, X_test, Y_test, csv, num_classes, num_train, learning_type, embedding_type, X_val=None, Y_val=None):
    Supervised_Learning(X_train, Y_train, X_test, Y_test, csv + '_supervised_learning', num_classes, num_train, learning_type, embedding_type, X_val, Y_val)

from pathlib import Path

def test_something(path):
    path = Path(path)
    assert path.is_file()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("X_train_filename", type=str, help="Dataset train file name (*.npy)")
    parser.add_argument("Y_train_filename", type=str, help="Label train filename (*.npy)")
    parser.add_argument("X_test_filename", type=str, help="Dataset test file name (*.npy)")
    parser.add_argument("Y_test_filename", type=str, help="Label test filename (*.npy)")
    parser.add_argument("X_val_filename", type=str, help="Dataset val file name (*.npy)")
    parser.add_argument("Y_val_filename", type=str, help="Label val filename (*.npy)")
    parser.add_argument("embedding_type", type=str, help="types of embeddings", choices=['Atis','balanced', 'new_dataset'])
    parser.add_argument("learning_type", type=str, help="types of classifier(AL: active learning, SL: supervised learning)", choices=['SL', 'AL'])
    parser.add_argument("path", type=str, help="Path to save csv")
    args = parser.parse_args()
    
    X_train_filename = args.X_train_filename
    Y_train_filename = args.Y_train_filename
    X_test_filename = args.X_test_filename
    Y_test_filename = args.Y_test_filename
    X_val_filename = args.X_val_filename
    Y_val_filename = args.Y_val_filename
    embedding_type = args.embedding_type
    learning_type = args.learning_type
    path = args.path
    X_train = np.load(X_train_filename)
    Y_train = np.load(Y_train_filename).astype(np.int32)
    X_test = np.load(X_test_filename)
    Y_test = np.load(Y_test_filename).astype(np.int32)

    X_val = np.load(X_val_filename)
    Y_val = np.load(Y_val_filename).astype(np.int32)


    # start time
    start_time = time.time()
    
    num_train = X_train.shape[0]
    num_classes = len(np.unique(Y_train))
    num_test = X_test.shape[0]
    print("Read dataset Ok")
    print("num_train {}".format(num_train))
    print("num_test {}".format(num_test))
    print('num classes {}'.format(num_classes))

    print("="*30)
    print(embedding_type)
    print("="*30)
    csv = path + embedding_type
    if learning_type == 'SL':
        calculate_Supervised_learning(X_train, Y_train, X_test, Y_test, csv, num_classes, num_train, learning_type,embedding_type, X_val, Y_val)         
    else:
        calculate_active_learning(X_train, Y_train, X_test, Y_test, csv, num_classes, num_train, embedding_type, X_val, Y_val)

    # end time
    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    
    # restore np.load for future normal usage
    np.load = np_load_old
    
if __name__ == "__main__":
    main()
