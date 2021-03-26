import argparse
import numpy as np
from src.CLASSIFIER.classification import classify_single_BERT, classify_dual_BERT

def convert_to_array(filename):
    X_train = np.load(filename)
    return X_train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['Single', 'Dual'], default='Single')
    parser.add_argument("X_mean_train_filename", type=str, help="Dataset train file name (*.npy)")
    parser.add_argument("X_four_train_filename", type=str, help="Dataset train file name (*.npy)")
    parser.add_argument("X_mean_test_filename", type=str, help="Dataset test file name (*.npy)")
    parser.add_argument("X_four_test_filename", type=str, help="Dataset test file name (*.npy)")
    parser.add_argument("Y_train_filename", type=str, help="Label train filename (*.npy)")
    parser.add_argument("Y_test_filename", type=str, help="Label test filename (*.npy)")
    parser.add_argument("csv", type=str, help="csv file (*.csv)")
    parser.add_argument("path", type=str, help="path to save classification")
    parser.add_argument("approach_name", type=str, help="Type approach_name")

    args = parser.parse_args()
    X_mean_train_filename = args.X_mean_train_filename
    X_four_train_filename = args.X_four_train_filename
    X_mean_test_filename = args.X_mean_test_filename
    X_four_test_filename = args.X_four_test_filename
    Y_train_filename = args.Y_train_filename
    Y_test_filename = args.Y_test_filename
    
    csv = args.csv
    path = args.path
    approach_name = args.approach_name
    action = str(args.action)

    if action == 'Dual':
        args = parser.parse_args()
        X_mean_dual_train_filename = '/home/jeanfranco/Movile_project/Semi_supervised_learning/src_chatbot/results/embeddings/bert_base_multilingual_uncased/Dual_BERT_only_query/X_mean_dual_train.npy'
        X_four_dual_train_filename = '/home/jeanfranco/Movile_project/Semi_supervised_learning/src_chatbot/results/embeddings/bert_base_multilingual_uncased/Dual_BERT_only_query/X_four_dual_train.npy'
        X_mean_dual_test_filename = '/home/jeanfranco/Movile_project/Semi_supervised_learning/src_chatbot/results/embeddings/bert_base_multilingual_uncased/Dual_BERT_only_query/X_mean_dual_test.npy'
        X_four_dual_test_filename = '/home/jeanfranco/Movile_project/Semi_supervised_learning/src_chatbot/results/embeddings/bert_base_multilingual_uncased/Dual_BERT_only_query/X_four_dual_test.npy'
    
    X_mean_train = convert_to_array(X_mean_train_filename)
    X_four_train = convert_to_array(X_four_train_filename)
    X_mean_test = convert_to_array(X_mean_test_filename)
    X_four_test = convert_to_array(X_four_test_filename)
    Y_train = convert_to_array(Y_train_filename).astype(np.int32)
    Y_test = convert_to_array(Y_test_filename).astype(np.int32)

    if action == 'Dual':
        X_mean_dual_train = convert_to_array(X_mean_dual_train_filename)
        X_four_dual_train = convert_to_array(X_four_dual_train_filename)
        X_mean_dual_test = convert_to_array(X_mean_dual_test_filename)        
        X_four_dual_test = convert_to_array(X_four_dual_test_filename)        
        classify_dual_BERT(X_mean_train, X_four_train, X_mean_test, X_four_test, Y_train, Y_test, X_mean_dual_train, X_four_dual_train, X_mean_dual_test, X_four_dual_test, csv, path, approach_name)
    else:
        classify_single_BERT(X_mean_train, X_four_train, X_mean_test, X_four_test, Y_train, Y_test, csv, path, approach_name)
        

if __name__ == "__main__":
    main()
