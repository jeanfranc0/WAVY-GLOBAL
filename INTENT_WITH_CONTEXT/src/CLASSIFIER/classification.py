import os 
import time
import torch
import random
import numpy as np 
from src.CLASSIFIER.ml_models import compute_classification

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def classify_single_BERT(X_mean_train, X_four_train, X_mean_test, X_four_test, y_train, y_test, csv, path, folder):
    print(X_mean_train.shape, X_mean_test.shape)
    print(X_four_train.shape, X_four_test.shape)
    print(y_train.shape, y_test.shape)
    seed_everything(seed=42)
    print(X_mean_train[0])
    print(y_train)
    compute_classification(X_mean_train, y_train, X_mean_test, y_test, csv, path, folder + 'mean',X_dual_train = None, X_dual_test = None)
    #compute_classification(X_four_train, y_train, X_four_test, y_test, csv, path, folder + 'four',X_dual_train = None, X_dual_test = None)

def classify_dual_BERT(X_mean_train, X_four_train, X_mean_test, X_four_test, y_train, y_test, X_mean_dual_train, X_four_dual_train, X_mean_dual_test, X_four_dual_test, csv, path, folder):
    print(X_mean_train.shape, X_mean_test.shape)
    print(X_four_train.shape, X_four_test.shape)
    print(X_mean_dual_train.shape, X_mean_dual_test.shape)
    print(X_four_dual_train.shape, X_four_dual_test.shape)
    print(y_train.shape, y_test.shape)
    seed_everything(seed=42)
    compute_classification(X_mean_train, y_train, X_mean_test, y_test, csv, path, folder + 'mean', X_dual_train = X_mean_dual_train, X_dual_test = X_mean_dual_test)
    compute_classification(X_four_train, y_train, X_four_test, y_test, csv, path, folder + 'four', X_dual_train = X_four_dual_train, X_dual_test = X_four_dual_test)


