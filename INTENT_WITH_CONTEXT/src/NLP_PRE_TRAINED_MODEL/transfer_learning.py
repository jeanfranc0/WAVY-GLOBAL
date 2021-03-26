import os
import shutil
import numpy as np
import collections
from sklearn.model_selection import train_test_split
from src.NLP_PRE_TRAINED_MODEL.type_approach import intent_isoladas, intent_query_and_answer, intent_only_query, intent_last_query_and_answer, intent_last_query, intent_last_answer
from classifier.classification import classify_single_BERT, classify_dual_BERT

from collections import OrderedDict
#https://huggingface.co/transformers/multilingual.html
#https://github.com/BramVanroy/bert-for-inference/blob/master/introduction-to-bert.ipynb

def class_and_labels(csv, list_index):
    labels = csv['intent'].tolist()
    nro_clases = len(list(set(labels)))
    K_messagem = {}
    for i in range(1000):
        K_messagem[str(i)] = []
    for i, item in enumerate(list_index):
        K_messagem[str(len(item))] = K_messagem[str(len(item))] + [i]
    return K_messagem, nro_clases

def generate_X_y(list_index, labels):
    new_labels = []
    w = 0
    for j in list_index:
        new_labels.append(labels[w:w + len(j)])
        w = w + len(j)
    return list_index, new_labels

def save_embeddings(b_mean_train = None, c_four_train = None, b_mean_test = None, c_four_test = None, Y_train= None, Y_test=None, path=None, name_folder=None, b_q_a_train = None, c_q_a_train = None, b_q_a_test = None, c_q_a_test = None):
    create_folder(path + '/' + name_folder)
    path = path + '/' + name_folder + '/'
    #print(np.array(b_mean_train).shape)
    #print(np.array(b_mean_test).shape)
    if name_folder.split('_')[0] == 'Single':
        np.save(path + 'X_mean_train.npy' , b_mean_train)
        np.save(path + 'X_four_train.npy' , c_four_train)
        np.save(path + 'X_mean_test.npy', b_mean_test)
        np.save(path + 'X_four_test.npy', c_four_test)
    np.save(path + 'Y_train.npy' , Y_train)
    np.save(path + 'Y_test.npy', Y_test)
    if name_folder.split('_')[0] == 'Dual':
        #print(np.array(b_q_a_train).shape)
        #print(np.array(b_q_a_test).shape)
        np.save(path + 'X_mean_dual_train.npy' , b_q_a_train)
        np.save(path + 'X_four_dual_train.npy' , c_q_a_train)
        np.save(path + 'X_mean_dual_test.npy', b_q_a_test)
        np.save(path + 'X_four_dual_test.npy', c_q_a_test)

def create_folder(path):
    try:
        print(path)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)

def transfer_l(type_embedding, perc_train, csv, list_index, path):
    K_messagem, nro_clases = class_and_labels(csv, list_index)
    labels = csv['intent'].tolist()
    X, y = generate_X_y(list_index, labels)
    X_train_f, X_test_f, Y_train, Y_test = train_test_split(X,y, random_state=2020, train_size=perc_train,stratify=y)
    Y_train = sum(Y_train, [])
    Y_test = sum(Y_test, [])
    path = path + type_embedding 
    create_folder(path)
    print(path)
    print('----------------------------------------------')
    print('---------------- Single BERT -----------------')
    print('----------------------------------------------')
    print('----------------------------------------------')
    print('-------------- intent isoladas ---------------')
    print('----------------------------------------------')
    b_mean_train, c_four_train = intent_isoladas(type_embedding, X_train_f, csv, K_messagem)
    b_mean_test, c_four_test = intent_isoladas(type_embedding, X_test_f, csv, K_messagem)                                                        
    save_embeddings(b_mean_train = b_mean_train, c_four_train = c_four_train, b_mean_test=b_mean_test, c_four_test=c_four_test, Y_train=Y_train, Y_test= Y_test, path=path, name_folder='Single_BERT_intent_isoladas',  b_q_a_train = None, c_q_a_train = None, b_q_a_test = None, c_q_a_test = None)

    #classify_single_BERT(b_mean_train, c_four_train, b_mean_test, c_four_test, Y_train, Y_test, csv, path, folder = 'intent_isoladas')
    print('----------------------------------------------')
    print('---------------query and answer---------------')
    print('----------------------------------------------')
    b_q_a_train, c_q_a_train = intent_query_and_answer(type_embedding, X_train_f, csv, K_messagem, siamesse = None)
    b_q_a_test, c_q_a_test = intent_query_and_answer(type_embedding, X_test_f, csv, K_messagem, siamesse = None)
    save_embeddings(b_mean_train= b_q_a_train, c_four_train = c_q_a_train, b_mean_test = b_q_a_test, c_four_test = c_q_a_test, Y_train =Y_train , Y_test=Y_test, path=path, name_folder='Single_BERT_query_and_answer',  b_q_a_train = None, c_q_a_train = None, b_q_a_test = None, c_q_a_test = None)

    #classify_single_BERT(b_q_a_train, c_q_a_train, b_q_a_test, c_q_a_test, Y_train, Y_test, csv, path, folder = 'query_and_answer')
    print('----------------------------------------------')
    print('---------------- only query ------------------')
    print('----------------------------------------------')
    b_only_q_train, c_only_q_train = intent_only_query(type_embedding, X_train_f, csv, K_messagem, siamesse = None)
    b_only_q_test, c_only_q_test = intent_only_query(type_embedding, X_test_f, csv, K_messagem, siamesse = None)
    save_embeddings(b_mean_train=b_only_q_train, c_four_train=c_only_q_train, b_mean_test=b_only_q_test, c_four_test=c_only_q_test, Y_train=Y_train, Y_test=Y_test,path= path, name_folder='Single_BERT_only_query', b_q_a_train = None, c_q_a_train = None, b_q_a_test = None, c_q_a_test = None)

    #classify_single_BERT(b_only_q_train, c_only_q_train, b_only_q_test, c_only_q_test, Y_train, Y_test, csv, path, folder = 'only_query')
    print('----------------------------------------------')
    print('------------ last query and answer -----------')
    print('----------------------------------------------')
    b_only_q_train, c_only_q_train = intent_last_query_and_answer(type_embedding, X_train_f, csv, K_messagem, siamesse = None)
    b_only_q_test, c_only_q_test = intent_last_query_and_answer(type_embedding, X_test_f, csv, K_messagem, siamesse = None)
    save_embeddings(b_mean_train=b_only_q_train, c_four_train=c_only_q_train, b_mean_test=b_only_q_test, c_four_test=c_only_q_test, Y_train=Y_train, Y_test=Y_test,path= path, name_folder='Single_BERT_last_query_and_answer', b_q_a_train = None, c_q_a_train = None, b_q_a_test = None, c_q_a_test = None)

    #classify_single_BERT(b_only_q_train, c_only_q_train, b_only_q_test, c_only_q_test, Y_train, Y_test, csv, path, folder = 'only_query')
    
    print('----------------------------------------------')
    print('---------------- last query ------------------')
    print('----------------------------------------------')
    b_only_q_train, c_only_q_train = intent_last_query(type_embedding, X_train_f, csv, K_messagem, siamesse = None)
    b_only_q_test, c_only_q_test = intent_last_query(type_embedding, X_test_f, csv, K_messagem, siamesse = None)
    save_embeddings(b_mean_train=b_only_q_train, c_four_train=c_only_q_train, b_mean_test=b_only_q_test, c_four_test=c_only_q_test, Y_train=Y_train, Y_test=Y_test,path= path, name_folder='Single_BERT_last_query', b_q_a_train = None, c_q_a_train = None, b_q_a_test = None, c_q_a_test = None)

    #classify_single_BERT(b_only_q_train, c_only_q_train, b_only_q_test, c_only_q_test, Y_train, Y_test, csv, path, folder = 'only_query')
    
    print('----------------------------------------------')
    print('---------------- last answer -----------------')
    print('----------------------------------------------')
    b_only_q_train, c_only_q_train = intent_last_answer(type_embedding, X_train_f, csv, K_messagem, siamesse = None)
    b_only_q_test, c_only_q_test = intent_last_answer(type_embedding, X_test_f, csv, K_messagem, siamesse = None)
    save_embeddings(b_mean_train=b_only_q_train, c_four_train=c_only_q_train, b_mean_test=b_only_q_test, c_four_test=c_only_q_test, Y_train=Y_train, Y_test=Y_test,path= path, name_folder='Single_BERT_last_answer', b_q_a_train = None, c_q_a_train = None, b_q_a_test = None, c_q_a_test = None)

    #classify_single_BERT(b_only_q_train, c_only_q_train, b_only_q_test, c_only_q_test, Y_train, Y_test, csv, path, folder = 'only_query')
    

    print('----------------------------------------------')
    print('-------------- Dual BERT ---------------------')
    print('----------------------------------------------')
    print('----------------------------------------------')
    print('---------------query and answer---------------')
    print('----------------------------------------------')
    b_q_a_train, c_q_a_train = intent_query_and_answer(type_embedding, X_train_f, csv, K_messagem, siamesse = 'yes')
    b_q_a_test, c_q_a_test = intent_query_and_answer(type_embedding, X_test_f, csv, K_messagem, siamesse = 'yes')                           
    save_embeddings(b_mean_train = None, c_four_train = None, b_mean_test = None, c_four_test = None, Y_train=Y_train, Y_test=Y_test, path=path, name_folder='Dual_BERT_query_and_answer', b_q_a_train = b_q_a_train, c_q_a_train = c_q_a_train, b_q_a_test = b_q_a_test, c_q_a_test = c_q_a_test)
    #classify_dual_BERT(b_mean_train, c_four_train, b_mean_train[0] + b_only_q_train, c_four_train[0] + c_only_q_train, b_mean_test, c_four_test, b_mean_test[0] + b_only_q_test, c_four_test[0] + c_only_q_test, Y_train, Y_test)

    print('----------------------------------------------')
    print('---------------- only query ------------------')
    print('----------------------------------------------')
    b_only_q_train, c_only_q_train = intent_only_query(type_embedding, X_train_f, csv, K_messagem, siamesse = 'yes')
    b_only_q_test, c_only_q_test = intent_only_query(type_embedding, X_test_f, csv, K_messagem, siamesse = 'yes')
    save_embeddings(b_mean_train = None, c_four_train = None, b_mean_test = None, c_four_test = None, Y_train=Y_train, Y_test=Y_test, path=path, name_folder='Dual_BERT_only_query', b_q_a_train = b_only_q_train, c_q_a_train = c_only_q_train, b_q_a_test = b_only_q_test, c_q_a_test = c_only_q_test)
    #classify_dual_BERT(b_mean_train, c_four_train, b_mean_train[0] + b_only_q_train, c_four_train[0] + c_only_q_train, b_mean_test, c_four_test, b_mean_test[0] + b_only_q_test, c_four_test[0] + c_only_q_test, Y_train, Y_test)
    
    print('----------------------------------------------')
    print('------------ last query and answer -----------')
    print('----------------------------------------------')
    b_q_a_train, c_q_a_train = intent_last_query_and_answer(type_embedding, X_train_f, csv, K_messagem, siamesse = 'yes')
    b_q_a_test, c_q_a_test = intent_last_query_and_answer(type_embedding, X_test_f, csv, K_messagem, siamesse = 'yes')                           
    save_embeddings(b_mean_train = None, c_four_train = None, b_mean_test = None, c_four_test = None, Y_train=Y_train, Y_test=Y_test, path=path, name_folder='Dual_BERT_last_query_and_answer', b_q_a_train = b_q_a_train, c_q_a_train = c_q_a_train, b_q_a_test = b_q_a_test, c_q_a_test = c_q_a_test)
    #classify_dual_BERT(b_mean_train, c_four_train, b_mean_train[0] + b_only_q_train, c_four_train[0] + c_only_q_train, b_mean_test, c_four_test, b_mean_test[0] + b_only_q_test, c_four_test[0] + c_only_q_test, Y_train, Y_test)
    
    print('----------------------------------------------')
    print('---------------- last query ------------------')
    print('----------------------------------------------')
    b_q_a_train, c_q_a_train = intent_last_query(type_embedding, X_train_f, csv, K_messagem, siamesse = 'yes')
    b_q_a_test, c_q_a_test = intent_last_query(type_embedding, X_test_f, csv, K_messagem, siamesse = 'yes')                           
    save_embeddings(b_mean_train = None, c_four_train = None, b_mean_test = None, c_four_test = None, Y_train=Y_train, Y_test=Y_test, path=path, name_folder='Dual_BERT_last_query', b_q_a_train = b_q_a_train, c_q_a_train = c_q_a_train, b_q_a_test = b_q_a_test, c_q_a_test = c_q_a_test)
    #classify_dual_BERT(b_mean_train, c_four_train, b_mean_train[0] + b_only_q_train, c_four_train[0] + c_only_q_train, b_mean_test, c_four_test, b_mean_test[0] + b_only_q_test, c_four_test[0] + c_only_q_test, Y_train, Y_test)

    print('----------------------------------------------')
    print('---------------- last answer -----------------')
    print('----------------------------------------------')
    b_q_a_train, c_q_a_train = intent_last_answer(type_embedding, X_train_f, csv, K_messagem, siamesse = 'yes')
    b_q_a_test, c_q_a_test = intent_last_answer(type_embedding, X_test_f, csv, K_messagem, siamesse = 'yes')                           
    save_embeddings(b_mean_train = None, c_four_train = None, b_mean_test = None, c_four_test = None, Y_train=Y_train, Y_test=Y_test, path=path, name_folder='Dual_BERT_last_answer', b_q_a_train = b_q_a_train, c_q_a_train = c_q_a_train, b_q_a_test = b_q_a_test, c_q_a_test = c_q_a_test)
    #classify_dual_BERT(b_mean_train, c_four_train, b_mean_train[0] + b_only_q_train, c_four_train[0] + c_only_q_train, b_mean_test, c_four_test, b_mean_test[0] + b_only_q_test, c_four_test[0] + c_only_q_test, Y_train, Y_test)
