import pandas as pd
import collections
import numpy as np
import matplotlib.pyplot as plt

def load_data(path):
    csv = pd.read_csv(path)
    #print(csv)
    messages = csv['query'].tolist()
    labels = csv['intent'].tolist()
    #labels = [intent.strip() for intent in labels]
    clases = list(set(labels))
    print('number of classes', len(clases))
    labels = [clases.index(intent) for intent in labels]
    counter=collections.Counter(labels)
    print(counter)
    return messages, labels

def load_data_fine_tunning(path):
    csv = pd.read_csv(path)
    print(csv)
    csv = csv.drop('vacio', 1)
    csv = csv[~csv['query'].isna()]
    csv = csv[~csv['intent'].isna()]
    print(csv)
    messages = csv['query'].tolist()
    labels = csv['intent'].tolist()
    #labels = [intent.strip() for intent in labels]
    clases = list(set(labels))
    print('number of classes', len(clases))
    labels = [clases.index(intent) for intent in labels]
    counter=collections.Counter(labels)
    print(counter)
    messages = np.asarray(messages)
    labels = np.asarray(labels)
    return messages, labels, len(clases)

def load_data_atis(path):
    message_train, intent_train = get_intent_and_query(path + 'datasets.atis.train.csv')
    message_test, intent_test = get_intent_and_query(path + 'datasets.atis.test.csv')
    message_dev, intent_dev = get_intent_and_query(path + 'datasets.atis.dev.csv')
    return message_train, intent_train, message_test, intent_test, message_dev, intent_dev 

def load_new_dataset(path):
    message_train, intent_train = get_intent_and_query_new_dataset(path + 'sample_train.csv')
    message_test, intent_test = get_intent_and_query_new_dataset(path + 'sample_test.csv')
    message_dev, intent_dev = get_intent_and_query_new_dataset(path + 'sample_valid.csv')
    return message_train, intent_train, message_test, intent_test, message_dev, intent_dev 

def get_intent_and_query_new_dataset(path):
    csv = pd.read_csv(path)
    messages = csv['text'].tolist()
    labels = csv['intent'].tolist()
    #labels = [intent.strip() for intent in labels]
    clases = list(set(labels))
    print('number of classes', len(clases))
    labels = [clases.index(intent) for intent in labels]
    counter = collections.Counter(labels)
    plt.bar(counter.keys(), counter.values())
    plt.show()
    print(counter)
    return messages, labels

def get_intent_and_query(path):
    csv = pd.read_csv(path)
    messages = csv['tokens'].tolist()
    labels = csv['intent'].tolist()
    #labels = [intent.strip() for intent in labels]
    clases = list(set(labels))
    labels = [clases.index(intent) for intent in labels]
    counter=collections.Counter(labels)
    print(counter)
    return messages, labels

def load_data_atis_finne_tu8ning(path):
    csv = pd.read_csv(path)
    messages = csv['tokens'].tolist()
    labels = csv['intent'].tolist()
    #labels = [intent.strip() for intent in labels]
    clases = list(set(labels))
    labels = [clases.index(intent) for intent in labels]
    counter=collections.Counter(labels)
    print(counter)
    return messages, labels, len(clases)


#import collections
#import matplotlib.pyplot as plt
#l = ['a', 'b', 'b', 'b', 'c']
#w = collections.Counter(l)
#