import pandas as pd
import collections
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

from src.NLP_PRE_TRAINED_MODEL.transfer_learning import transfer_l

def single_conversation(csv, delete = None):
    b = []
    for index, row in csv.iterrows():
        intnt,confidence,time,channel,user,session,query,response = csv.iloc[index]
        list_index = np.asarray(np.where((csv.user == user) & (csv.session == session)))[0]
        b.append(list_index)
    b = list(map(list, OrderedDict.fromkeys(map(tuple, b)).keys()))
    if delete == 'yes':
        i = [w for w in b if len(w) < 3 or len(w) > 15]
        z = [j for s in i for j in s]
        c = list(set(z))
        for l in c:
            csv.drop(l, inplace=True)
    csv.reset_index(drop=True)
    return csv, b

def run(conversation_dir, type_embedding, perc_train, path):
    csv = pd.read_csv(conversation_dir)
    labels = csv['intent'].tolist()
    clases = sorted(set(labels), key=labels.index)
    csv = csv[csv.intent.isin(csv.intent.value_counts().tail(len(clases) + 1).index)].copy() # select 8 low dimensional categories
    map_label = dict(enumerate(csv.intent.factorize()[1]))
    csv['intent'] = csv.intent.factorize()[0]
    csv, b  = single_conversation(csv, delete = None)
    transfer_l(type_embedding, perc_train, csv, b, path)
    
