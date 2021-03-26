#Elmo and word2vect embedding http://vectors.nlpl.eu/repository/
#fastText https://github.com/davidsbatista/website/blob/3d557343ad9c9d05be0e48c621aa4464a484ea50/_posts/2019-11-03-Portuguese-Embeddings.md
#Elmo https://github.com/HIT-SCIR/ELMoForManyLangs

import os
import numpy as np
from nltk import word_tokenize
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = stopwords.words('portuguese')
from PREPROCESSING.cleaning import clean_text
from allennlp.commands.elmo import ElmoEmbedder

#you need to change the path to models' weight 
DIR = "/home/jeanfranco/Movile_project/Semi_supervised_learning/data/embedding_data/"
DIR_en = "/home/jeanfranco/Movile_project/Semi_supervised_learning/data/Embedding_data_english/"
options_file = '/home/jeanfranco/Movile_project/Semi_supervised_learning/data/embedding_data/Elmo/Elmo_weight/elmo_pt_options.json'
weight_file = '/home/jeanfranco/Movile_project/Semi_supervised_learning/data/embedding_data/Elmo/Elmo_weight/elmo_pt_weights.hdf5'
embeddings_index = {}

def CarregarWordEmbeddings(embedding):
    if embedding == 'Glove':
        embedding = 'Glove/glove_s300.txt'
    if embedding == 'Word2Vec':
        embedding = 'Word2Vec/cbow_s300.txt'
    if embedding == 'FastText':
        embedding = 'FastText/cbow_s300.txt'
    if embedding == 'Wang2Vec':
        embedding = 'Wang2Vec/cbow_s300.txt'
    f = open(os.path.join(DIR,embedding), encoding='utf8', errors='replace')
    print('Indexing word vectors.')
    for line in f:
        try:
            values = line.split()
            palavra = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings_index[palavra] = embedding
        except:
            pass
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index

def CarregarWordEmbeddings_english(embedding):
    if embedding == 'Glove':
        embedding = 'Glove/glove.6B.300d.txt'
    if embedding == 'Word2Vec':
        embedding = 'Word2Vec/1960.txt'
    if embedding == 'FastText':
        embedding = 'FastText/model.txt'
    f = open(os.path.join(DIR_en,embedding), encoding='utf8', errors='replace')
    print('Indexing word vectors.')
    for line in f:
        try:
            values = line.split()
            palavra = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings_index[palavra] = embedding
        except:
            pass
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index

def Elmo2word(list_querys):
    try:
        words = str(list_querys).lower()
        words = word_tokenize(words)
        embedding = Embedder(Elmo_DIR)
        embedd = [np.mean(embeds, axis=0) for embeds in embedding.sents2elmo(words)]
    except:        
        continue

    embedd = np.array(embedd)
    v = embedd.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(1024)
    return v / np.sqrt((v ** 2).sum())

def Elmo_embedding(s):
    try:
        elmo_embedding = ElmoEmbedder(options_file=options_file,weight_file=weight_file)
        words = clean_text(s)
        words = word_tokenize(words)
        words = [w for w in words if not w in stop_words]
        words = [w for w in words if w.isalpha()]
        vectors = elmo_embedding.embed_sentence(words)
    except:
    embedd = np.array(vectors)
    v = embedd.sum(axis=0)  
    if type(v) != np.ndarray:
        return np.zeros(1024)
    return embedd
    

def sent2vec(embeddings_index, s ):
    words = clean_text(s)
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(RetornarVetor(embeddings_index, w))
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v ** 2).sum())

def RetornarVetor(embeddings_index, palavra):
    return embeddings_index[palavra]  # return embedding for each word
