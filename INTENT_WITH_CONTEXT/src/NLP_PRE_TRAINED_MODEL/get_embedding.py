import time
import torch
import numpy as np
from datetime import timedelta
from transformers import BertModel, BertTokenizer, XLMRobertaModel, XLMRobertaTokenizer, XLMModel, XLMTokenizer, DistilBertTokenizer, DistilBertModel, ElectraTokenizer, ElectraModel, BartTokenizer, BartModel
MAX_LENGHT = 512 
def tokenizer_and_model(type_embedding):
    #########
    #PORTUGUESE
    #########
    if type_embedding.split('_')[0] == 'BERT' or type_embedding.split('_')[0] == 'bert':
        if type_embedding == 'BERT_portuguese_large_neural_mind':
            path = '/home/jeanfarfan/bin/Semi_supervised_learning/data/Brazilian_Bert/BERT_large_portuguese/'
        elif type_embedding == 'BERT_portuguese_base_neural_mind':
            path = '/home/jeanfarfan/bin/Semi_supervised_learning/data/Brazilian_Bert/BERT_base_portuguese/'
        elif type_embedding == 'bert_base_multilingual_cased':
            path = 'bert-base-multilingual-cased' 
        elif type_embedding == 'bert_base_multilingual_uncased':
            path = 'bert-base-multilingual-uncased' 
        #load tokenizer and model
        tokenizer = BertTokenizer.from_pretrained(path)
        model = BertModel.from_pretrained(path, output_hidden_states=True, return_dict=True)
        special_tokens_dict = {'additional_special_tokens': ['[USER]','[SYSTEM]']}
        orig_num_tokens = len(tokenizer)
        num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
        total_num_tokens = orig_num_tokens + num_added_tokens
        model.resize_token_embeddings(total_num_tokens)
    elif type_embedding.split('_')[0] == 'xlmroberta':
        if type_embedding == 'xlmroberta_base':
            path = 'xlm-roberta-base'
        elif type_embedding == 'xlmroberta_large':
            path = 'xlm-roberta-large'
        #load tokenizer and model
        tokenizer = XLMRobertaTokenizer.from_pretrained(path)
        model = XLMRobertaModel.from_pretrained(path, output_hidden_states=True, return_dict=True)
    elif type_embedding == 'xlm':
        path = 'xlm-mlm-100-1280'
        #load tokenizer and model
        tokenizer = XLMTokenizer.from_pretrained(path)
        model = XLMModel.from_pretrained(path, output_hidden_states=True, return_dict=True)
    #########
    #ENGLISH
    #########
    elif type_embedding == 'en_bert_base_uncased':
        path = 'bert-base-uncased'
        #load tokenizer and model
        tokenizer = BertTokenizer.from_pretrained(path)
        model = BertModel.from_pretrained(path, output_hidden_states=True, return_dict=True)
    elif type_embedding == 'en_xlm_mlm_enfr_1024':
        path = 'xlm-mlm-enfr-1024'
        #load tokenizer and model
        tokenizer = XLMTokenizer.from_pretrained(path)
        model = XLMModel.from_pretrained(path, output_hidden_states=True, return_dict=True)
    elif type_embedding == 'en_xlm_roberta_base':
        path = 'xlm-roberta-base'
        #load tokenizer and model
        tokenizer = XLMRobertaTokenizer.from_pretrained(path)
        model = XLMRobertaModel.from_pretrained(path, output_hidden_states=True, return_dict=True)
    elif type_embedding == 'distilbert_base_cased':
        path = 'distilbert-base-cased'
        #load tokenizer and model
        tokenizer = DistilBertTokenizer.from_pretrained(path)
        model = DistilBertModel.from_pretrained(path, output_hidden_states=True, return_dict=True)
    elif type_embedding == 'Mobile_Bert':
        path = 'google/mobilebert-uncased'
        #load tokenizer and model
        tokenizer = MobileBertTokenizer.from_pretrained(path)
        model = MobileBertModel.from_pretrained(path, output_hidden_states=True, return_dict=True)
    elif type_embedding == 'Electra':
        path = 'google/electra-small-discriminator'
        #load tokenizer and model
        tokenizer = ElectraTokenizer.from_pretrained(path)
        model = ElectraModel.from_pretrained(path, output_hidden_states=True, return_dict=True)
    elif type_embedding == 'BART':
        path = 'facebook/bart-large'
        #load tokenizer and model
        tokenizer = BartTokenizer.from_pretrained(path)
        model = BartModel.from_pretrained(path, output_hidden_states=True, return_dict=True)

    return tokenizer, model

def return_embedding(out):
    hidden_states = out.hidden_states
    #last layer
    last_state = hidden_states[-1].squeeze().tolist()
    #mean of layers
    sentence_mean_embedding = torch.mean(hidden_states[-1], dim=1).squeeze().tolist()
    # get last four layers 
    last_four_layers = [hidden_states[i] for i in (-1, -2, -3, -4)]
    # cast layers to a tuple and concatenate over the last dimension
    cat_hidden_states = torch.cat(tuple(last_four_layers), dim=-1)
    # take the mean of the concatenated vector over the token dimension
    cat_sentence_embedding = torch.mean(cat_hidden_states, dim=1).squeeze().tolist()

    return last_state, sentence_mean_embedding, cat_sentence_embedding

def feed_bert(type_embedding, query, tokenizer, model, device, type_approach, prev_inten= None, siamesse = None):
    def encoded(type_embedding, input_ids, model, device):
        input_id = input_ids[0]
        if type_embedding in ['BERT_portuguese_large_neural_mind', 'BERT_portuguese_base_neural_mind', 'bert_base_multilingual_cased', 'bert_base_multilingual_uncased']:
            token_type_ids = input_ids[1]
        
        if type_embedding in ['BERT_portuguese_large_neural_mind', 'BERT_portuguese_base_neural_mind', 'bert_base_multilingual_cased', 'bert_base_multilingual_uncased']:
            input_id = torch.LongTensor(input_id)
            #if type_embedding in ['BERT_portuguese_large_neural_mind', 'BERT_portuguese_base_neural_mind', 'bert_base_multilingual_cased', 'bert_base_multilingual_uncased']:
            token_type_ids = torch.LongTensor(token_type_ids)
        else:
            input_id = torch.LongTensor(input_id)

        model = model.to(device)
        input_id = input_id.to(device)
        if type_embedding in ['BERT_portuguese_large_neural_mind', 'BERT_portuguese_base_neural_mind', 'bert_base_multilingual_cased', 'bert_base_multilingual_uncased']:
            token_type_ids = token_type_ids.to(device)

        model.eval()
        input_id = input_id.unsqueeze(0)
        if type_embedding in ['BERT_portuguese_large_neural_mind', 'BERT_portuguese_base_neural_mind', 'bert_base_multilingual_cased', 'bert_base_multilingual_uncased']:
            token_type_ids = token_type_ids.unsqueeze(0)

        with torch.no_grad():
            if type_embedding in ['BERT_portuguese_large_neural_mind', 'BERT_portuguese_base_neural_mind', 'bert_base_multilingual_cased', 'bert_base_multilingual_uncased']:
                out = model(input_ids=input_id, token_type_ids=token_type_ids)
            else:
                out = model(input_ids=input_id)
        return out

    if type_approach == 'isolated':
        sentence_ids = tokenizer('[USER]' + query, add_special_tokens=True, truncation=True)['input_ids'][1:-1]   
        if type_embedding in ['BERT_portuguese_large_neural_mind', 'BERT_portuguese_base_neural_mind', 'bert_base_multilingual_cased', 'bert_base_multilingual_uncased']:            
            segments_ids = [0 for i in range(len(sentence_ids))]
            input_ids = (sentence_ids, segments_ids)
        else:
            input_ids = (sentence_ids, 1)
        return encoded(type_embedding, input_ids, model, device)
    if type_approach == 'query_answer' and siamesse == 'yes':
        q_a_join = []
        segments_ids = []
        for sentence in prev_inten:
            sentenceA, sentenceB = '[USER]' + sentence[0], '[SYSTEM]' + sentence[1]
            sentenceA_ids = tokenizer(sentenceA, add_special_tokens=True, truncation=True)['input_ids']   
            sentenceB_ids = tokenizer(sentenceB, add_special_tokens=True, truncation=True)['input_ids']              
            sentenceB_ids = sentenceB_ids[:30] + [102] if len(sentenceB_ids) > 30 else sentenceB_ids
            segments_ids = segments_ids + [0 for i in range(len(sentenceA_ids))] + [1 for i in range(len(sentenceB_ids))]
            q_a_join = q_a_join + sentenceA_ids + sentenceB_ids
        q_a_join = q_a_join[:MAX_LENGHT]
        segments_ids = segments_ids[:MAX_LENGHT]
        if type_embedding in ['BERT_portuguese_large_neural_mind', 'BERT_portuguese_base_neural_mind', 'bert_base_multilingual_cased', 'bert_base_multilingual_uncased']:
            input_ids = (q_a_join, segments_ids)
        else:
            input_ids = (q_a_join, 1)
        return encoded(type_embedding, input_ids, model, device)
    elif type_approach == 'query_answer':
        q_a_join = []
        segments_ids = []
        for sentence in prev_inten:
            sentenceA, sentenceB = '[USER]' + sentence[0], '[SYSTEM]' + sentence[1]
            sentenceA_ids = tokenizer(sentenceA, truncation=True)['input_ids'][1:-1]
            sentenceB_ids = tokenizer(sentenceB, truncation=True)['input_ids'][1:-1]
            sentenceB_ids = sentenceB_ids[:30] if len(sentenceB_ids) > 30 else sentenceB_ids
            segments_ids = segments_ids + [0 for i in range(len(sentenceA_ids))] + [1 for i in range(len(sentenceB_ids))]
            q_a_join = q_a_join + sentenceA_ids + sentenceB_ids
        query = '[USER]' + query 
        query = tokenizer(query, truncation=True)['input_ids'][1:-1]
        segment_query = [0 for i in range(len(query))]
        #space to query
        q_a_join = q_a_join[:MAX_LENGHT - len(query)]
        segments_ids = segments_ids[:MAX_LENGHT - len(segment_query)]
        #adding query + query_answer
        q_a_join = q_a_join + query
        segments_ids = segments_ids + segment_query
        if type_embedding in ['BERT_portuguese_large_neural_mind', 'BERT_portuguese_base_neural_mind', 'bert_base_multilingual_cased', 'bert_base_multilingual_uncased']:
            input_ids = (q_a_join, segments_ids)
        else:
            input_ids = (q_a_join, 1)
        return encoded(type_embedding, input_ids, model, device)

    if type_approach == 'only_query' and siamesse == 'yes':
        q_a_join = []
        segments_ids = []
        for sentenceA in prev_inten:
            sentenceA_ids = tokenizer(sentenceA, add_special_tokens=True, truncation=True)['input_ids']   
            segments_ids = segments_ids + [0 for i in range(len(sentenceA_ids))]
            q_a_join = q_a_join + sentenceA_ids
        q_a_join = q_a_join[:MAX_LENGHT]
        segments_ids = segments_ids[:MAX_LENGHT]
        if type_embedding in ['BERT_portuguese_large_neural_mind', 'BERT_portuguese_base_neural_mind', 'bert_base_multilingual_cased', 'bert_base_multilingual_uncased']:
            input_ids = (q_a_join, segments_ids)
        else:
            input_ids = (q_a_join, 1)
        return encoded(type_embedding, input_ids, model, device)
    elif type_approach == 'only_query':
        q_a_join = []
        segments_ids = []
        for sentenceA in prev_inten:
            sentenceA_ids = tokenizer(sentenceA, add_special_tokens=True, truncation=True)['input_ids']   
            segments_ids = segments_ids + [0 for i in range(len(sentenceA_ids))]
            q_a_join = q_a_join + sentenceA_ids
        #compute query
        query = '[USER]' + query 
        query = tokenizer(query, truncation=True)['input_ids'][1:-1]
        segment_query = [0 for i in range(len(query))]
        #space query
        q_a_join = q_a_join[:MAX_LENGHT - len(query)]
        segments_ids = segments_ids[:MAX_LENGHT - len(segment_query)]
        #adding query + query
        q_a_join = q_a_join + query
        segments_ids = segments_ids + segment_query
        if type_embedding in ['BERT_portuguese_large_neural_mind', 'BERT_portuguese_base_neural_mind', 'bert_base_multilingual_cased', 'bert_base_multilingual_uncased']:
            input_ids = (q_a_join, segments_ids)
        else:
            input_ids = (q_a_join, 1)
        return encoded(type_embedding, input_ids, model, device)

    if type_approach == 'last_query_or_answer' and siamesse == 'yes':
        q_a_join = []
        segments_ids = []
        for sentenceA in prev_inten:
            sentenceA_ids = tokenizer(sentenceA, truncation=True)['input_ids'][1:-1]
            if sentenceA[:6] == '[USER]':
                print(sentenceA)
                segments_ids = segments_ids + [0 for i in range(len(sentenceA_ids))]
            else:
                segments_ids = segments_ids + [1 for i in range(len(sentenceA_ids))]
            q_a_join = q_a_join + sentenceA_ids
        q_a_join = q_a_join[:MAX_LENGHT]
        segments_ids = segments_ids[:MAX_LENGHT]
        if type_embedding in ['BERT_portuguese_large_neural_mind', 'BERT_portuguese_base_neural_mind', 'bert_base_multilingual_cased', 'bert_base_multilingual_uncased']:
            input_ids = (q_a_join, segments_ids)
        else:
            input_ids = (q_a_join, 1)
        return encoded(type_embedding, input_ids, model, device)
    elif type_approach == 'last_query_or_answer':
        q_a_join = []
        segments_ids = []
        for sentenceA in prev_inten:
            sentenceA_ids = tokenizer(sentenceA, truncation=True)['input_ids'][1:-1]
            if sentenceA[:6] == '[USER]':
                segments_ids = segments_ids + [0 for i in range(len(sentenceA_ids))]
            else:
                segments_ids = segments_ids + [1 for i in range(len(sentenceA_ids))]
            q_a_join = q_a_join + sentenceA_ids
        query = '[USER]' + query 
        query = tokenizer(query, truncation=True)['input_ids'][1:-1]
        segment_query = [0 for i in range(len(query))]
        #space to query
        q_a_join = q_a_join[:MAX_LENGHT - len(query)]
        segments_ids = segments_ids[:MAX_LENGHT - len(segment_query)]
        #adding query + query_answer
        q_a_join = q_a_join + query
        segments_ids = segments_ids + segment_query
        if type_embedding in ['BERT_portuguese_large_neural_mind', 'BERT_portuguese_base_neural_mind', 'bert_base_multilingual_cased', 'bert_base_multilingual_uncased']:
            input_ids = (q_a_join, segments_ids)
        else:
            input_ids = (q_a_join, 1)
        return encoded(type_embedding, input_ids, model, device)

def get_embedding(type_embedding, csv, K_messagem, X_train_f, type_approach=None, siamesse = None):
    start_time = time.time()
    # Set the device to GPU (cuda) if available, otherwise stick with CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    list_of_mean = []
    list_of_four_last_embeddings = []
    prev_inten = []
    tokenizer, model = tokenizer_and_model(type_embedding)
    if type_approach == 'isolated':
        X_train_f = sum(X_train_f,[])
        for value in X_train_f:
            _,_,_,_,_,_,query,_ = csv.iloc[value]
            #----------------------------------------------------
            #-----------------Isoled query-----------------------
            #----------------------------------------------------
            out = feed_bert(type_embedding, query, tokenizer, model, device, type_approach, prev_inten= None, siamesse = None)
            last_state, mean_state, four_last_states = return_embedding(out)
            list_of_mean.append(mean_state)
            list_of_four_last_embeddings.append(four_last_states)
        print('isolated')
        print(len(list_of_four_last_embeddings))
        end_time = time.time()
        time_dif = end_time - start_time
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
        return list_of_mean, list_of_four_last_embeddings

    if siamesse == 'yes':
        if type_approach == 'query_answer':
            for i in X_train_f:
                _,_,_,_,_,_,query,response = csv.iloc[i[0]]
                out2 = feed_bert(type_embedding, query,tokenizer, model, device, 'isolated', prev_inten = None, siamesse = None)
                last_state, mean_state, four_last_states = return_embedding(out2)
                list_of_mean.append(mean_state)
                list_of_four_last_embeddings.append(four_last_states)
                prev_inten.append([query,response])
                for j in i[1:]:
                    _,_,_,_,_,_,query,respuesta = csv.iloc[j]
                    #----------------------------------------------------
                    #-----------------Query and Answer-------------------
                    #----------------------------------------------------
                    out2 = feed_bert(type_embedding, query,tokenizer, model, device, 'query_answer', prev_inten = prev_inten, siamesse = 'yes')
                    #----------Embedding of Query and response-----------
                    last_state_q_a, mean_state_q_a, four_last_states_q_a = return_embedding(out2) 
                    list_of_mean.append(mean_state_q_a)
                    list_of_four_last_embeddings.append(four_last_states_q_a)

                    prev_inten.append([query,respuesta])
                
                prev_inten = []
            print('q and r')
            print(len(list_of_four_last_embeddings))
            end_time = time.time()
            time_dif = end_time - start_time
            print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
            return list_of_mean, list_of_four_last_embeddings
        
        if type_approach == 'only_query':
            for i in X_train_f:
                _,_,_,_,_,_,query,_ = csv.iloc[i[0]]
                out2 = feed_bert(type_embedding, query,tokenizer, model, device, 'isolated', prev_inten = None, siamesse = None)
                last_state, mean_state, four_last_states = return_embedding(out2)
                list_of_mean.append(mean_state)
                list_of_four_last_embeddings.append(four_last_states)
                prev_inten.append(query)
                for j in i[1:]:
                    _,_,_,_,_,_,query,_ = csv.iloc[j]
                    #----------------------------------------------------
                    #---------------------Only querys--------------------
                    #----------------------------------------------------
                    out2 = feed_bert(type_embedding, query,tokenizer, model, device, 'only_query', prev_inten = prev_inten, siamesse = 'yes')
                    #----------Embedding of Query and response-----------
                    last_state_q_a, mean_state_q_a, four_last_states_q_a = return_embedding(out2) 
                    list_of_mean.append(mean_state_q_a)
                    list_of_four_last_embeddings.append(four_last_states_q_a)

                    prev_inten.append(query)
                
                prev_inten = []
            print('only_query')
            print(len(list_of_four_last_embeddings))
            end_time = time.time()
            time_dif = end_time - start_time
            print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
            return list_of_mean, list_of_four_last_embeddings

        if type_approach == 'last_query_and_answer':
            for i in X_train_f:
                _,_,_,_,_,_,query,answer = csv.iloc[i[0]]
                out2 = feed_bert(type_embedding, query,tokenizer, model, device, 'isolated', prev_inten = None, siamesse = None)
                last_state, mean_state, four_last_states = return_embedding(out2)
                list_of_mean.append(mean_state)
                list_of_four_last_embeddings.append(four_last_states)
                prev_inten.append([query, answer])
                for j in i[1:]:
                    _,_,_,_,_,_,query,answer = csv.iloc[j]
                    #----------------------------------------------------
                    #---------------------Only querys--------------------
                    #----------------------------------------------------
                    out2 = feed_bert(type_embedding, query,tokenizer, model, device, 'query_answer', prev_inten = prev_inten, siamesse = 'yes')
                    #----------Embedding of Query and response-----------
                    last_state_q_a, mean_state_q_a, four_last_states_q_a = return_embedding(out2) 
                    list_of_mean.append(mean_state_q_a)
                    list_of_four_last_embeddings.append(four_last_states_q_a)
                    prev_inten[0] = [query,answer]
                prev_inten = []
                
            print('last_query_answer')
            print(len(list_of_four_last_embeddings))
            end_time = time.time()
            time_dif = end_time - start_time
            print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
            return list_of_mean, list_of_four_last_embeddings

        if type_approach == 'last_query':
            for i in X_train_f:
                _,_,_,_,_,_,query,_ = csv.iloc[i[0]]
                out2 = feed_bert(type_embedding, query,tokenizer, model, device, 'isolated', prev_inten = None, siamesse = None)
                last_state, mean_state, four_last_states = return_embedding(out2)
                list_of_mean.append(mean_state)
                list_of_four_last_embeddings.append(four_last_states)
                prev_inten.append('[USER]' + query)
                for j in i[1:]:
                    _,_,_,_,_,_,query,_ = csv.iloc[j]
                    #----------------------------------------------------
                    #---------------------Only querys--------------------
                    #----------------------------------------------------
                    out2 = feed_bert(type_embedding, query,tokenizer, model, device, 'last_query_or_answer', prev_inten = prev_inten, siamesse = 'yes')
                    #----------Embedding of Query and response-----------
                    last_state_q_a, mean_state_q_a, four_last_states_q_a = return_embedding(out2) 
                    list_of_mean.append(mean_state_q_a)
                    list_of_four_last_embeddings.append(four_last_states_q_a)

                    prev_inten[0] = '[USER]' + query
                
                prev_inten = []
            print('last_query')
            print(len(list_of_four_last_embeddings))
            end_time = time.time()
            time_dif = end_time - start_time
            print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
            return list_of_mean, list_of_four_last_embeddings

        if type_approach == 'last_answer':
            for i in X_train_f:
                _,_,_,_,_,_,query,answer = csv.iloc[i[0]]
                out2 = feed_bert(type_embedding, query,tokenizer, model, device, 'isolated', prev_inten = None, siamesse = None)
                last_state, mean_state, four_last_states = return_embedding(out2)
                list_of_mean.append(mean_state)
                list_of_four_last_embeddings.append(four_last_states)
                prev_inten.append('[SYSTEM]' + answer)
                for j in i[1:]:
                    _,_,_,_,_,_,query,response = csv.iloc[j]
                    #----------------------------------------------------
                    #---------------------Only querys--------------------
                    #----------------------------------------------------
                    out2 = feed_bert(type_embedding, query,tokenizer, model, device, 'last_query_or_answer', prev_inten = prev_inten, siamesse = 'yes')
                    #----------Embedding of Query and response-----------
                    last_state_q_a, mean_state_q_a, four_last_states_q_a = return_embedding(out2) 
                    list_of_mean.append(mean_state_q_a)
                    list_of_four_last_embeddings.append(four_last_states_q_a)

                    prev_inten[0] = '[SYSTEM]' + response
                
                prev_inten = []
            print('last_answer')
            print(len(list_of_four_last_embeddings))
            end_time = time.time()
            time_dif = end_time - start_time
            print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
            return list_of_mean, list_of_four_last_embeddings
    else:
        if type_approach == 'query_answer':
            for i in X_train_f:
                _,_,_,_,_,_,query,response = csv.iloc[i[0]]
                out2 = feed_bert(type_embedding, query,tokenizer, model, device, 'isolated', prev_inten = None, siamesse = None)
                last_state, mean_state, four_last_states = return_embedding(out2)
                list_of_mean.append(mean_state)
                list_of_four_last_embeddings.append(four_last_states)
                prev_inten.append([query,response])
                for j in i[1:]:
                    _,_,_,_,_,_,query,respuesta = csv.iloc[j]
                    #----------------------------------------------------
                    #-----------------Query and Answer-------------------
                    #----------------------------------------------------
                    out2 = feed_bert(type_embedding, query,tokenizer, model, device, 'query_answer', prev_inten = prev_inten, siamesse = None)
                    #----------Embedding of Query and response-----------
                    last_state_q_a, mean_state_q_a, four_last_states_q_a = return_embedding(out2) 
                    list_of_mean.append(mean_state_q_a)
                    list_of_four_last_embeddings.append(four_last_states_q_a)

                    prev_inten.append([query,respuesta])
                
                prev_inten = []
            print('q and r')
            print(len(list_of_four_last_embeddings))
            end_time = time.time()
            time_dif = end_time - start_time
            print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
            return list_of_mean, list_of_four_last_embeddings
        
        if type_approach == 'only_query':
            for i in X_train_f:
                _,_,_,_,_,_,query,_ = csv.iloc[i[0]]
                out2 = feed_bert(type_embedding, query,tokenizer, model, device, 'isolated', prev_inten = None, siamesse = None)
                last_state, mean_state, four_last_states = return_embedding(out2)
                list_of_mean.append(mean_state)
                list_of_four_last_embeddings.append(four_last_states)
                prev_inten.append(query)
                for j in i[1:]:
                    _,_,_,_,_,_,query,_ = csv.iloc[j]
                    #----------------------------------------------------
                    #---------------------Only querys--------------------
                    #----------------------------------------------------
                    out2 = feed_bert(type_embedding, query,tokenizer, model, device, 'only_query', prev_inten = prev_inten, siamesse = None)
                    #----------Embedding of Query and response-----------
                    last_state_q_a, mean_state_q_a, four_last_states_q_a = return_embedding(out2) 
                    list_of_mean.append(mean_state_q_a)
                    list_of_four_last_embeddings.append(four_last_states_q_a)

                    prev_inten.append(query)
                
                prev_inten = []
            print('only_query')
            print(len(list_of_four_last_embeddings))
            end_time = time.time()
            time_dif = end_time - start_time
            print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
            return list_of_mean, list_of_four_last_embeddings

        if type_approach == 'last_query_and_answer':
            for i in X_train_f:
                _,_,_,_,_,_,query,response = csv.iloc[i[0]]
                out2 = feed_bert(type_embedding, query,tokenizer, model, device, 'isolated', prev_inten = None, siamesse = None)
                last_state, mean_state, four_last_states = return_embedding(out2)
                list_of_mean.append(mean_state)
                list_of_four_last_embeddings.append(four_last_states)
                prev_inten.append([query, response])
                for j in i[1:]:
                    _,_,_,_,_,_,query,response = csv.iloc[j]
                    #----------------------------------------------------
                    #---------------------last query---------------------
                    #----------------------------------------------------
                    out2 = feed_bert(type_embedding, query,tokenizer, model, device, 'query_answer', prev_inten = prev_inten, siamesse = None)
                    #----------Embedding of Query and response-----------
                    last_state_q_a, mean_state_q_a, four_last_states_q_a = return_embedding(out2) 
                    list_of_mean.append(mean_state_q_a)
                    list_of_four_last_embeddings.append(four_last_states_q_a)
                
                    prev_inten[0] = [query,response]
                prev_inten = []
            print('only_query')
            print(len(list_of_four_last_embeddings))
            end_time = time.time()
            time_dif = end_time - start_time
            print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
            return list_of_mean, list_of_four_last_embeddings
        
        if type_approach == 'last_query':
            for i in X_train_f:
                _,_,_,_,_,_,query,_ = csv.iloc[i[0]]
                out2 = feed_bert(type_embedding, query,tokenizer, model, device, 'isolated', prev_inten = None, siamesse = None)
                last_state, mean_state, four_last_states = return_embedding(out2)
                list_of_mean.append(mean_state)
                list_of_four_last_embeddings.append(four_last_states)
                prev_inten.append(['[USER]' + query])
                for j in i[1:]:
                    _,_,_,_,_,_,query,_ = csv.iloc[j]
                    #----------------------------------------------------
                    #---------------------last query---------------------
                    #----------------------------------------------------
                    out2 = feed_bert(type_embedding, query,tokenizer, model, device, 'last_query_or_answer', prev_inten = prev_inten, siamesse = None)
                    #----------Embedding of Query and response-----------
                    last_state_q_a, mean_state_q_a, four_last_states_q_a = return_embedding(out2) 
                    list_of_mean.append(mean_state_q_a)
                    list_of_four_last_embeddings.append(four_last_states_q_a)
                    prev_inten[0] = ['[USER]' + query]
                prev_inten = []
            print('only_query')
            print(len(list_of_four_last_embeddings))
            end_time = time.time()
            time_dif = end_time - start_time
            print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
            return list_of_mean, list_of_four_last_embeddings
        

        if type_approach == 'last_answer':
            for i in X_train_f:
                _,_,_,_,_,_,query,response = csv.iloc[i[0]]
                out2 = feed_bert(type_embedding, query,tokenizer, model, device, 'isolated', prev_inten = None, siamesse = None)
                last_state, mean_state, four_last_states = return_embedding(out2)
                list_of_mean.append(mean_state)
                list_of_four_last_embeddings.append(four_last_states)
                prev_inten.append(['[SYSTEM]' + response])
                for j in i[1:]:
                    _,_,_,_,_,_,query,response = csv.iloc[j]
                    #----------------------------------------------------
                    #---------------------last query---------------------
                    #----------------------------------------------------
                    out2 = feed_bert(type_embedding, query,tokenizer, model, device, 'last_query_or_answer', prev_inten = prev_inten, siamesse = None)
                    #----------Embedding of Query and response-----------
                    last_state_q_a, mean_state_q_a, four_last_states_q_a = return_embedding(out2) 
                    list_of_mean.append(mean_state_q_a)
                    list_of_four_last_embeddings.append(four_last_states_q_a)
                
                    prev_inten[0] = ['[SYSTEM]' + response]
                prev_inten = []
            print('only_query')
            print(len(list_of_four_last_embeddings))
            end_time = time.time()
            time_dif = end_time - start_time
            print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
            return list_of_mean, list_of_four_last_embeddings
        
