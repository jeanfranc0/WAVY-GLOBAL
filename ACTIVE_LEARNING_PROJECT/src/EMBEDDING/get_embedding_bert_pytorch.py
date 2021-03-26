import torch
import numpy as np
from transformers import BertModel, BertTokenizer, XLMRobertaModel, XLMRobertaTokenizer, XLMModel, XLMTokenizer, DistilBertTokenizer, DistilBertModel, ElectraTokenizer, ElectraModel, BartTokenizer, BartModel
#https://huggingface.co/transformers/multilingual.html
#https://github.com/BramVanroy/bert-for-inference/blob/master/introduction-to-bert.ipynb
def get_embedding(type_embedding, data):
    if type_embedding.split('_')[0] == 'BERT' or type_embedding.split('_')[0] == 'bert':
        if type_embedding == 'BERT_portuguese_large_neural_mind':
            path = '/home/jeanfranco/Movile_project/Semi_supervised_learning/data/Brazilian_Bert/BERT_large_portuguese/'
        elif type_embedding == 'BERT_portuguese_base_neural_mind':
            path = '/home/jeanfranco/Movile_project/Semi_supervised_learning/data/Brazilian_Bert/BERT_base_portuguese/'
        elif type_embedding == 'bert_base_multilingual_cased':
            path = 'bert-base-multilingual-cased' 
        elif type_embedding == 'bert_base_multilingual_uncased':
            data =  [x.lower() for x in data]
            path = 'bert-base-multilingual-uncased' 
        #load tokenizer and model
        tokenizer = BertTokenizer.from_pretrained(path)
        model = BertModel.from_pretrained(path, output_hidden_states=True)
    elif type_embedding.split('_')[0] == 'xlmroberta':
        if type_embedding == 'xlmroberta_base':
            path = 'xlm-roberta-base'
        elif type_embedding == 'xlmroberta_large':
            path = 'xlm-roberta-large'
        #load tokenizer and model
        tokenizer = XLMRobertaTokenizer.from_pretrained(path)
        model = XLMRobertaModel.from_pretrained(path, output_hidden_states=True)
    elif type_embedding == 'xlm':
        path = 'xlm-mlm-100-1280'
        #load tokenizer and model
        tokenizer = XLMTokenizer.from_pretrained(path)
        model = XLMModel.from_pretrained(path, output_hidden_states=True)
    #########
    #ENGLISH
    #########
    elif type_embedding == 'en_bert_base_uncased':
        path = 'bert-base-uncased'
        #load tokenizer and model
        tokenizer = BertTokenizer.from_pretrained(path)
        model = BertModel.from_pretrained(path, output_hidden_states=True)
    elif type_embedding == 'en_xlm_mlm_enfr_1024':
        path = 'xlm-mlm-enfr-1024'
        #load tokenizer and model
        tokenizer = XLMTokenizer.from_pretrained(path)
        model = XLMModel.from_pretrained(path, output_hidden_states=True)
    elif type_embedding == 'en_xlm_roberta_base':
        path = 'xlm-roberta-base'
        #load tokenizer and model
        tokenizer = XLMRobertaTokenizer.from_pretrained(path)
        model = XLMRobertaModel.from_pretrained(path, output_hidden_states=True)
    elif type_embedding == 'distilbert_base_cased':
        path = 'distilbert-base-cased'
        #load tokenizer and model
        tokenizer = DistilBertTokenizer.from_pretrained(path)
        model = DistilBertModel.from_pretrained(path, output_hidden_states=True)
    elif type_embedding == 'Mobile_Bert':
        path = 'google/mobilebert-uncased'
        #load tokenizer and model
        tokenizer = MobileBertTokenizer.from_pretrained(path)
        model = MobileBertModel.from_pretrained(path, output_hidden_states=True)
    elif type_embedding == 'Electra':
        path = 'google/electra-small-discriminator'
        #load tokenizer and model
        tokenizer = ElectraTokenizer.from_pretrained(path)
        model = ElectraModel.from_pretrained(path, output_hidden_states=True)
    elif type_embedding == 'BART':
        path = 'facebook/bart-large'
        #load tokenizer and model
        tokenizer = BartTokenizer.from_pretrained(path)
        model = BartModel.from_pretrained(path, output_hidden_states=True)

    # Set the device to GPU (cuda) if available, otherwise stick with CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    list_of_four_last_embeddings = []
    list_of_mean = []

    for l in data:
        # Convert the string "granola bars" to tokenized vocabulary IDs
        input_ids = tokenizer.encode(l)
        #print(input_ids)
        # Convert the list of IDs to a tensor of IDs 
        input_ids = torch.LongTensor(input_ids)
        #print(input_ids)
        model = model.to(device)
        input_ids = input_ids.to(device)
        #print(input_ids)
        model.eval()

        # unsqueeze IDs to get batch size of 1 as added dimension
        input_ids = input_ids.unsqueeze(0)
        with torch.no_grad():
            out = model(input_ids=input_ids)

        # we only want the hidden_states
        if type_embedding == 'xlm':
            hidden_states = out[1]
        else:
            hidden_states = out[2]
        #mean of layers
        sentence_embedding = torch.mean(hidden_states[-1], dim=1).squeeze()
        list_of_mean.append(sentence_embedding.tolist())

        # get last four layers 
        last_four_layers = [hidden_states[i] for i in (-1, -2, -3, -4)]
        # cast layers to a tuple and concatenate over the last dimension
        cat_hidden_states = torch.cat(tuple(last_four_layers), dim=-1)
        
        # take the mean of the concatenated vector over the token dimension
        cat_sentence_embedding = torch.mean(cat_hidden_states, dim=1).squeeze()
        list_of_four_last_embeddings.append(cat_sentence_embedding.tolist())

    #print('list of four last embeddings', np.array(list_of_four_last_embeddings).shape)  
    #print('list of mean', np.array(list_of_mean).shape)
    
    return list_of_mean, list_of_four_last_embeddings

