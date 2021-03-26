import argparse
import utils
import time
import numpy as np
import torch
from datetime import timedelta
from PREPROCESSING.utils import load_data, load_data_atis, load_new_dataset
from EMBEDDING.embedding import CarregarWordEmbeddings, sent2vec, Elmo2word, Elmo_embedding, CarregarWordEmbeddings_english
from sklearn.model_selection import train_test_split
from EMBEDDING.get_embedding_bert_pytorch import get_embedding

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("conversation_dir", type=str, help="Directory of conversations .csv")
	parser.add_argument("dataset_type", type=str, help="type of dataset", choices=['balanced', 'atis', 'new_dataset'])
	parser.add_argument("embedding_type", type=str, help="types of embeddings", choices=['Word2Vec', 
																						'Wang2Vec',
																						'FastText',
																						'Glove',
																						'BERT_portuguese_large_neural_mind',
																						'BERT_portuguese_base_neural_mind',
																						'bert_base_multilingual_cased',
																						'bert_base_multilingual_uncased',
																						'xlmroberta_base',
																						'xlmroberta_large',
																						'xlm',
																						'en_bert_base_uncased',
																						'en_xlm_mlm_enfr_1024',
																						'en_xlm_roberta_base',
																						'Mobile_Bert',
																						'Electra',
																						'BART'])
	parser.add_argument("perc_train", type=float, help="Percentage of train samples")
	parser.add_argument("X_train_filename", type=str, help="Dataset train file name (*.npy)")
	parser.add_argument("Y_train_filename", type=str, help="Label train filename (*.npy)")
	parser.add_argument("X_test_filename", type=str, help="Dataset test file name (*.npy)")
	parser.add_argument("Y_test_filename", type=str, help="Label test filename (*.npy)")
	args = parser.parse_args()

	conversation_dir = args.conversation_dir
	dataset_type = args.dataset_type
	embedding_type = args.embedding_type
	perc_train = args.perc_train
	X_train_filename = args.X_train_filename
	Y_train_filename = args.Y_train_filename
	X_test_filename = args.X_test_filename
	Y_test_filename = args.Y_test_filename
	start_time = time.time()
	
	set_seed(0)

	if dataset_type == 'balanced':
		query, intent = load_data(conversation_dir)
		X_train_f, X_test_f, Y_train, Y_test = train_test_split(query, 
                                                    intent, random_state=42, train_size=perc_train
													,stratify=intent)

		if embedding_type == 'Word2Vec' or embedding_type == 'FastText' or embedding_type == 'Glove' or embedding_type == 'Wang2Vec':
			embeddings_index = CarregarWordEmbeddings(embedding_type)
			X_train = [sent2vec(embeddings_index, w) for w in X_train_f]
			X_test = [sent2vec(embeddings_index, w) for w in X_test_f]
			np.save(X_train_filename , X_train)
			np.save(Y_train_filename , Y_train)
			np.save(X_test_filename , X_test)
			np.save(Y_test_filename , Y_test)
		elif embedding_type == 'BERT_portuguese_large_neural_mind' or embedding_type == 'BERT_portuguese_base_neural_mind' or embedding_type == 'bert_base_multilingual_cased' or embedding_type == 'bert_base_multilingual_uncased' or embedding_type == 'xlmroberta_base' or embedding_type == 'xlmroberta_large' or embedding_type == 'xlm':
			list_of_mean_X_train, list_of_four_last_embeddings_X_train = get_embedding(embedding_type, X_train_f)
			list_of_mean_X_test, list_of_four_last_embeddings_X_test = get_embedding(embedding_type, X_test_f)
			np.save(X_train_filename + 'mean_X_train.npy' , list_of_mean_X_train)
			np.save(X_test_filename + 'four_last_X_train.npy', list_of_four_last_embeddings_X_train)
			np.save(Y_train_filename , Y_train)
			np.save(X_test_filename + 'mean_X_test.npy', list_of_mean_X_test)
			np.save(X_test_filename + 'four_last_X_test.npy', list_of_four_last_embeddings_X_test)
			np.save(Y_test_filename , Y_test)

	elif dataset_type == 'atis':
		X_train, Y_train, X_test, Y_test, X_dev, Y_dev = load_data_atis(conversation_dir)
		if embedding_type == 'en_bert_base_uncased' or embedding_type == 'en_xlm_mlm_enfr_1024' or embedding_type == 'en_xlm_roberta_base':
			list_of_mean_X_train, list_of_four_last_embeddings_X_train = get_embedding(embedding_type, X_train)
			list_of_mean_X_test, list_of_four_last_embeddings_X_test = get_embedding(embedding_type, X_test)
			list_of_mean_X_dev, list_of_four_last_embeddings_X_dev = get_embedding(embedding_type, X_dev)
			np.save(X_train_filename + 'mean_X_train.npy' , list_of_mean_X_train)
			np.save(X_train_filename + 'four_last_X_train.npy', list_of_four_last_embeddings_X_train)
			np.save(Y_train_filename , Y_train)
			np.save(X_test_filename + 'mean_X_test.npy', list_of_mean_X_test)
			np.save(X_test_filename + 'four_last_X_test.npy', list_of_four_last_embeddings_X_test)
			np.save(Y_test_filename , Y_test)
			np.save(X_test_filename + 'mean_X_dev.npy', list_of_mean_X_dev)
			np.save(X_test_filename + 'four_last_X_dev.npy', list_of_four_last_embeddings_X_dev)
			np.save(X_test_filename + 'Y_dev.npy' , Y_dev)
		elif embedding_type == 'Word2Vec' or embedding_type == 'FastText' or embedding_type == 'Glove' or embedding_type == 'Wang2Vec':
			embeddings_index = CarregarWordEmbeddings_english(embedding_type)
			X_train = [sent2vec(embeddings_index, w) for w in X_train]
			X_test = [sent2vec(embeddings_index, w) for w in X_test]
			X_dev = [sent2vec(embeddings_index, w) for w in X_dev]			
			np.save(X_train_filename + 'X_train.npy' , X_train)
			np.save(Y_train_filename , Y_train)
			np.save(X_test_filename + 'X_test.npy', X_test)
			np.save(Y_test_filename , Y_test)
			np.save(Y_train_filename + 'X_dev.npy', X_dev)
			np.save(Y_train_filename + 'Y_dev.npy', Y_dev)

	# elif dataset_type == 'new_dataset':
	# 	X_train, Y_train, X_test, Y_test, X_dev, Y_dev = load_new_dataset(conversation_dir)
	# 	if embedding_type == 'en_bert_base_uncased' or embedding_type == 'xlmroberta_base' or embedding_type == 'xlmroberta_large' or embedding_type == 'en_xlm_mlm_enfr_1024' or embedding_type == 'en_xlm_roberta_base' or embedding_type == 'bert_base_multilingual_cased' or embedding_type == 'bert_base_multilingual_uncased' or embedding_type == 'Mobile_Bert' or embedding_type == 'Electra' or embedding_type == 'BART':
	# 		list_of_mean_X_train, list_of_four_last_embeddings_X_train = get_embedding(embedding_type, X_train)
	# 		list_of_mean_X_test, list_of_four_last_embeddings_X_test = get_embedding(embedding_type, X_test)
	# 		list_of_mean_X_dev, list_of_four_last_embeddings_X_dev = get_embedding(embedding_type, X_dev)
	# 		np.save(X_train_filename + 'mean_X_train.npy' , list_of_mean_X_train)
	# 		np.save(X_train_filename + 'four_last_X_train.npy', list_of_four_last_embeddings_X_train)
	# 		np.save(Y_train_filename , Y_train)
	# 		np.save(X_test_filename + 'mean_X_test.npy', list_of_mean_X_test)
	# 		np.save(X_test_filename + 'four_last_X_test.npy', list_of_four_last_embeddings_X_test)
	# 		np.save(Y_test_filename , Y_test)
	# 		np.save(X_test_filename + 'mean_X_dev.npy', list_of_mean_X_dev)
	# 		np.save(X_test_filename + 'four_last_X_dev.npy', list_of_four_last_embeddings_X_dev)
	# 		np.save(X_test_filename + 'Y_dev.npy' , Y_dev)

	print("Embeddings saved")
	end_time = time.time()
	time_dif = end_time - start_time
	print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

if __name__ == "__main__":
    	main()
