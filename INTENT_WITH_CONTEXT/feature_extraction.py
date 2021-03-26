import argparse
from src.utils.load_data import run
 
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("conversation_dir", type=str, help="Directory of conversations(.csv)")
	parser.add_argument("type_embedding", type=str, help="types of embeddings", choices=['BERT_portuguese_large_neural_mind',
																						'BERT_portuguese_base_neural_mind',
																						'bert_base_multilingual_cased',
																						'bert_base_multilingual_uncased',
																						'xlmroberta_base',
																						'xlmroberta_large',
																						'xlm'])
	parser.add_argument("perc_train", type=float, help="Percentage of train samples")
	parser.add_argument("path_save", type=str, help="fale path to save the results(.csv)")

	args = parser.parse_args()

	conversation_dir = args.conversation_dir
	type_embedding = args.type_embedding
	perc_train = args.perc_train
	path_save = args.path_save

	run(conversation_dir, type_embedding, perc_train, path_save) 


if __name__ == "__main__":
    main()
