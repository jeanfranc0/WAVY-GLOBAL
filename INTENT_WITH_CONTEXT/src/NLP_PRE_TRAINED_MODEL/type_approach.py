from src.NLP_PRE_TRAINED_MODEL.get_embedding import get_embedding

def intent_isoladas(type_embedding, X_train_f, csv, K_messagem):
    list_of_mean, list_of_four_last_embeddings = get_embedding(type_embedding, csv, K_messagem, X_train_f, type_approach = 'isolated')
    return list_of_mean, list_of_four_last_embeddings

def intent_query_and_answer(type_embedding, X_train_f, csv, K_messagem, siamesse = None):
    if siamesse == 'yes':
        list_of_mean_q_a, list_of_four_last_embeddings_q_a = get_embedding(type_embedding, csv, K_messagem, X_train_f, type_approach = 'query_answer', siamesse='yes')
    else:
        list_of_mean_q_a, list_of_four_last_embeddings_q_a = get_embedding(type_embedding, csv, K_messagem, X_train_f, type_approach = 'query_answer', siamesse=None)
    return list_of_mean_q_a, list_of_four_last_embeddings_q_a

def intent_only_query(type_embedding, X_train_f, csv, K_messagem, siamesse = None):
    if siamesse == 'yes':
        list_of_mean_only_q, list_of_four_last_embeddings_only_q = get_embedding(type_embedding, csv, K_messagem, X_train_f, type_approach = 'only_query', siamesse='yes')
    else:
        list_of_mean_only_q, list_of_four_last_embeddings_only_q = get_embedding(type_embedding, csv, K_messagem, X_train_f, type_approach = 'only_query', siamesse=None)
    return list_of_mean_only_q, list_of_four_last_embeddings_only_q

def intent_last_query_and_answer(type_embedding, X_train_f, csv, K_messagem, siamesse = None):
    if siamesse == 'yes':
        list_of_mean_only_q, list_of_four_last_embeddings_only_q = get_embedding(type_embedding, csv, K_messagem, X_train_f, type_approach = 'last_query_and_answer', siamesse='yes')
    else:
        list_of_mean_only_q, list_of_four_last_embeddings_only_q = get_embedding(type_embedding, csv, K_messagem, X_train_f, type_approach = 'last_query_and_answer', siamesse=None)
    return list_of_mean_only_q, list_of_four_last_embeddings_only_q
    
def intent_last_query(type_embedding, X_train_f, csv, K_messagem, siamesse = None):
    if siamesse == 'yes':
        list_of_mean_only_q, list_of_four_last_embeddings_only_q = get_embedding(type_embedding, csv, K_messagem, X_train_f, type_approach = 'last_query', siamesse='yes')
    else:
        list_of_mean_only_q, list_of_four_last_embeddings_only_q = get_embedding(type_embedding, csv, K_messagem, X_train_f, type_approach = 'last_query', siamesse=None)
    return list_of_mean_only_q, list_of_four_last_embeddings_only_q

def intent_last_answer(type_embedding, X_train_f, csv, K_messagem, siamesse = None):
    if siamesse == 'yes':
        list_of_mean_only_q, list_of_four_last_embeddings_only_q = get_embedding(type_embedding, csv, K_messagem, X_train_f, type_approach = 'last_answer', siamesse='yes')
    else:
        list_of_mean_only_q, list_of_four_last_embeddings_only_q = get_embedding(type_embedding, csv, K_messagem, X_train_f, type_approach = 'last_answer', siamesse=None)
    return list_of_mean_only_q, list_of_four_last_embeddings_only_q
