import sys
import spacy
import nltk
import re
import gensim
import csv
from importlib import reload
import nltk
nltk.download('words')
import collections
import pandas as pd
import numpy as np
# Modules for text preprocessing
from stop_words import get_stop_words
from gensim import corpora
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger') # For POS tagging
nltk.download('wordnet')
import nltk
nltk.download('punkt')
import argparse
from collections import OrderedDict

class clean_csv(object):
    def __init__(self, csv):
        #del csv['time']
        #del csv['channel']
        #del csv['user']
        #del csv['session']
        self.csv = csv
        print(self.csv)

        #Fantástico! Seu pedido já está na transportadora. 
        #Você receberá até o dia ##/##/####,
        #Escreva *o número* da opção desejada,
        #1 - Acompanhar a entrega do seu último Pedido
        #2 - Consultar a data da próxima entrega
        #3 - Consultar débito ou segunda via de boleto
        #4 - Consultar pontuação do Meu Mundo [[COMPANY_NAME]]
        #5 - Novo Modelo de Negócio [[COMPANY_NAME]]
        #6 - Outros assuntos
        #6 - Consultar uma Revendedora da equipe
        #7 - Alterar o registro de consulta
        self.csv.loc[(self.csv['intent'] == "[CT]last_order") & (self.csv['query'].str.isdigit() == True), 'query'] = "Acompanhar a entrega do seu último Pedido"#1
        self.csv.loc[(self.csv['intent'] == "[CT]next_delivery") & (self.csv['query'].str.isdigit() == True), 'query'] = "Consultar a data da próxima entrega"#2
        self.csv.loc[(self.csv['intent'] == "[CT]bank_slip") & (self.csv['query'].str.isdigit() == True), 'query'] = "Consultar débito ou segunda via de boleto"#3
        self.csv.loc[(self.csv['intent'] == "[CT]loyalty_program") & (self.csv['query'].str.isdigit() == True), 'query'] = "Consultar pontuação do Meu Mundo [[COMPANY_NAME]]"#4
        self.csv.loc[(self.csv['intent'] == "action_menu_modelo_comercial") & (self.csv['query'].str.isdigit() == True), 'query'] = "Novo Modelo de Negócio [[COMPANY_NAME]]"#5
        self.csv.loc[(self.csv['intent'] == "[IT]other_subjects") & (self.csv['query'].str.isdigit() == True), 'query'] = "Outros assuntos"#6
        self.csv.loc[(self.csv['intent'] == "main_menu>missing_registry") & (self.csv['query'].str.isdigit() == True), 'query'] = "Consultar uma Revendedora da equipe"#6
        self.csv.loc[(self.csv['intent'] == "change_registry") & (self.csv['query'].str.isdigit() == True), 'query'] = "Alterar o registro de consulta"#7

        #O Novo Modelo de Negócio da [[COMPANY_NAME]] é exclusivo para alguns Setores. Se você foi selecionada para fazer parte e quer conhecer mais, escolha uma das seguintes opções:,
        #1 - Saiba o que é o Novo Modelo de Negócio
        #2 - Conheça as mudanças
        #3 - Informe-se sobre o seu nível
        #4 - Mais informações
        self.csv.loc[(self.csv['intent'] == "oque_e_novo_modelo_comercial") & (self.csv['query'].str.isdigit() == True), 'query'] = "Saiba o que é o Novo Modelo de Negócio"#1
        self.csv.loc[(self.csv['intent'] == "oque_muda_novo_modelo_comercial") & (self.csv['query'].str.isdigit() == True), 'query'] = "Conheça as mudanças"#2
        self.csv.loc[(self.csv['intent'] == "como_fico_sabendo_nivel") & (self.csv['query'].str.isdigit() == True), 'query'] = "Informe-se sobre o seu nível"#3
        self.csv.loc[(self.csv['intent'] == "como_saber_mais_novo_modelo_comercial") & (self.csv['query'].str.isdigit() == True), 'query'] = "Mais informações"#4

        #Olha, a gente pensou em tudo! A transformação começa, inclusive, pela forma como você passa a ser chamada na [[COMPANY_NAME]]: Empreendedora! Sentiu o poder?🥰,Já pensou em ter um plano de crescimento que te permita lucrar ainda mais com a venda dos nossos produtos?  Então, prepare-se! Olha só os novos níveis que você poderá alcançar:,👉🏻Empreendedora Bronze
        #👉🏻Empreendedora Prata
        #👉🏻Empreendedora Ouro
        #👉🏻Estrela VIP
        #👉🏻Estrela Safira
        #👉🏻Estrela Diamante
        #Quanto mais alto você chegar, mais vai lucrar!  🤑,Escolha um dos números para ver a opção:
        #1 - Voltar
        #2 - Menu principal
        self.csv.loc[(self.csv['intent'] == "action_menu_modelo_comercial") & (self.csv['query'].str.isdigit() == True), 'query'] = "Voltar"#1
        self.csv.loc[(self.csv['intent'] == "main_menu") & (self.csv['query'].str.isdigit() == True), 'query'] = "Menu principal"#2

        #Devolveu algum produto e tem crédito com a [[COMPANY_NAME]]?  Você pode fazer o seguinte:
        #- acesse o site www.[[COMPANY_NAME]].com.br;
        #- clique em *Meus Pedidos*, depois em *Pagar Boleto*;
        #- no site, estará disponível a opção *Utilizar Créditos*.,Essa resposta resolveu a sua dúvida?,
        #1 - *Sim*
        #2 - *Não*
        self.csv.loc[(self.csv['intent'] == "[IT]survey>positive_rating") & (self.csv['query'].str.isdigit() == True), 'query'] = "Sim"#1
        self.csv.loc[(self.csv['intent'] == "[IT]survey>negative_rating") & (self.csv['query'].str.isdigit() == True), 'query'] = "Nao"#2

        #Meus Pedidos e depois “Trocas e Faltas”.  ;) 
        #1 - Quando realizar a troca
        #2 - Troca de tamanho
        #3 - Devolver caixa toda
        self.csv.loc[(self.csv['intent'] == "[IT]return_policy") & (self.csv['query'].str.isdigit() == True), 'query'] = "Quando realizar a troca"#1
        self.csv.loc[(self.csv['intent'] == "[IT]size_change") & (self.csv['query'].str.isdigit() == True), 'query'] = "Troca de tamanho"#2
        self.csv.loc[(self.csv['intent'] == "[IT]return_entire_box") & (self.csv['query'].str.isdigit() == True), 'query'] = "Devolver caixa toda"#3

        #Ao receber o prêmio, se você constatar qualquer defeito, dano ocasionado no transporte ou que não se trata do produto resgatado, entre em contato com o Serviço de Atendimento à Revendedora no #### ### #### de segunda a sábado das ## às ## horas, e abra uma reclamação para análise e possível substituição por outro produto similar em perfeitas condições.
        #Considere os seguintes prazos para abertura de reclamação, após o recebimento do prêmio:
        #- Defeito: 7 dias corridos;
        #- Recebimento de um prêmio danificado: Caso o produto recebido tenha sido danificado no transporte, recuse o recebimento e entre em contato com o  Serviço de Atendimento à Revendedora;
        #- Recebimento de item diferente do resgatado: ## dias corridos;
        #- Troca de voltagem:  ## dias corridos.,Essa resposta resolveu a sua dúvida?,
        #1 - *Sim*
        #2 - *Não*
        self.csv.loc[(self.csv['intent'] == "[IT]survey>positive_rating") & (self.csv['query'].str.isdigit() == True), 'query'] = "Sim"#1
        self.csv.loc[(self.csv['intent'] == "[IT]survey>negative_rating") & (self.csv['query'].str.isdigit() == True), 'query'] = "Nao"#2

        #_Meu Mundo [[COMPANY_NAME]]_ é o programa de premiação para revendedores [[COMPANY_NAME]]! :)
        #Envie o número para saber mais:
        #1 - Consulta de pontos
        #2 - Resgate de prêmios
        #3 - Funcionamento do programa
        self.csv.loc[(self.csv['intent'] == "[CT]loyalty_program_menu>points") & (self.csv['query'].str.isdigit() == True), 'query'] = "Consulta de pontos"#1
        self.csv.loc[(self.csv['intent'] == "[IT]loyalty_program_menu>claim_prizes") & (self.csv['query'].str.isdigit() == True), 'query'] = "Resgate de prêmios"#2
        self.csv.loc[(self.csv['intent'] == "[IT]loyalty_program_menu>how_it_works") & (self.csv['query'].str.isdigit() == True), 'query'] = "Funcionamento do programa"#3

        #Que pena. 😞
        #Tente perguntar de novo, mas com outras palavras!
        #Se preferir, selecione um dos assuntos dessa lista.,
        #1 - Acompanhar Pedido
        #2 - Segunda via de boleto
        #3 - Débito não baixou
        #4 - Folheto Digital
        self.csv.loc[(self.csv['intent'] == "main_menu") & (self.csv['query'].str.isdigit() == True), 'query'] = "Acompanhar Pedido"#3
        self.csv.loc[(self.csv['intent'] == "[CT]bank_slip") & (self.csv['query'].str.isdigit() == True), 'query'] = "Segunda via de boleto"#3
        self.csv.loc[(self.csv['intent'] == "[IT]survey") & (self.csv['query'].str.isdigit() == True), 'query'] = "Débito não baixou"#3
        self.csv.loc[(self.csv['intent'] == "[CT]ebrochure") & (self.csv['query'].str.isdigit() == True), 'query'] = "Folheto Digital"#4

        #Qual das opções abaixo você quer receber?,
        #1 - Imagens dos Lançamentos
        #2 - Ofertas da Campanha
        #3 - Imagens da linha Moda&Casa
        self.csv.loc[(self.csv['intent'] == "[IT]campaign_memes>menu>releases") & (self.csv['query'].str.isdigit() == True), 'query'] = "Imagens dos Lançamentos"#1
        self.csv.loc[(self.csv['intent'] == "[IT]campaign_memes>menu>sales") & (self.csv['query'].str.isdigit() == True), 'query'] = "Ofertas da Campanha"#2
        self.csv.loc[(self.csv['intent'] == "[IT]campaign_memes>menu>fashion_and_home") & (self.csv['query'].str.isdigit() == True), 'query'] = "Imagens da linha Moda&Casa"#3

        #O processo de entrega do seu Pedido é muito simples e seguro! Cada Setor possui uma data programada para entrega dentro do horário comercial, das ##h## às ##h##, de segunda à sábado, por meio de uma transportadora contratada pela [[COMPANY_NAME]].
        #No ato da entrega será exigido um documento de identificação com foto e o entregador pode solicitar que você apresente o comprovante de pagamento do Pedido anterior para a liberação da caixa, por isso, tenha sempre em mãos o comprovante no momento da entrega.
        #Caso a entrega não seja realizada devido falta de apresentação do comprovante de pagamento do Pedido anterior (quando solicitado pelo entregador),  ausência de um responsável pelo recebimento no ato da entrega ou endereço não localizado, a caixa retornará para o depósito e ficará retida por 7 dias úteis aguardando que você faça a retirada.
        #Escolha um dos números para ver a opção:,
        #1 - Acompanhar pedido
        #2 - Quem pode receber a caixa
        #3 - Caixa voltou para [[COMPANY_NAME]]
        #4 - Receber comprovante na entrega
        self.csv.loc[(self.csv['intent'] == "main_menu") & (self.csv['query'].str.isdigit() == True), 'query'] = "Acompanhar pedido"#1
        self.csv.loc[(self.csv['intent'] == "[IT]who_can_receive_delivery") & (self.csv['query'].str.isdigit() == True), 'query'] = "Quem pode receber a caixa"#2
        self.csv.loc[(self.csv['intent'] == "[IT]my_box_returned") & (self.csv['query'].str.isdigit() == True), 'query'] = "Caixa voltou para [[COMPANY_NAME]]"#3
        self.csv.loc[(self.csv['intent'] == "action_comprovante_entrega_avon") & (self.csv['query'].str.isdigit() == True), 'query'] = "Receber comprovante na entrega"#4

        #Se você deixou de revender nossos produtos há mais de 1 ano e quer voltar a Revender [[COMPANY_NAME]] novamente, é necessário entrar no site [[COMPANY_NAME]] e clicar na opção ""Revender"" no menu e preencher o cadastro. Uma Gerente de Setor entrará em contato com você para concluir seu novo cadastro.
        #Ir para o site: http://www.[[COMPANY_NAME]].com.br 
        #Agora, se você está sem revender nossos produtos há menos de 6 meses, entre em contato com o Serviço de Atendimento à Revendedora no #### ### #### de segunda a sábado das ## às ## horas, e siga as orientações para ser ativada novamente.
        #,Essa resposta resolveu a sua dúvida?,
        #1 - *Sim*
        #2 - *Não*
        self.csv.loc[(self.csv['intent'] == "[IT]survey>positive_rating") & (self.csv['query'].str.isdigit() == True), 'query'] = "sim"#1
        self.csv.loc[(self.csv['intent'] == "[IT]survey>negative_rating") & (self.csv['query'].str.isdigit() == True), 'query'] = "Nao"#2

        #A sua caixa pode voltar para a [[COMPANY_NAME]] nas seguintes situações:
        #- Ausência de um responsável pelo recebimento;
        #- Falta de apresentação do comprovante de pagamento do Pedido anterior, quando solicitado pelo entregador;
        #- Endereço não localizado.
        #Para qualquer uma dessas situações, a caixa retornará para o depósito e ficará retida por 7 dias úteis aguardando que você faça a retirada. Para saber o endereço do depósito da [[COMPANY_NAME]], entre em contato com o Serviço de Atendimento à Revendedora- #### ### #### – de segunda a sábado, das ## às ## horas.
        #Se a sua caixa voltar para o depósito e você não for retirar em até 7 dias úteis, ela será desmanchada e não e você não conseguirá mais  resgatá-la 😔
        #Caso deseje, você pode acompanhar a entrega do seu último pedido por aqui, basta escrever o número do item que quer informação:,
        #1 - Acompanhar pedido
        #2 - Outros assuntos
        self.csv.loc[(self.csv['intent'] == "main_menu") & (self.csv['query'].str.isdigit() == True) & (self.csv['response'] == 'Digite o *número do seu registro cadastral* na [[COMPANY_NAME]] para conhecer as opções de consulta.'), 'query'] = "Acompanhar pedido"#1
        self.csv.loc[(self.csv['intent'] == "[IT]other_subjects") & (self.csv['query'].str.isdigit() == True) & (self.csv['response'] == 'Ok, outros assuntos! Sobre o que você quer falar? Pode perguntar!'), 'query'] = "Outros assuntos"#2

        #Qualquer pessoa que esteja no seu endereço de entrega e seja maior de ## anos. O responsável pelo recebimento do Pedido deve assinar o documento de entrega com letra legível, informando a data e o grau de parentesco ou relacionamento com você.
        #Caso não tenha ninguém no seu local de entrega e a caixa esteja liberada, o transportador pode entregar no vizinho da frente, da esquerda ou da direita, desde que a pessoa apresente um documento de identidade.
        #O transportador, ao deixar sua caixa para um determinado vizinho solicitará a assinatura do mesmo (letra legível) e anotará no canhoto de entrega( Reconhecimento de Dívida) para quem foi entregue. Como parte do procedimento, deixará um bilhete no local com todas as informações.
        #Caso deseje, você pode acompanhar a entrega do seu último pedido por aqui, basta clicar no botão abaixo:
        #,1 - Acompanhar pedido
        #2 - Outros assuntos
        self.csv.loc[(self.csv['intent'] == "[CT]last_order") & (self.csv['query'].str.isdigit() == True), 'query'] = "Acompanhar pedido"#2
        self.csv.loc[(self.csv['intent'] == "[IT]other_subjects") & (self.csv['query'].str.isdigit() == True), 'query'] = "Outros assuntos"#2

        #Já faz 2 dias úteis que você pagou seu boleto e o débito ainda não baixou?
        #Mande o comprovante de pagamento para o e-mail [[E-MAIL]] A resposta virá em até 4 dias úteis. 
        #Se tiver mais dúvidas, ligue para #### ### #### de um telefone fixo. Este número funciona de segunda a sábado, das 8h às ##h.
        #,Essa resposta resolveu a sua dúvida?,
        #1 - *Sim*
        #2 - *Não*
        self.csv.loc[(self.csv['intent'] == "[IT]survey>positive_rating") & (self.csv['query'].str.isdigit() == True), 'query'] = "Sim"#2
        self.csv.loc[(self.csv['intent'] == "[IT]survey>negative_rating") & (self.csv['query'].str.isdigit() == True), 'query'] = "Não"#2

        #Você quer receber as imagens de qual campanha?,
        #1 - Campanha atual
        #2 - Próxima campanha
        self.csv.loc[(self.csv['intent'] == "[IT]campaign_memes>menu") & (self.csv['query'].str.isdigit() == True), 'query'] = "Próxima campanha"#1
        self.csv.loc[(self.csv['intent'] == "[IT]campaign_memes>menu") & (self.csv['query'].str.isdigit() == True), 'query'] = "Próxima campanha"#2

        #Você pode lucrar muito revendendo [[COMPANY_NAME]]! Deixa eu te explicar como funciona a lucratividade:
        #- Para Folhetos Cosméticos [[COMPANY_NAME]], a lucratividade sugerida é de ##%;
        #- Para Folhetos Moda & Casa, a lucratividade sugerida é de ##%;
        #- E ainda ofertas incríveis, dedicadas exclusivamente para as nossas Revendedoras na Revista [[COMPANY_NAME]] & Você, aonde você determina o quanto quer lucrar!
        #Escolha um dos números para ver a opção:,
        #1 - Pedido mínimo
        #2 - Prazo de envio
        #3 - Limite de crédito
        #4 - Rejeição de pedido
        self.csv.loc[(self.csv['intent'] == "[IT]profitability>minimum_order") & (self.csv['query'].str.isdigit() == True), 'query'] = "Pedido mínimo"#1
        self.csv.loc[(self.csv['intent'] == "[CT]profitability>site_closure") & (self.csv['query'].str.isdigit() == True), 'query'] = "Prazo de envio"#2
        self.csv.loc[(self.csv['intent'] == "[CT]profitability>credit_limit") & (self.csv['query'].str.isdigit() == True), 'query'] = "Limite de crédito"#3
        self.csv.loc[(self.csv['intent'] == "[IT]profitability>rejected_order") & (self.csv['query'].str.isdigit() == True), 'query'] = "Rejeição de pedido"#4
        print(self.csv)       

    def clean(self, thresold, bool_undefined, bool_confi, number, confi = None):
        if bool_undefined == 'yes':
          a = self.delete_undefined() 
          if len(a) > 0:
            self.csv = self.delete(a).reset_index(drop=True)
        
        if bool_confi =='yes':
          a = self.delete_confidence(confi)
          if len(a) > 0:
            self.csv = self.delete(a).reset_index(drop=True)
        
        a = self.delete_digits_in_query()
        if len(a) > 0:
          self.csv = self.delete(a).reset_index(drop=True)
        
        self.csv = self.single_conversation(number).reset_index(drop=True)

        a = self.delete_nan('response')
        if len(a) > 0:
          self.csv = self.delete(a).reset_index(drop=True)

        a = self.delete_nan('confidence')
        if len(a) > 0:
          self.csv = self.delete(a).reset_index(drop=True)

        counter, labels, clases = self.basic()
        inte = self.delete_thresold(thresold, counter, clases)
        a = self.clear_index(inte)
        if len(a) > 0:
          self.csv = self.delete(a).reset_index(drop=True)

        self.csv = self.change_intent_to_digit()
        
        return self.csv.reset_index(drop=True)
        
    def delete_undefined(self):
        intent = self.csv['intent'].tolist()
        list_unde = [index for index,item in enumerate(intent) if item == 'not_understood']
        a = []
        for i in list_unde:
          _,_,_,_,_,_,user,session = self.csv.iloc[i]
          list_index = np.asarray(np.where((self.csv.user == user) & (self.csv.session == session)))[0]
          a.append(list_index)
        return a   

    def delete_confidence(self, confi):
        confidence = self.csv['confidence'].tolist()
        list_confi = [index for index,item in enumerate(confidence) if item < confi]
        a = []
        for i in list_confi:
          _,_,_,_,_,_,user,session = self.csv.iloc[i]
          list_index = np.asarray(np.where((self.csv.user == user) & (self.csv.session == session)))[0]
          a.append(list_index)   
        return a 
    
    def delete_digits_in_query(self):
        messages = self.csv['query'].tolist()
        numbers = [index for index,item in enumerate(messages) if item.isdigit() == True]
        a = []
        return self.clear_numbers(numbers, a)

    def delete_thresold(self, thresold, counter, clases):
        inte = []
        for i in counter:
          if int(counter[i]) < thresold:
            inte.append(i)
        return [clases[i] for i in inte]
    
    def basic(self):
        labels = self.csv['intent'].tolist()
        clases = list(set(labels))
        labels = [clases.index(intent) for intent in labels]
        print('basic')
        counter = collections.Counter(labels)
        print(counter, labels, clases)
        return counter, labels, clases

    def clear_numbers(self, numbers, a):
        for i in numbers:
            _,_,_,_,_,_,user,session = self.csv.iloc[i]
            list_index = np.asarray(np.where((self.csv.user == user) & (self.csv.session == session)))[0]
            a.append(list_index)
        return a
        
    def clear_index(self, inte):
        a = []
        for index, row in self.csv.iterrows():
            _,intnt,_,_,_,_,user,session = self.csv.iloc[index]
            if intnt in inte:
                list_index = np.asarray(np.where((self.csv.user == user) & (self.csv.session == session)))[0]
                a.append(list_index)
        return a

    def delete_nan(self, atributo):
        a = self.csv[self.csv[atributo].isnull()]
        b = []
        for index, row in a.iterrows():
          _,intnt,_,_,_,_,user,session = self.csv.iloc[index]
          list_index = np.asarray(np.where((self.csv.user == user) & (self.csv.session == session)))[0]
          b.append(list_index)
        return b

    def change_intent_to_digit(self):
        labels = self.csv['intent'].tolist()
        clases = list(set(labels))
        self.csv = self.csv[self.csv.intent.isin(self.csv.intent.value_counts().tail(len(clases)).index)].copy() # select numer of clases low dimensional categories
        map_label = dict(enumerate(self.csv.intent.factorize()[1]))
        self.csv['intent'] = self.csv.intent.factorize()[0]
        return self.csv

    def delete(self, a):
        b = [j for i in a for j in i]
        c = list(set(b))
        print('cantidad de mensajes a eliminar', len(c))
        for l in c:
            self.csv.drop(l, inplace=True)
        print(self.csv)
        counter, labels, clases = self.basic()
        self.csv.reset_index(drop=True)
        return self.csv  

    def single_conversation(self, number):
        b = []
        for index, row in self.csv.iterrows():
            _,_,_,_,_,_,user,session = self.csv.iloc[index]
            list_index = np.asarray(np.where((self.csv.user == user) & (self.csv.session == session)))[0]
            b.append(list_index)
            
        b = list(map(list, OrderedDict.fromkeys(map(tuple, b)).keys()))
        to_delete = [w for w in b if len(w) < number]
        
        self.csv = self.delete(to_delete)
        self.csv.reset_index(drop=True)

        return self.csv

class Preprocessing(object):
    ''' Perform preprocessing steps on dataset. '''

    def __init__(self):
        self.stop_words = nltk.corpus.stopwords.words('english')
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stops_english = get_stop_words('english')
        self.stops_portuguese = get_stop_words('portuguese')
        self.stops_spanish = get_stop_words('spanish')
        self.is_english = set(nltk.corpus.words.words())

    def remove_null(self, dataframe):
        ''' Helper function to remove NA values. '''
        dataframe = dataframe.dropna(axis = 0, how = 'any')
        return dataframe

    def preprocess_text(self, text_area):
        ''' Function to preprocess text data. Steps taken are:
            1. Tokenization
            2. Removal of any non alphanumeric character
            3. Lowercase words
            4. Removal of stop words
            5. Bigram / Trigram collocation detection (/frequently co-occurring tokens) using Gensim's Phrases
            6. Lemmatization (not stemming to avoid reduction of interpretability
            Parameters: list of strings
            ----------
            Returns: Preprocessed list of strings.
            -------
        '''
        #text_area = [line.encode('utf-8') for line in text_area] # for encoding problems
        text_area = [word_tokenize(line) for line in text_area] # tokenization
        #text_area = [[word for word in line if len(word) > 1] for line in text_area] # remove single character strings
        text_area = [[word for word in line if word.isalnum()] if len([word for word in line if word.isalnum()]) > 0 else line for line in text_area] # remove punctuation
        text_area = [[word.lower() for word in line] if len([word.lower() for word in line]) else line for line in text_area] #lowercase
        text_area = [[word for word in line if not word in self.stops_portuguese] if len([word for word in line if not word in self.stops_portuguese]) > 0 else line for line in text_area]# remove Portuguese stopwords        
        #text_area = [line for line in text_area if line != []]
        text_area = self.make_n_grams(text_area, n_gram = 'bigram') # bigram
        text_area = self.make_n_grams(text_area, n_gram = 'trigram') # trigram
        text_area = self.pruning(text_area, 'lemmatization') # prune words using lemmatization
        text_area = [" ".join(str(x) for x in i) for i in text_area]
        print(len(text_area))
        return text_area

    def pruning(self, text_area, pruning_method = ''):
        ''' Prune words to reduce them to their most basic form. '''
        if pruning_method == 'stemming':
            text_area = [[self.stemmer.stem(word) for word in line] for line in text_area]
        elif pruning_method == 'lemmatization':
            text_area = [[self.lemmatizer.lemmatize(word) for word in line] for line in text_area]
        else:
            raise ValueError('Please set pruning_method to "stemming" or "lemmatization".')
        return text_area

    def make_n_grams(self, text_area, n_gram = ''):
        # build bigram/trigram models
        bigram = gensim.models.Phrases(text_area, min_count = 5, threshold = 100)
        trigram = gensim.models.Phrases(bigram[text_area], threshold = 100)
        # faster way to get a sentence clubbed as a bigram/trigram
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)
        if n_gram == 'bigram':
            return [bigram_mod[doc] for doc in text_area]
        elif n_gram == 'trigram':
            return [trigram_mod[bigram_mod[doc]] for doc in text_area]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Dataset path (*.csv)")
    parser.add_argument("destino_path", type=str, help="Path to save the file (*.csv)")
    args = parser.parse_args()
    
    path = args.path
    destino_path = args.destino_path
    #read csv  
    csv = pd.read_csv(path)
    #csv = clean_csv(csv).clean(600,'not', 'not', confi = None)
    #csv = clean_csv(csv).clean(600,'yes', 'not', confi = None)
    csv = clean_csv(csv).clean(700,'yes', 'yes',3, confi = 0.65)
    csv = csv.reset_index(drop=True)
    
    #preprocessing each list
    p1 = Preprocessing()
    lista_query = p1.preprocess_text(csv['query'].tolist())
    lista_response = p1.preprocess_text([str(i) for i in csv['response'].tolist()])
    
    #delete query and response columns
    csv.drop(['query', 'response'], axis=1, inplace=True)
    #add query and response columns
    csv['query'] = lista_query
    csv['response'] = lista_response
    csv = csv.reset_index(drop=True)
    #save csv file
    csv.to_csv(destino_path + 'dataset_confidence_065__3.csv', index=False)
    
if __name__ == "__main__":
    main()

'''
path = '/content/full.csv'

csv = pd.read_csv(path)

p = clean_csv(csv).clean(600)


p = Preprocessing()

text = ["Brasil! Mostra tua cara  Quero ver quem paga Pra gente ficar assim  Brasil! Qual é o teu negócio? O nome do teu sócio? Confia em mim", "Brasil! Mostra tua cara  Quero ver quem paga Pra gente ficar assim  Brasil! Qual é o teu negócio? O nome do teu sócio? Confia em mim",'Consultar a posição de entrega do último pedido',"Fantástico! Seu pedido já está na transportadora. Você receberá até o dia ##/##/####,Escreva *o número* da opção desejada,1 - Acompanhar a entrega do seu último Pedido 2 - Consultar a data da próxima entrega 3 - Consultar débito ou segunda via de boleto 4 - Consultar pontuação do Meu Mundo [[COMPANY_NAME]] 5 - Novo Modelo de Negócio [[COMPANY_NAME]] 6 - Outros assuntos 7 - Alterar o registro de consulta", ' ']

preprocess_text = p.preprocess_text(text)


'''