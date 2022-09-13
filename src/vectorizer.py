import itertools
from select import select
from tokenize import endpats
import numpy as np
import tensorflow as tf
# import tensorflow_hub as hub
import tensorflow.keras
from tensorflow.keras import backend as K

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import metrics as kmetrics
from tensorflow.keras.layers import *      # Embedding, GlobalAveragePooling1D 
from tensorflow.keras.models import Model

# Bert
import os
import transformers
from transformers import *
import tensorflow_hub as hub
import tensorflow_text 

os.environ["TF_KERAS"]='1'

def BOW_vectorizer(tokens,vectorizer_ctor,**args):
    Id = lambda x: x
    # skip preprocessing and tokenizing steps
    vectorizer = vectorizer_ctor(preprocessor=Id,tokenizer=Id,lowercase=False,**args)
    X = vectorizer.fit_transform(tokens)
    return (X,vectorizer)

class W2V_Vectoriser:
    def __init__(self,w2v_model,w2v_size=300): #,maxlen=24
        #self.maxlen       = maxlen
        self.w2v_model    = w2v_model  #word2vec model
        self.w2v_size     = w2v_size   #word2vec vector
        self.words        = w2v_model.index_to_key
        self.tokenizer    = None
        self.word_index   = None
        self.emb_model    = None # embedding model

    def fit_tokeniser(self,X):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(X.values)
        self.word_index = self.tokenizer.word_index 

    def create_embedding_matrix(self):
        self.embedding_matrix = np.zeros((len(self.word_index)+1, self.w2v_size))
        for word, idx in self.word_index.items():
            if word in self.words:
                embedding_vector = self.w2v_model[word]
                if embedding_vector is not None:
                    self.embedding_matrix[idx] = embedding_vector

    def fit(self,X):
        self.fit_tokeniser(X)
        self.create_embedding_matrix()
        maxlen = X.apply(lambda x: len(x)).max()
        word_input=Input(shape=(maxlen,),dtype='float64')  
        word_embedding=Embedding(input_dim=len(self.word_index)+1,
                         output_dim=self.w2v_size,
                         weights = [self.embedding_matrix],
                         input_length=maxlen)(word_input)
        word_vec=GlobalAveragePooling1D()(word_embedding)
        self.emb_model = Model([word_input],word_vec)
    
    def transform(self,X):
        return self.emb_model.predict(pad_sequences(
            self.tokenizer.texts_to_sequences(X.values),
            #maxlen=self.maxlen,
            padding='post')
        )

    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)

# Fonction de préparation des sentences
def bert_inp_fct(sentences, bert_tokenizer, max_length) :
    input_ids=[]
    token_type_ids = []
    attention_mask=[]
    bert_inp_tot = []

    for sent in sentences:
        bert_inp = bert_tokenizer.encode_plus(sent,
                                              add_special_tokens = True,
                                              max_length = max_length,
                                              padding='max_length',
                                              return_attention_mask = True, 
                                              return_token_type_ids=True,
                                              truncation=True,
                                              return_tensors="tf")
    
        input_ids.append(bert_inp['input_ids'][0])
        token_type_ids.append(bert_inp['token_type_ids'][0])
        attention_mask.append(bert_inp['attention_mask'][0])
        bert_inp_tot.append((bert_inp['input_ids'][0], 
                             bert_inp['token_type_ids'][0], 
                             bert_inp['attention_mask'][0]))

    input_ids = np.asarray(input_ids)
    token_type_ids = np.asarray(token_type_ids)
    attention_mask = np.array(attention_mask)
    
    return input_ids, token_type_ids, attention_mask, bert_inp_tot


# Fonction de création des features
def feature_BERT_fct(model, model_type, sentences, max_length, b_size, mode='HF') :
    batch_size = b_size
    batch_size_pred = b_size
    bert_tokenizer = AutoTokenizer.from_pretrained(model_type)

    r = len(sentences)%batch_size
    Q = len(sentences)//batch_size
    last_step = Q-1
    for step in range(Q) :
        idx = step*batch_size
        end = idx+batch_size

        if step == last_step:
            end += r

        input_ids, token_type_ids, attention_mask, bert_inp_tot = bert_inp_fct(sentences[idx:end], 
                                                                      bert_tokenizer, max_length)
        
        if mode=='HF' :    # Bert HuggingFace
            outputs = model.predict([input_ids, attention_mask, token_type_ids], batch_size=batch_size_pred)
            last_hidden_states = outputs.last_hidden_state

        if mode=='TFhub' : # Bert Tensorflow Hub
            text_preprocessed = {"input_word_ids" : input_ids, 
                                 "input_mask" : attention_mask, 
                                 "input_type_ids" : token_type_ids}
            outputs = model(text_preprocessed)
            last_hidden_states = outputs['sequence_output']
             
        if step ==0 :
            last_hidden_states_tot = last_hidden_states
            last_hidden_states_tot_0 = last_hidden_states
        else :
            last_hidden_states_tot = np.concatenate((last_hidden_states_tot,last_hidden_states))
    
    features_bert = np.array(last_hidden_states_tot).mean(axis=1)
     
    return features_bert, last_hidden_states_tot

class BERT_Vectoriser:

    def __init__(self,mode,max_length = 64,batch_size = 10):
            self.mode = mode
            self.max_length = max_length
            self.batch_size = batch_size
            if mode=='HF' :
                self.model = TFAutoModel.from_pretrained('bert-base-uncased')
            if mode=='TFhub':
                model_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4'
                self.model = hub.KerasLayer(model_url, trainable=True)
            self.model_type = 'bert-base-uncased'

    def fit_transform(self,X):
        return self.transform(X)
    
    def transform(self,X):
            features, _ = feature_BERT_fct(self.model, self.model_type, X, self.max_length, self.batch_size, mode=self.mode)
            return features

def feature_USE_fct(model,sentences, b_size) :
    batch_size = b_size

    r = len(sentences)%batch_size
    Q = len(sentences)//batch_size
    last_step = Q-1
    for step in range(Q) :
        idx = step*batch_size
        end = idx+batch_size

        if step == last_step:
            end += r

        idx = step*batch_size
        feat = model(sentences[idx:end])

        if step ==0 :
            features = feat
        else :
            features = np.concatenate((features,feat))

    return features

class USE_Vectorizer:
    def __init__(self,batch_size = 10):
        self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        self.batch_size = batch_size

    def fit_transform(self,X):
        return self.transform(X)
    
    def transform(self,X):
            return feature_USE_fct(self.model, X,self.batch_size)

#
def id(x): return x
def vectorize(tokens,vectorizer_ctor,**args):

    if  str(vectorizer_ctor).__contains__('sklearn.feature_extraction'):
        # skip preprocessing and tokenizing steps
        vectorizer = vectorizer_ctor(preprocessor=id,tokenizer=id,lowercase=False,**args)
    else:
        vectorizer = vectorizer_ctor

    X = vectorizer.fit_transform(tokens)
    return (X,vectorizer)