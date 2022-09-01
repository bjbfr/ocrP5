#numpy
import numpy as np

#gensim
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary


corpus = lambda docs,dic: [dic.doc2bow(doc) for doc in docs] 

def tuple_to_list(ts,num_topics,p_topics):
    ret = [0.0]*num_topics
    def set_value(ret,t,min):
        ret[t[0]] = t[1] if t[1] > p_topics else 0.0
    _ = list(map(lambda t:set_value(ret,t,min),ts))
    return np.array(ret)

topics = lambda tokens,lda,dictionary,p_topic: np.array( list( map( 
                    lambda x: tuple_to_list(lda.get_document_topics(x),lda.num_topics,p_topic),
                    corpus(tokens,dictionary)
                  )
                ))

def predicted_topics(tokens,lda,dictionary,p_topic,n_topic):
  """
  """
  if n_topic is not None:
    topics_ = topics(tokens,lda,dictionary,-0.01)
    ret = np.zeros_like(topics_)
    def set_value(t):
      i = t[0]
      js = t[1]
      for j in js:
        ret[i][j] = topics_[i][j]
    _ = list( map(lambda t: set_value(t) ,enumerate(np.argsort(topics_)[:,-n_topic:].tolist())))
    return ret

  if p_topic is not None:
    return topics(tokens,lda,dictionary,p_topic)

def predicted_terms(predicted_topics,lda,p_term,n_term):
  """
  """
  terms = np.matmul(
    predicted_topics,
    lda.get_topics()
  )
  
  if n_term is not None:
    ret = [ [(lda.id2word[j],terms[i][j]) for j in js]  for (i,js) in enumerate(np.argsort(terms)[:,-n_term:].tolist()) ]

  if p_term is not None:
    m = predicted_topics.shape[0]
    ret = [[]]*m 
    _ = list(map(lambda t: ret[t[0]].append((lda.id2word[t[1]],terms[t[0]][t[1]])) ,np.argwhere(terms >= p_term).tolist()))
  
  return np.array(ret,dtype=object) 

def check_predict(lda,p_topic,n_topic,p_term,n_term):
    not_all_set = lambda x,y:not(x is not None and y is not None)
    assert not_all_set(p_topic,n_topic), f"predict - Both p_topic and n_topic are set."
    assert not_all_set(p_term,n_term), f"predict - Both p_term and n_term are set."
    def out_of_bounds(x,l,d,msg): 
        if x is not None and x > l:
          print(msg)
          return d
        return x
    p_l = 1.0
    p_d = 0.95 # default probability
    p_topic = out_of_bounds(p_topic,p_l,p_d,f"p_topic is out of bounds {p_l} forced to {p_d}")
    p_term  = out_of_bounds(p_term,p_l,p_d,f"p_term is out of bounds {p_l} forced to {p_d}")
    n_topic = out_of_bounds(n_topic,lda.num_topics,lda.num_topics,f"n_topic is out of bounds {lda.num_topics} forced to {lda.num_topics}")
    n_term  = out_of_bounds(n_term,lda.num_terms,lda.num_terms,f"n_term is out of bounds {lda.num_terms} forced to {lda.num_terms}")

    # if no topic values is given, consider all topics
    if n_topic is None and p_topic is None:
      n_topic = lda.num_topics
      #print(f"n_topic is forced to {lda.num_topics}")

    # if no term values is given, consider default probability value
    if n_term is None and p_term is None:
      p_term = p_d
      #print(f"p_term is forced to {p_term}")
    
    return (p_topic,n_topic,p_term,n_term)

def predict(tokens,lda,dictionary,p_topic=None,n_topic=None,p_term=None,n_term=None):
  """
    - predicts terms for documents in column 'input_tokens' of dataframe 'df' based on lda model (lda,dictionary)
    (predicted terms came from dictionary)
    - p_topic is the minimum probability for topics to be considered - n_topic is the number of topics to be considered 
      (both p_topic and n_topic CANNOT be set)
    - n_topic is the minimum probability for terms to be considered - n_term is the number of terms to be considered 
      (both p_term and n_term CANNOT be set)
  """
  (p_topic,n_topic,p_term,n_term) = check_predict(lda,p_topic,n_topic,p_term,n_term)

  # predicted topics
  topics_ = predicted_topics(tokens,lda,dictionary,p_topic,n_topic)
  # predicted terms
  return predicted_terms(topics_,lda,p_term,n_term)

def fit(tokens,num_topics,**args):
        """
            fits lda model with 'num_topics' based on column 'input_tokens' of dataframe 'X_train'
            returns (lda model,dictionary)
        """
        dictionary = Dictionary(tokens)
        temp = dictionary[0] # load dictionary to get access to id2token
        # fit
        c = corpus(tokens,dictionary)
        lda = LdaModel(corpus=c,num_topics=num_topics,id2word=dictionary.id2token,**args)
        return (lda,dictionary,c)