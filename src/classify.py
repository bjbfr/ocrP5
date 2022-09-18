
import sklearn as sk
from scipy.sparse import csr_matrix
import lda as ldam

from src import preprocessing
from src import common
from src import vectorizer
from src import tools

filter_tokens = lambda df,Y_tokens,nb_tags: df[Y_tokens].apply(preprocessing.make_filter_list(include=preprocessing.n_most_used_tokens(nb_tags,df,Y_tokens)['tokens'].values))

def unsupervised_classify(df,body_tokens,title_tokens,Y_tokens,body_max_df,body_min_df,nb_tags,title_min_df,title_max_df,n_term,num_topics,nb_strat = 30,**args):
  
  #on ne garde que les nb_tags tags les plus utilisés
  df[tools.filtered_name(Y_tokens)] = filter_tokens(df,Y_tokens,nb_tags) 

  #création de la colonne pour stratification (d'après les nb_strat tags les plus courants)
  df[tools.filtered_name(Y_tokens) + '-str'] = preprocessing.pipe(df,
    [
    preprocessing.make_filter_list(include=preprocessing.n_most_used_tokens(nb_strat,df,Y_tokens)['tokens'].values),
    preprocessing.list_2_str
    ],
    tools.filtered_name(Y_tokens)
  )

  input_tokens = 'tokens'
  # filtrage
  df[tools.filtered_name(body_tokens)]  = preprocessing.filter_term_docfreq(df,body_tokens,max=body_max_df,min=body_min_df)
  df[tools.filtered_name(title_tokens)] = preprocessing.filter_term_docfreq(df,title_tokens,max=title_max_df,min=title_min_df)
  
  #fusion body|title
  df[input_tokens] = preprocessing.merge_tokens(df,[tools.filtered_name(body_tokens),tools.filtered_name(title_tokens)])  
  
  #split train-test
  X_train, X_test, y_train, y_test = common.split_df(df,x_cols=[input_tokens,'Id'],y_col=[Y_tokens,tools.filtered_name(Y_tokens),'Id'],stratified_col =(tools.filtered_name(Y_tokens) + '-str'))
  
  # fit
  (lda,dictionary,corpus) = ldam.fit(X_train[input_tokens],num_topics=num_topics,**args)
  
  #predict
  predicted_tag_probas = ldam.predict(X_test[input_tokens],lda,dictionary,n_term=n_term)
  
  #score
  scores = {}
  predicted_tags = predicted_tag_probas[:,:,0]
  mlb =  sk.preprocessing.MultiLabelBinarizer()#sparse_output=True

  # fit encoder with orignal tags plus the ones found by unsupervised process
  all_tags = df[tools.filtered_name(Y_tokens)].values.tolist()
  all_tags.extend(predicted_tags.tolist())
  mlb.fit(all_tags)
  t = mlb.transform(y_test[tools.filtered_name(Y_tokens)])
  t_predicted = mlb.transform(predicted_tags)
  scores['jaccard']  = sk.metrics.jscore(csr_matrix(t),csr_matrix(t_predicted))

  #clean-up
  df.drop(columns=[input_tokens,tools.filtered_name(body_tokens),tools.filtered_name(title_tokens),tools.filtered_name(Y_tokens),tools.filtered_name(Y_tokens)+'-str'],inplace=True)
  
  return {
    **{
    'model':lda,
    'dict':dictionary,
    'corpus':corpus,
    'predicted_tag_probas':predicted_tag_probas,
    't':y_test[Y_tokens].values,
    },
    **scores
  }

def BOW_supervised_classify(data,pipe,pipe_params,vect,Y_tokens,nb_tags,
                            body_tokens=None,body_max_df=None,body_min_df=None,title_tokens=None,title_max_df=None,title_min_df=None,
                            col_id='Id'):

    #stratified
    stratified_col_ = 'filtered-tag-str'
    data[tools.filtered_name(Y_tokens)] = filter_tokens(data,Y_tokens,nb_tags)
    data[stratified_col_] = data[tools.filtered_name(Y_tokens)].apply(preprocessing.list_2_str)

    #merge title and body
    input_tokens = 'input-tokens'
    data[tools.filtered_name(body_tokens)]  = preprocessing.filter_term_docfreq(data,body_tokens,max=body_max_df,min=body_min_df)
    data[tools.filtered_name(title_tokens)] = preprocessing.filter_term_docfreq(data,title_tokens,max=title_max_df,min=title_min_df)
    data[input_tokens] = preprocessing.merge_tokens(data,[tools.filtered_name(title_tokens),tools.filtered_name(body_tokens)])  

    # split 
    X_train, X_test, y_train, y_test = common.split_df(data,
                                                x_cols=[input_tokens,col_id],
                                                y_col=[Y_tokens,tools.filtered_name(Y_tokens),col_id],
                                                stratified_col = stratified_col_
                                            )
    # train/test y
    y_train_v = y_train[tools.filtered_name(Y_tokens)].values
    y_test_v  = y_test[tools.filtered_name(Y_tokens)].values   
    all_tokens = data[tools.filtered_name(Y_tokens)].values

    #vectorize
    (X,vectorizer_) = vectorizer.BOW_vectorizer(X_train[input_tokens],vect)
    Z              = vectorizer_.transform(X_test[input_tokens])

    return { 
      **classify(X,Z,y_train_v,y_test_v,all_tokens,pipe,pipe_params),
      **{'vectorizer':vectorizer_}
    }

def classify(X,Z,y_train_v,y_test_v,all_tokens,pipe,pipe_params):
    #y: encode tags
    mlb = sk.preprocessing.MultiLabelBinarizer()
    mlb.fit(all_tokens)
    y   = mlb.transform(y_train_v)
    t   = mlb.transform(y_test_v)
    
    #fit
    pipe.set_params(**pipe_params)
    model = sk.multiclass.OneVsRestClassifier(pipe).fit(X,y)
    
    #predict
    t_predicted = model.predict(Z)

    #scores
    scores = {}
    scores['jaccard']  = sk.metrics.jscore(csr_matrix(t),csr_matrix(t_predicted))

    return {
        **{'X':X,'y':y,'Z':Z,'t':t,'t_predicted':t_predicted,
        'mlb':mlb,'model':model},
        **scores
    }