from itertools import takewhile,chain
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wordcloud
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from src import preprocessing
from src import vectorizer

#p% of tags' usage for n most used tags
tokens_usage  = lambda tokens,n,col: tokens.nlargest(n,col)[col].sum()

# nb of tokens required to get p% (0<p<=1) of tokens' usage
nb_tokens_for_coverage = lambda stats,p,col: list(takewhile(lambda k: tokens_usage(stats,k,col) < p,range(len(stats))))[-1]+1

def termf_coverage(data,col,ns):
    """
        Plot cumulative term frequency for key values (10% to 90%)
        plus the values for values in ns (number of tokens) 
    """
    stats_termf = preprocessing.count_tokens(data,col,term_freq=True,out_df=True)
    # number of tokens for key values
    ns = sorted(list(map(lambda p: nb_tokens_for_coverage(stats_termf,p,'term_freq'),np.arange(0.1,1.0,0.1))) + ns)

    #
    usage = lambda stats,col: list(map(lambda n: tokens_usage(stats,n,col),ns))
    df =  pd.DataFrame(
        data={'n most used tokens':ns,'cumulative term frequency':usage(stats_termf,'term_freq')}
    )

    # display
    plot_params = {
        'figsize':(18,12),
        'grid':True,
        'yticks':np.arange(0,1.05,0.05),
        'legend':False,
        'title':f"Cumulative term frequency for {col}"
    } 

    df.plot(kind="bar",x='n most used tokens',y='cumulative term frequency',**plot_params)


#n_less_used_tokens = lambda n,data,col:preprocessing.count_tokens(data,col,doc_freq=True,out_df=True).nsmallest(n,'doc_freq')

def plot_docf(data,col,n,figsize_=(25,50)):
    """
        Plot document frequency of the n most used tokens contained in data[col]
    """
    # compute stats
    stat_col = 'doc_freq'
    #n_most_used_tokens = preprocessing.count_tokens(data,col,doc_freq=True,out_df=True).nlargest(n,stat_col)
    n_most_used_tokens_ = preprocessing.n_most_used_tokens(n,data,col)

    # create fig and axes
    N = 25                             # number of tokens per ax
    p = 4                              # number of ax per row
    m = math.ceil(math.ceil(n/N)/p)    # number of rows needed
    fig, axs = plt.subplots(nrows=m, ncols=p, figsize=figsize_,constrained_layout=True,sharey=False)
    
    z = list(zip(list(range(0,n,N)),list(range(N,n,N))))
    #if n%N != 0:
    end = z[-1] if len(z) > 0 else (0,0)
    z.append((end[1],n))

    fig.suptitle(f"Document frequency for {col} - lhs: percent - rhs: docs count")
    l = len(data[col])
    for ((start,end),ax) in zip(z,chain(*axs) if len(axs.shape) == 2 else axs):
        secax = ax.secondary_yaxis('right',functions=(lambda x :x*l ,lambda x : x/l))
        _ = n_most_used_tokens_.iloc[start:end,:].plot(kind="bar",x="tokens",y=stat_col,ax=ax,legend=False,grid=True,xlabel="")

# def df_docCount_nmost_used(d,col,ids):
#     c = preprocessing.count_tokens(d,col,doc_freq=True,out_df=True).sort_values(by=['doc_freq'],ascending=False).iloc[ids,:]
#     c['doc count'] = len(d)*c['doc_freq']
#     c['n most used tokens'] = ids
#     return c.loc[:,['n most used tokens','doc_freq','doc count']]

def display_word_cloud(data,col,max_words=None):
    """
        Plot a word cloud (based on document frequency) for tokens in data[col]
        Optionally, only display max_words first tokens.
    """
    freq = preprocessing.count_tokens(data,col,doc_freq=True)
    max_words = len(freq.keys()) if max_words is None else max_words
    wc = wordcloud.WordCloud(
        random_state=8,
        normalize_plurals=False,
        width=1800,height=600,
        max_words=max_words,
        stopwords=[]
    )

    wc.generate_from_frequencies(freq)

    # create a figure
    fig, ax = plt.subplots(1,1, figsize = (25,12))
    # add interpolation = bilinear to smooth things out
    plt.imshow(wc, interpolation='bilinear')
    # and remove the axis
    plt.axis("off")

def do_pca(df):
    """
        compute pca
    """
    df_filtered = df.select_dtypes('number')
    p = Pipeline([("1",StandardScaler()),("2",PCA())])
    p.fit_transform(df_filtered)
    return (p[-1],df_filtered.columns)

def display_pca(data,body_min,body_max,title_min,title_max,col_body='body-tokens'):
    """
        Compute and display scree plot for pca
    """
    data['filtered-body-tokens'] = preprocessing.filter_term_docfreq(data,col_body,min=body_min,max=body_max)
    data['filtered-title-tokens'] = preprocessing.filter_term_docfreq(data,'title-tokens',min=title_min,max=title_max)
    data['tokens'] = preprocessing.merge_tokens(data,['filtered-body-tokens','filtered-title-tokens'])
    (values,_) = vectorizer.BOW_vectorizer(data['tokens'],CountVectorizer)
    pca_df = pd.DataFrame(data=values.toarray())
    (pca,_) = do_pca(pca_df)
    print(f"body_min: {body_min} - body_max: {body_max} - title_min: {title_min} - title_max: {title_max}")
    print(f'pca.shape: {pca_df.shape}')
    display_scree_plot(pca)
    data.drop(columns=['tokens','filtered-body-tokens','filtered-title-tokens'],inplace=True)

def display_scree_plot(pca,figsize_=(36,12),N=5):
    """
        Display scree plot for pca
    """
    take_one_every = lambda l,n: list(map(lambda i: l[i],range(0,len(l),n)))
    csum = (pca.explained_variance_ratio_*100).cumsum()
    scree = np.array(take_one_every(csum,N))
    xs = N*(np.arange(len(scree)))   
    # add last point
    np.append(scree,csum[-1])
    np.append(xs,len(csum))
    _,ax = plt.subplots(figsize=figsize_)
    ax.plot(xs, scree,c="red",marker='o',alpha=1)
    ax.set_ylim(ymin=0.0)
    ax.set_xlim(xmin=0.0,xmax=max(xs))
    ax.set_xticks(ticks=xs,labels=xs,rotation=90.0)
    ax.set_yticks(range(0,105,5))
    ax.grid(True)
    plt.xlabel("Rang de l'axe d'inertie")
    plt.ylabel("Pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)