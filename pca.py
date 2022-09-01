# sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#itertools
import itertools

# pandas
import pandas as pd

#numpy 
import numpy as np

# matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
import matplotlib.patches as mpatches

#seaborn
import seaborn as sns

def do_pca(df):
    df_filtered = df.select_dtypes('number')
    p = Pipeline([("1",StandardScaler()),("2",PCA())])
    p.fit_transform(df_filtered)
    return (p[-1],df_filtered.columns)

def display_one_component(pca,features,component,rot=45):
    sns.set_style("ticks",{'axes.grid' : True})
    df = pd.DataFrame(data=list(zip(pca.components_[component],features)),columns=['weight','feature'])
    g = sns.catplot(data=df,x='feature',y='weight',aspect=3,palette="deep")
    if rot is not None:
        g.set_xticklabels(features, rotation=rot, ha='right').add_legend(title=f"C{component+1}")
    plt.show() 

def num_components(variance_ratios,p):
    total = 0
    def below(x):
        nonlocal total 
        total = total+x
        return (total < p)
    return range(len(list(itertools.takewhile(below,variance_ratios) )))

def display_components(pca,features,p=0.9,rot=45):
    for c in num_components(pca.explained_variance_ratio_,p):
        display_one_component(pca,features,c,rot=rot)

def display_scree_plot(pca,figsize_=(36,12)):
    scree = pca.explained_variance_ratio_*100
    _,ax = plt.subplots(figsize=figsize_)
    ax.bar(np.arange(len(scree))+1, scree)
    ax.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o',alpha=1)
    #ax.set_xticks(range(1,len(scree)+1,50),rotation=90.0)
    xl
    ax.set_xticklabels(range(1,len(scree)+1,10),rotation=90.0)
    ax.set_yticks(range(0,105,5))
    ax.grid(True)
    plt.xlabel("Rang de l'axe d'inertie")
    plt.ylabel("Pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)
    
def get_N_distinct_colors(N,excludes=['light','ish']):
    def iter_colors(N):
        f = lambda x: [e not in x for e in excludes]
        i = 0
        for (n,c) in mcolors.get_named_colors_mapping().items():
            if(i == N):
                return
            if( all(f(n)) ):
                i = i +1
                yield c
    return [c for c in iter_colors(N)]

def display_circles(pca,labels=None, label_rotation=0, lims=None,figsize_=(20, 20)):
    pcs = pca.components_
    N = pcs.shape[1]
    #colors = [mcolors.to_rgba(c) for c in list(mcolors.get_named_colors_mapping().values())[:N]]
    colors = [mcolors.to_rgba(c) for c in get_N_distinct_colors(N)]
    n = N -2
    for d1, d2 in list(zip(list(range(n)),list(range(1,n+1)))): 
        # initialisation de la figure
        fig, ax = plt.subplots(figsize=figsize_)
        
        # définition des limites du graphique (suite)
        xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])
        
        #l = max(map( abs,[xmin, xmax, ymin, ymax]))
        #xmax, ymax = [1.1*l]*2
        #xmin, ymin = [-1.1*l]*2 
        xmax, ymax = [1]*2
        xmin, ymin = [-1]*2 
        
        # representation des features
        lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
        ax.add_collection(LineCollection(lines, axes=ax, alpha=1, colors=colors))

        # affichage des noms des variables  
        if labels is not None:
            # creation de la legende
            labels_ = map(lambda t: f'({t[1]}) {t[0]}',zip(labels,list(range(1,len(labels)+1))))
            patches = list(map(lambda p: mpatches.Patch(label=p[0],color=p[1]),zip(labels_,colors)))
            ax.legend(handles=patches,loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=1)  
            for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                    plt.text(x, y, str(i+1), fontsize='10', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)

        # affichage du cercle
        #circle = plt.Circle((0,0), 1.08*l, facecolor='none', edgecolor='grey', ls='--')
        circle = plt.Circle((0,0), 1.0, facecolor='none', edgecolor='grey', ls='--')
        plt.gca().add_artist(circle)

        # définition des limites du graphique
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

        # affichage des lignes horizontales et verticales
        plt.plot([-1, 1], [0, 0], color='grey', ls='--')
        plt.plot([0, 0], [-1, 1], color='grey', ls='--')

        # nom des axes, avec le pourcentage d'inertie expliqué
        plt.xlabel('C{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
        plt.ylabel('C{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

        plt.title("Cercle des corrélations (C{} et C{})".format(d1+1, d2+1))
        plt.show(block=False)  