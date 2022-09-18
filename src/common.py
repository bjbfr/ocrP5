import itertools
import math 

import pandas as pd
import sklearn as sk

def split_df(df,x_cols,y_col,stratified_col,test_size=0.2,random_state=42):
    index = None
    try:
        index = df.loc[df.duplicated(subset=stratified_col, keep=False),:].index
    except:
        pass

    if index is not None:
        #stratified part
        d = df.loc[index,:]
        X_train_s, X_test_s, y_train_s, y_test_s = sk.model_selection.train_test_split(d[x_cols],d[y_col],test_size=test_size,random_state=random_state,stratify=d[stratified_col])

        # not stratified part
        index = df.index.difference(index)
        d = df.loc[index,:]
        X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(d[x_cols],d[y_col],test_size=test_size,random_state=random_state)

        return (
                pd.concat([X_train_s,X_train]),
                pd.concat([X_test_s,X_test]),
                pd.concat([y_train_s,y_train]),
                pd.concat([y_test_s,y_test])
            )

    else:
        
        X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(df[x_cols],df[y_col],test_size=test_size,random_state=random_state)

        return (
            X_train,
            X_test,
            y_train,
            y_test
        )


# Jaccard score
jscore = lambda y_true, y_pred: sk.metrics.jaccard_score(y_true, y_pred, average='samples',zero_division=0.0)

class PCAFeaturesSelection(sk.base.TransformerMixin):
    """
        Transformer for manual selection of features using sklearn style transform method.  
    """

    def __init__(self,level=0.8):
        self.__level__ = level
        self.__pipe__  = sk.pipeline.Pipeline([("1",sk.preprocessing.StandardScaler()),("2",sk.decomposition.PCA())])
        self.__range__ = None
        pass

    def set_params(self,level):
        self.__level__ = level

    def __range_components__(self):
        total = 0
        def below(x):
            nonlocal total 
            total = total+x
            return (total < self.__level__)
        self.__range__ = range(len(list(itertools.takewhile(below, self.__pipe__[-1].explained_variance_ratio_) )))

    def fit(self, X, y=None):
        self.__pipe__.fit(X)
        self.__range_components__()
        return self
 
    def transform(self, X,y=None):
        X_trans = self.__pipe__.transform(X)
        return X_trans[:,self.__range__]


take = lambda n,res: list(map(lambda t: t[n],res))

class pipe_params:
    """helping class to generate grid parameters for a pipe"""

    def __init__(self,pipe):
        self.__params__ = pipe.get_params()

    @staticmethod
    def __is_pipe_params__(pipe):
            return lambda s: any(map(lambda x: f"{x}__" in s , take(0,pipe.get_params()['steps'])))

    @staticmethod
    def pipe_params(pipe):
        p = pipe_params.__is_pipe_params__(pipe)
        return {k:v for (k,v) in pipe.get_params().items() if p(k) }

    @staticmethod
    def __overload_pipe_params__(pipe_params,param_grid):
        return {**pipe_params,**param_grid}

    @staticmethod
    def gen_grid_params(param_grid):
        """
        >param_grid={'a':[1,2],'b':[3,4],'c':[7]}
        >gen_grid_params(param_grid)
        gives:
        [   {'a': 1, 'b': 3, 'c': 7},
            {'a': 1, 'b': 4, 'c': 7},
            {'a': 2, 'b': 3, 'c': 7},
            {'a': 2, 'b': 4, 'c': 7}
        ]
        """
        dict_keys     = lambda d : [k for k in d.keys()]
        dict_prod_len = lambda d : math.prod(map(lambda x: len(x),d.values()))
        return [ {k:v for (k,v) in t}
                for t in [
                        list(zip(x[0],x[1])) for x in 
                            list( zip(
                                    [dict_keys(param_grid)]*dict_prod_len(param_grid),
                                    list(itertools.product(*param_grid.values())),
                                )
                            )
                        ]
        ]

    def make(self,param_grid):
        """Generate grid parameters for pipe based on param_grid"""
        return [
            pipe_params.__overload_pipe_params__(self.__params__,pg)
            for pg in pipe_params.gen_grid_params(param_grid)
        ]