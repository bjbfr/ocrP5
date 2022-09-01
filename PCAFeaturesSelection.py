from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA
import itertools

class PCAFeaturesSelection(TransformerMixin):
    """
    Transformer for manual selection of features using sklearn style transform method.  
    """

    def __init__(self,level=0.8):
        self.__level__ = level
        self.__pipe__  = Pipeline([("1",StandardScaler()),("2",PCA())])
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
