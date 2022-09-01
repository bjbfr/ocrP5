#joblib
from pyexpat import model
import joblib

#sklearn
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier

#gensim
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary

#local
from vectorizer import *
import lda as ldam

__PATH__  = "./models"

#supervised model
__SUPVS_NAME__   = "supvs_res3"
__SUPVS_INDEX__ = 3

#unsupervised model
__UNSUPVS_NAME__   = "lda"
__UNSUPVS_INDEX__ = 0

__DIRECT_MATCHING__ = "tags"

Vn = lambda d,k,n: d[k][n]

class SupervisedModel:
    def __init__(self) -> None:
        raw_model = joblib.load(f"{__PATH__}/{__SUPVS_NAME__}.joblib")
        self.model = Vn(raw_model,__SUPVS_INDEX__,'model')
        self.vect  = Vn(raw_model,__SUPVS_INDEX__,'vectorizer')

    def predict(self,tokens):
        # vectorize input tokens
        v = self.vect.transform([tokens]).toarray()
        # predict
        r = self.model.predict(v)
        #get tags
        tags = self.vect.inverse_transform(r)[0].tolist()
        return tags

class UnSupervisedModel:
    def __init__(self) -> None:
        raw_model = joblib.load(f"{__PATH__}/{__UNSUPVS_NAME__}.joblib")
        self.model = Vn(raw_model,__UNSUPVS_INDEX__,'model')
        self.dict  = Vn(raw_model,__UNSUPVS_INDEX__,'dict')

    def predict(self,tokens):
        return ldam.predict([tokens],self.model,self.dict)[0].tolist()

class DirectMatching:
        def __init__(self) -> None:
            raw_model = joblib.load(f"{__PATH__}/{__DIRECT_MATCHING__}.joblib")
            self.model = raw_model
        
        def predict(self,tokens):
            return [ token for token in tokens if token in self.model]

