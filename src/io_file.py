import pandas as pd
import joblib
from pathlib import Path,PureWindowsPath,PurePosixPath

root = PureWindowsPath(r'C:\Users\benjamin.bouchard\Documents\PERSONNEL\OCR\courseNLP\P5\final')
p = lambda rel,r=root: r / rel

input_file = "QueryResults.csv"
results = "stackOverFlow.csv"

def deserialize_list(df,cols):
    to_list_ = lambda x: list(map(lambda y: y.replace("'",""),x.replace("[","").replace("]","").split(", ")))
    for col in cols:
        df[col] = df[col].apply(to_list_)

def load_input():
    data = pd.read_csv( p(f'data/{input_file}') )
    return data

def load_data():
    data = pd.read_csv( p(f'data/{input_file}') )
    deserialize_list(data,['body-tokens','body-tokens-wov','title-tokens','ntags'])
    return data

def save_model(results,cs,name,prefix="supvs_model_"):
    return joblib.dump(
        value={
            k:{ c: results[k][c] for c in cs} for k in results.keys() 
        },
        filename= p(f"data/models/{prefix}{name}.joblib")
    )

save_results = lambda v,name,prefix="supvs_res_": joblib.dump(value=v,filename=p(f"data/results/{prefix}{name}.joblib"))
load_results = lambda name,prefix="supvs_res_": joblib.load(filename=p(f"data/results/{prefix}{name}.joblib"))

def save_emb(features,name):
    joblib.dump(value=features,filename=p(f"/data/embeddings/{name}.joblib"))

def load_emb(name):
    return joblib.load(filename=p(f"/data/embeddings/{name}.joblib"))
