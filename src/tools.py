import re
import itertools

from src import tools
from src import classify
from src import io_file
from src import common
from src import vectorizer

# hooks for config names transforms
vect_reg_exp = re.compile("<class 'sklearn.feature_extraction.text.(.*)'>")
vect_trans = lambda x: vect_reg_exp.findall(f"{x}")[0] if len(vect_reg_exp.findall(f"{x}")) > 0 else x

#  log_loss  Vs squared_error
loss_trans = lambda x : 'Logistic' if x == 'squared_error' else 'SVM'
df_trans = lambda x : f"{round(100*x,3)}%"

transformers = {'classifier__loss': loss_trans, 'vect': vect_trans,'body_min_df':df_trans,'body_max_df':df_trans,'title_min_df':df_trans,'title_max_df':df_trans}
# tranform config names
transform = lambda k,v: transformers[k](v) if transformers.get(k) is not None else v


# define names of config
dict_keys = lambda d0,d1: [k for k in {**d0,**d1}.keys()]
fmt_str  = lambda l: ('&'.join(list(map(lambda t: f"{t[0]}-{{{t[1]}}}",l))))
config_name = lambda d0,d1: fmt_str(
      filter(
        lambda t: t[1] in  dict_keys(d0,d1),
        [
          ('Classifier','classifier__loss'),('Vect','vect'), ('var-level','var-reducer__level'), ('nb_tags','nb_tags'),
          ('body_min_df','body_min_df'),('body_max_df','body_max_df'), ('title_min_df','title_min_df'),
          ('title_max_df','title_max_df')
        ],
      )
    ).format_map({k: transform(k,v) for (k,v) in {**d0,**d1}.items()})

filtered_name = lambda name: 'filtered-' + name

## Run config
def do_supervised_classify(data,config):
  pipe = config['pipe']
  type = config['type']
  #ret = {}
  i = 0
  for (pipe_params_,params) in list(
                                  itertools.product(
                                                    common.pipe_params(pipe).make(config['pipe_params']),
                                                    common.pipe_params.gen_grid_params(config['params'])
                                                  )
                                  ):
    name = config_name(pipe_params_,params)
    
    print(f"Run config:\n{name} ...")

    if (type == 'bow'):
      results = classify.BOW_supervised_classify(data,**params,pipe=pipe,pipe_params=pipe_params_)
    else:
      embs = vectorizer.do_embedding(data,**params)
      results = classify.classify(data,**embs,pipe=pipe,pipe_params=pipe_params_)

    res = { 
            **results, 
            **{'pipe_params':pipe_params_,'params':params,'name':name}
          }

    io_file.save_results(res,name)
    io_file.save_model(res,['model','vectorizer','name'],name)
    i = i + 1
  #return ret

def run_config(data,config):
    _ = do_supervised_classify(data,config)
    