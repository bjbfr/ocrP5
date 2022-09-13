import re
from itertools import accumulate,chain,takewhile,compress
from collections import defaultdict

import pandas as pd

def count_tokens(df,col,term_count=False,term_freq=False,doc_count=False,doc_freq=False,out_df=False):
    """
    Compute statistics for tokens in <col> of dataframe <df> (elements of df[col] are expected to be lists).
    Statistics are either 
    - the number of times a term appears in the corpus (term_count = true) or its associated frequency (term_freq = true)
    - the number of documents in which a term appears (doc_count == true) or its associated frequency (doc_freq = true)

    term_count, term_freq, doc_count, doc_freq are mutually exclusive and it is expected that only one of it is set to true.

    if out_df is set to true a dataframe is returned otherwise a dictionary type is returned.

    """

    def get_out_col(**kwargs):
        """" 
            Returns out_col name based on input statistic to be computed.
            Throws an exception if more than one or no statistic is asked.
        """
        tmp = list(compress(kwargs.keys(),kwargs.values()))
        if len(tmp) != 1:
            raise Exception("No statistic were asked." if len(tmp) == 0 else f"Several statistics were asked ({tmp}).")
        else:
            return tmp[0]


    out_col = get_out_col(term_count=term_count,term_freq=term_freq,doc_count=doc_count,doc_freq=doc_freq)
    is_freq_stat = lambda   : out_col.endswith('_freq')
    is_term_stat = lambda   : out_col.startswith('term_')
    src_tokens   = lambda l : set(l) if is_freq_stat() else l

    # do counting.
    counter = defaultdict(int)
    def inc(x):
        counter[x] += 1

    df[col].apply(
            lambda l: list(
                map(lambda x: inc(x),src_tokens(l))
            )
        )

    # compute frequency if asked.
    if (is_freq_stat()):
        N = sum(counter.values())  if is_term_stat() else len(df[col])
        ret= {k:v/N for (k,v) in counter.items()}
    else:
        ret = counter

    if out_df == True:
        return pd.DataFrame(ret.items(),columns=['tokens',out_col])
    else:
        return ret

def pipe(df,actions,col):
    """
        Executes list of <actions> on column <col> of dataframe <df>.
        Returns the final result.
    """
    return list(accumulate(iterable=actions, func=lambda s,f: s.apply(f),initial=df[col]))[-1]

lower_str  = lambda x: x.lower()
lower_list = lambda l: list(map(lower_str,l))

list_2_str  = lambda l: '_'.join(sorted(l))
filter_empty_list = lambda l: list(filter(lambda x: len(x) > 0,l))

def make_sort_list(sorter=None):
    if sorter is not None:
        sort_list  = lambda l: sorted(l,key=sorter)
    else:
        sort_list  = lambda l: sorted(l)
    return sort_list

def make_count_sorter(count):
    sorter = lambda x: count.get(x) if count.get(x) is not None else -1
    return sorter

# binary list to int
list_2_int = lambda b:ba2int(bitarray(b.tolist()))

# def make_remove_pattern_list(signs=None,patterns=None):
#     if signs is None and patterns is None:
#         raise Exception("make_remove_pattern_list: nor signs or patterns is set !")
#     #signs patterns    
#     reg_exp0 = "[" + ' | '.join(signs)+ "]" if signs    is not None else ""
#     #other patterns
#     reg_exp1 =       ' | '.join(patterns)   if patterns is not None else "" 
#     #merge patterns
#     merge_patterns = lambda a,b: b if a == "" else (a if b == "" else a + " | " + b)
#     reg_exp = merge_patterns(reg_exp0,reg_exp1)
#     print(reg_exp)
#     p = re.compile(reg_exp)
#     #returned function
#     list_filter = lambda l: list(map(lambda x: re.sub(p,"",x),l))
#     return list_filter

def make_remove_list(pattern,rpl,exceptions):
    reg_exp = re.compile(pattern)
    #returned function
    list_filter = lambda l: list(
        map(lambda x: 
            re.sub(reg_exp,rpl,x) if x not in exceptions else x,
            l
            )
        )
    return list_filter

def make_remove_signs_str(signs=None):
    if signs is None:
        raise Exception("make_remove_signs_str: signs is not set !")
    #signs patterns    
    pattern = "[" + ' | '.join(signs)+ "]"
    reg_exp = re.compile(pattern)
    f = lambda str: re.sub(reg_exp," ",str)
    return f

def make_remove_signs_list(signs=None,rpl="",exceptions=[]):
    if signs is None:
        raise Exception("make_remove_signs_list: signs is not set !")
    #signs patterns    
    pattern = "[" + ' | '.join(signs)+ "]"
    return make_remove_list(pattern,rpl,exceptions)

def make_remove_patterns_list(patterns=None,rpl="",exceptions=[]):
    if patterns is None:
        raise Exception("make_remove_patterns_list: patterns is not set !")
    #reg exp from patterns
    pattern = ' | '.join(patterns) 
    return make_remove_list(pattern,rpl,exceptions)

def make_filter_list(exclude=None,include=None):
    # not working ??
    #cond = lambda x: x not in exclude if exclude is not None else ( (lambda x: x in include) if include is not None else (lambda x: True))
    if exclude is not None and include is not None:
        raise Exception("make_filter_list: both exclude and include are set !")
    if exclude is None and include is None:
        return  lambda l: l
    else:
        if exclude is not None:
            cond = lambda x: x not in exclude
        else:
            cond = lambda x: x in include
        return lambda l: list(filter(cond,l)) 

def make_tokenizer_list(pattern):
    reg_exp = re.compile(pattern)
    def split(acc,x):
        acc.extend(reg_exp.split(x))
        return acc
    def tokenizer(l):
         return list(accumulate(l,split,initial=[]))[-1]
    return tokenizer

def make_extractor_pattern_str(pattern):
    regexp = re.compile(pattern)
    def extractor(x):
        return regexp.findall(x)
    return extractor     

def merge_tokens(df,cols):
    """ merge list (tokens) contained in cols of df """
    merger = lambda r: list(chain(*[r[k] for k in cols]))
    return df.apply(merger,axis=1)

def filter_term_docfreq(df,col,max=1.0,min=0.0):
    """
        filter-out terms that have doc frequency less than min and more than max
    """
    counts = count_tokens(df,col,doc_freq=True)
    return df[col].apply(make_filter_list(include=[k for (k,v) in counts.items() if v > min and v < max]))        