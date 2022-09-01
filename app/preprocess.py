import numpy as np
import pandas as pd
import math

import re
from collections import defaultdict
from html.parser import HTMLParser
from itertools import accumulate,chain,takewhile
import warnings
import math

# sklearn
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

#wordcloud
import wordcloud

#nltk
import nltk
from nltk.stem import WordNetLemmatizer

#spacy
#import spacy
#sp_full=  spacy.load("en_core_web_trf")

warnings.warn('User provided device_type of \'cuda\', but CUDA is not available. Disabling')


def pipe(df,actions,col):
    """
        Executes list of <actions> on column <col> of dataframe <df>.
        Returns the final result.
    """
    return list(accumulate(iterable=actions, func=lambda s,f: s.apply(f),initial=df[col]))[-1]

lower_str         = lambda x: x.lower()
lower_list        = lambda l: list(map(lower_str,l))

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

def count_tokens(df,col,term_count=True,term_freq=False,doc_freq=False,out_df=False):
    """ count tokens contain in col of dataframe df """
    if term_freq == doc_freq == term_count:
        raise Exception("count_tokens: term_freq and doc_freq are equal !")

    out_col = 'count'    
    counter = defaultdict(int)
    def inc(x):
        counter[x] += 1

    if term_freq == True:
        out_col = 'term_freq'
        df[col].apply(
            lambda l: list(
                map(lambda x: inc(x),l)
            )
        )

    if doc_freq == True:
        out_col = 'doc_freq'
        df[col].apply(
            lambda l: list(
                map(lambda x: inc(x),set(l))  # remove duplicates set(l)
            )
        )

    if (term_freq is True or doc_freq is True):
        N = sum(counter.values())  if term_freq == True else len(df[col])
        ret= {k:v/N for (k,v) in counter.items()}
    else:
        ret = counter

    if out_df == True:
        return pd.DataFrame(ret.items(),columns=['tokens',out_col])
    else:
        return ret

def filter_term_docfreq(df,col,max=1.0,min=0.0):
    """
        filter-out terms that have doc frequency less than min and more than max
    """
    counts = count_tokens(df,col,doc_freq=True)
    return df[col].apply(make_filter_list(include=[k for (k,v) in counts.items() if v > min and v < max]))


class MyHTMLParser(HTMLParser):
    """
        Html parser to parse Body field.
        Note that Body is not fully-compliant html.
    """
    def __init__(self,filter):
        self.tags    = []
        self.filter  = filter
        self.content = defaultdict(list)
        self.single_tags = ['br','img','hr'] 
        super().__init__() 

    def pop_stack(self,tag):
        N = len(self.tags)
        if  N > 0:
            try:
                j = list(reversed(self.tags)).index(tag)    
                i = N - 1 - j
                self.tags.remove(i)
            except:
                pass

    def handle_starttag(self, tag, attrs):
        if tag not in self.single_tags:
            #print("Encountered a start tag:", tag)
            self.tags.append(tag)

    def handle_endtag(self, tag):
        if tag not in self.single_tags:
            #print("Encountered an end tag :", tag)
            self.pop_stack(tag)
            #top_tag = self.tags.pop()
            #assert top_tag == tag, f"Error when unstacking tag (seen tag: {tag} <> unstacked tag: {top_tag})"

    def handle_data(self, data):
        top_tag = self.tags[-1] if len(self.tags) > 0 else 'NO_TAG'
        if( not(self.filter(top_tag)) ):
            self.content[top_tag].append(data)
            #print("Encountered some data  :", data)
    
    def clear(self):
        self.content.clear()
        self.tags.clear()
        self.reset()

def do_parse(x,html_parser):
    """
        Clear and feed html parser.
        Function to be called in pipe to treat the full dataset
    """
    html_parser.clear()
    try:
        html_parser.feed(x)
    except:
        print(f"EXCEPTION:{x}")
    return html_parser.content.copy()    

def make_extractor_keys(exclude=None,include=None):
    """
        helping function used in  make_keys_getter
    """
    if include is None and exclude is not None:
        def extractor(acc,item):
            (k,v) = item
            if k not in exclude:
                acc.extend(v)
            return acc
    elif exclude is None and include is not None:
        def extractor(acc,item):
            (k,v) = item
            if k in include:
                acc.extend(v)
            return acc
    else:
        def extractor(acc,item):
            (k,v) = item
            acc.extend(v)
            return acc
    return extractor

def make_keys_getter(exclude=None,include=None):
    """
        extract keys in dict returned by html parser
    """
    extractor_keys = make_extractor_keys(exclude,include)
    return lambda x : list(accumulate(x.items(),extractor_keys,initial=[]))[-1]

from nltk.corpus import wordnet as wn
tag_map = {
        'CC':None, # coordin. conjunction (and, but, or)  
        'CD':wn.NOUN, # cardinal number (one, two)             
        'DT':None, # determiner (a, the)                    
        'EX':wn.ADV, # existential ‘there’ (there)           
        'FW':None, # foreign word (mea culpa)             
        'IN':wn.ADV, # preposition/sub-conj (of, in, by)   
        'JJ':[wn.ADJ, wn.ADJ_SAT], # adjective (yellow)                  
        'JJR':[wn.ADJ, wn.ADJ_SAT], # adj., comparative (bigger)          
        'JJS':[wn.ADJ, wn.ADJ_SAT], # adj., superlative (wildest)           
        'LS':None, # list item marker (1, 2, One)          
        'MD':None, # modal (can, should)                    
        'NN':wn.NOUN, # noun, sing. or mass (llama)          
        'NNS':wn.NOUN, # noun, plural (llamas)                  
        'NNP':wn.NOUN, # proper noun, sing. (IBM)              
        'NNPS':wn.NOUN, # proper noun, plural (Carolinas)
        'PDT':[wn.ADJ, wn.ADJ_SAT], # predeterminer (all, both)            
        'POS':None, # possessive ending (’s )               
        'PRP':None, # personal pronoun (I, you, he)     
        'PRP$':None, # possessive pronoun (your, one’s)    
        'RB':wn.ADV, # adverb (quickly, never)            
        'RBR':wn.ADV, # adverb, comparative (faster)        
        'RBS':wn.ADV, # adverb, superlative (fastest)     
        'RP':[wn.ADJ, wn.ADJ_SAT], # particle (up, off)
        'SYM':None, # symbol (+,%, &)
        'TO':None, # “to” (to)
        'UH':None, # interjection (ah, oops)
        'VB':wn.VERB, # verb base form (eat)
        'VBD':wn.VERB, # verb past tense (ate)
        'VBG':wn.VERB, # verb gerund (eating)
        'VBN':wn.VERB, # verb past participle (eaten)
        'VBP':wn.VERB, # verb non-3sg pres (eat)
        'VBZ':wn.VERB, # verb 3sg pres (eats)
        'WDT':None, # wh-determiner (which, that)
        'WP':None, # wh-pronoun (what, who)
        'WP$':None, # possessive (wh- whose)
        'WRB':None, # wh-adverb (how, where)
        '$':None, #  dollar sign ($)
        '#':None, # pound sign (#)
        '“':None, # left quote (‘ or “)
        '”':None, # right quote (’ or ”)
        '(':None, # left parenthesis ([, (, {, <)
        ')':None, # right parenthesis (], ), }, >)
        ',':None, # comma (,)
        '.':None, # sentence-final punc (. ! ?)
        ':':None # mid-sentence punc (: ; ... – -)
    }
def pos_tagger(nltk_tag):
    return tag_map.get(nltk_tag)


def make_lemmatizer(allowed_postags=['VB','VBD','VBG','VBN','VBP','VBZ','NN','NNS','NNP','NNPS']):
    wnl = WordNetLemmatizer()
    lemmatizer = lambda doc: [ 
        wnl.lemmatize(token,pos_tagger(nltk_tag)) if nltk_tag in allowed_postags else ''
        for (token,nltk_tag) in nltk.pos_tag(doc)   
    ]
    return lemmatizer

common_signs   = [
    '\\(','\\)','\\[','\\]','\\{','\\}','\\?','\\.',';',',',':','\\|','\\=','\\-','\\+','"',"'",'&','%','\\*','\\','/','\\!','#',
    '_','<','>','@','§','~'
]
common_pattern = ["[0-9]*"]
exceptions = ['c++','g++','c#']

# merge stops words from different sources
# https://www.ranks.nl/stopwords
rank_nl= ["able","about","above","abst","accordance","according","accordingly","across","act","actually","added","adj","affected","affecting","affects","after","afterwards","again","against","ah","all","almost","alone","along","already","also","although","always","am","among","amongst","an","and","announce","another","any","anybody","anyhow","anymore","anyone","anything","anyway","anyways","anywhere","apparently","approximately","are","aren","arent","arise","around","as","aside","ask","asking","at","auth","available","away","awfully","b","back","be","became","because","become","becomes","becoming","been","before","beforehand","begin","beginning","beginnings","begins","behind","being","believe","below","beside","besides","between","beyond","biol","both","brief","briefly","but","by","c","ca","came","can","cannot","can't","cause","causes","certain","certainly","co","com","come","comes","contain","containing","contains","could","couldnt","d","date","did","didn't","different","do","does","doesn't","doing","done","don't","down","downwards","due","during","e","each","ed","edu","effect","eg","eight","eighty","either","else","elsewhere","end","ending","enough","especially","et","et-al","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","except","f","far","few","ff","fifth","first","five","fix","followed","following","follows","for","former","formerly","forth","found","four","from","further","furthermore","g","gave","get","gets","getting","give","given","gives","giving","go","goes","gone","got","gotten","h","had","happens","hardly","has","hasn't","have","haven't","having","he","hed","hence","her","here","hereafter","hereby","herein","heres","hereupon","hers","herself","hes","hi","hid","him","himself","his","hither","home","how","howbeit","however","hundred","i","id","ie","if","i'll","im","immediate","immediately","importance","important","in","inc","indeed","index","information","instead","into","invention","inward","is","isn't","it","itd","it'll","its","itself","i've","j","just","k","keep	keeps","kept","kg","km","know","known","knows","l","largely","last","lately","later","latter","latterly","least","less","lest","let","lets","like","liked","likely","line","little","'ll","look","looking","looks","lot","ltd","m","made","mainly","make","makes","many","may","maybe","me","mean","means","meantime","meanwhile","merely","mg","might","million","miss","ml","more","moreover","most","mostly","mr","mrs","much","mug","must","my","myself","n","na","name","namely","nay","nd","near","nearly","necessarily","necessary","need","needs","neither","never","nevertheless","new","next","nine","ninety","no","nobody","non","none","nonetheless","noone","nor","normally","nos","not","noted","nothing","now","nowhere","o","obtain","obtained","obviously","of","off","often","oh","ok","okay","old","omitted","on","once","one","ones","only","onto","or","ord","other","others","otherwise","ought","our","ours","ourselves","out","outside","over","overall","owing","own","p","page","pages","part","particular","particularly","past","per","perhaps","placed","please","plus","poorly","possible","possibly","potentially","pp","predominantly","present","previously","primarily","probably","promptly","proud","provides","put","q","que","quickly","quite","qv","r","ran","rather","rd","re","readily","really","recent","recently","ref","refs","regarding","regardless","regards","related","relatively","research","respectively","resulted","resulting","results","right","run","s","said","same","saw","say","saying","says","sec","section","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sent","seven","several","shall","she","shed","she'll","shes","should","shouldn't","show","showed","shown","showns","shows","significant","significantly","similar","similarly","since","six","slightly","so","some","somebody","somehow","someone","somethan","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specifically","specified","specify","specifying","still","stop","strongly","sub","substantially","successfully","such","sufficiently","suggest","sup","sure	t","take","taken","taking","tell","tends","th","than","thank","thanks","thanx","that","that'll","thats","that've","the","their","theirs","them","themselves","then","thence","there","thereafter","thereby","thered","therefore","therein","there'll","thereof","therere","theres","thereto","thereupon","there've","these","they","theyd","they'll","theyre","they've","think","this","those","thou","though","thoughh","thousand","throug","through","throughout","thru","thus","til","tip","to","together","too","took","toward","towards","tried","tries","truly","try","trying","ts","twice","two","u","un","under","unfortunately","unless","unlike","unlikely","until","unto","up","upon","ups","us","use","used","useful","usefully","usefulness","uses","using","usually","v","value","various","'ve","very","via","viz","vol","vols","vs","w","want","wants","was","wasnt","way","we","wed","welcome","we'll","went","were","werent","we've","what","whatever","what'll","whats","when","whence","whenever","where","whereafter","whereas","whereby","wherein","wheres","whereupon","wherever","whether","which","while","whim","whither","who","whod","whoever","whole","who'll","whom","whomever","whos","whose","why","widely","willing","wish","with","within","without","wont","words","world","would","wouldnt","www","x","y","yes","yet","you","youd","you'll","your","youre","yours","yourself","yourselves","you've","z","zero"]

# stop words sources
#stopwords_sources = [wordcloud.STOPWORDS,nltk.corpus.stopwords.words('english'),sp_full.Defaults.stop_words,rank_nl]
stopwords_sources = [wordcloud.STOPWORDS,nltk.corpus.stopwords.words('english'),rank_nl]

def merge_stop_words(*args):
    return list(set(chain(*args)))

__STOPWORDS__ = sorted(merge_stop_words(*stopwords_sources))

# API
def preprocess_body(input):
    data = pd.DataFrame(data=[input],columns=['Body'])
    data['contents'] = data['Body'].apply(do_parse,args=(MyHTMLParser(filter= lambda x: False ),))
    data['body-tokens-wov'] = pipe(
        data,
        [ 
            make_keys_getter(exclude=['code']), 
            lower_list,
            make_remove_signs_list(signs=common_signs,rpl=" ",exceptions=exceptions),
            make_tokenizer_list('\\s+'),
            make_remove_patterns_list(patterns=common_pattern,rpl=""),
            filter_empty_list,
            make_filter_list(exclude=__STOPWORDS__),
            make_lemmatizer(allowed_postags=['NN','NNS','NNP','NNPS']),
            filter_empty_list
        ],
        col='contents'
    )
    data['body-tokens-wov'] = data['body-tokens-wov'].apply(make_filter_list(exclude=['code','question','problem','data','help','thing','idea','imagine']))
    return data['body-tokens-wov'].values[0]


def preprocess_title(input):
    data = pd.DataFrame(data=[input],columns=['Title'])
    data['title-tokens'] = pipe(data,
    [ 
        lambda x: re.compile('\\s+').split(x),
        make_remove_patterns_list(patterns=common_pattern,rpl=""),
        lower_list,
        make_remove_signs_list(signs=common_signs,rpl=" ",exceptions=exceptions),
        make_tokenizer_list('\\s+'),
        filter_empty_list,
        make_filter_list(exclude=__STOPWORDS__),
        make_lemmatizer(),
        filter_empty_list
    ],
    col='Title'
    )
    return data['title-tokens'].values[0]

# TEST
# body="""<p>Two points I don’t understand about RDBMS being CA in CAP Theorem :</p>

# <p>1) It says RDBMS is <strong>not</strong> <strong>Partition Tolerant</strong> but how is RDBMS <strong>any less</strong> Partition Tolerant than other technologies like MongoDB or Cassandra? Is there a RDBMS setup where we give up CA to make it AP or CP?</p>

# <p>2) How is it CAP-Available? Is it through master-slave setup? As in when the master dies, slave takes over writes?</p>

# <p>I’m a novice at DB architecture and CAP theorem so please bear with me.</p>
# """

# title="Why isn't RDBMS Partition Tolerant in CAP Theorem and why is it Available?"

# print(preprocess_body(body))
# print(preprocess_title(title))