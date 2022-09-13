from collections import defaultdict
from html.parser import HTMLParser
from html import unescape 
from itertools import accumulate

#from code_lexer import try_guess_lexer
from src import code_lexer

class Strcontainer:

    def __init__(self):
        self.content = ""

    def add(self,tag,data):
        self.content += data
    
    def clear(self):
        self.content = ""

    def copy(self):
        return self.content

class MixContainer:
    def __init__(self):
        self.clear()

    def add_text(self,data):
        self.content['text'] += f" {data} "

    def add(self,tag,data):
        if tag == "code":
            lexer = code_lexer.try_guess_lexer(data)
            if lexer is not None:
                if lexer == 'Text only':
                    self.add_text(data)
                else:
                    self.content['text'] += f" [{lexer}] "
                    self.content['code'].append(data)

                #print(f"{lexer} {data}")
        else:
            self.add_text(data)
    
    def clear(self):
        self.content = {'text':"", 'code':[]}

    def copy(self):
        return self.content['text']


class DefaultListcontainer:

    def __init__(self):
        self.content = defaultdict(list)

    def add(self,tag,data):
        self.content[tag].append(data)
    
    def clear(self):
        self.content.clear()

    def copy(self):
        return self.content.copy()

class MyHTMLParser(HTMLParser):
    """
        Html parser to parse Body field.
        Note that Body is not fully-html compliant.
    """
    def __init__(self,filter,container=DefaultListcontainer):
        self.tags    = []
        self.filter  = filter
        self.content = container()
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
            #assert top_tag == tag, f"Error when unstacking tag (seen tag: {tag} <> unstacked tag: {top_tag})"

    def handle_data(self, data):
        data = data.strip()
        if data != "":
            top_tag = self.tags[-1] if len(self.tags) > 0 else 'NO_TAG'
            if( not(self.filter(top_tag)) ):
                self.content.add(top_tag,data)
                #print(f"Handle data:\n\ttag: {top_tag}\n\tdata: =>{data.strip()}<=")
    
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
    except BaseException as ex:
        print(f"ERROR:{x} (exception: {ex})")
    return html_parser.content.copy()    

def make_extractor_keys(exclude=None,include=None):
    """
        helping function used in make_keys_getter
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