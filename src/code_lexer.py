#import re
import itertools

from pygments.lexers import guess_lexer
from pygments.util  import ClassNotFound

#reg_exp_lexer = re.compile("<class 'pygments.lexers.(.*)'>")

def try_guess_lexer(code):
    try:
        lexer = guess_lexer(code)
        #t = str(type(guess))
        #lexer = reg_exp_lexer.findall(t)[0]
    except ClassNotFound as e:
        return None
    except BaseException as err:
        print(f"Error: code: {code} - Exception: {err}")
        return None
    return lexer.name

def guess_code_language(acc,code):
    lexer = try_guess_lexer(code)
    if lexer is not None:
        if lexer == 'special.TextLexer':
            acc['text'].append(code)
        else:
            acc['lexer'].append(lexer)
    return acc


handle_code_section = lambda l : list(itertools.accumulate(l,guess_code_language,initial={'text':[],'lexer':[]}))[-1]


