def id(x): return x

def BOW_vectorizer(tokens,vectorizer_ctor,**args):
    # skip preprocessing and tokenizing steps
    vectorizer = vectorizer_ctor(preprocessor=id,tokenizer=id,lowercase=False,**args)
    X = vectorizer.fit_transform(tokens)
    return (X,vectorizer)