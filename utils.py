'''
Created on Nov, 2016

@author: hugo

'''

import numpy as np

def revdict(d):
    """
    Reverse a dictionary mapping.
    When two keys map to the same value, only one of them will be kept in the
    result (which one is kept is arbitrary).
    """
    return dict((v, k) for (k, v) in d.iteritems())

def norm(vec):
    return vec / np.linalg.norm(vec)

def load_wordembs(in_model, type_, vocab=None):
    if type_ == 'w2v':
        from gensim.models import word2vec
        return word2vec.Word2Vec.load_word2vec_format(in_model, binary=True)
    elif type_ == 'txt':
        if not vocab:
            raise ValueError("vocab is not given")
        wordembs = {}
        try:
            with open(in_model, 'r') as f:
                for line in f:
                    word_emb = line.strip('\r ').split()
                    if word_emb[0] in vocab:
                        wordembs[word_emb[0]] = np.array([float(x) for x in word_emb[1:]])
        except Exception as e:
            raise e
        else:
            f.close()
            return wordembs
    else:
        raise ValueError("Unknown type_: %s" % type_)
