'''
Created on Nov, 2016

@author: hugo

'''

import sys
import json
import numpy as np
from scipy import linalg


sys.path.append("../")
from gaussian_mixture import *
from utils import *


def load_corpus(in_corpus):
    corpus = []
    try:
        with open(in_corpus, 'r') as f:
            for line in f:
                doc = [(int(each.split(':')[0]), int(each.split(':')[1])) for each in line.strip('\r ').split()]
                corpus.append(doc)
    except Exception as e:
        print e
        return
    else:
        f.close()
        return corpus

def load_vocab(in_vocab):
    try:
        with open(in_vocab, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print e
        return
    else:
        f.close()
        return data

def get_vocabembs(vocab_list, wordembs):
    new_vocab_list = []
    vocab_emb = []
    for word in vocab_list:
        if word in wordembs:
            new_vocab_list.append(word)
            vec = norm(wordembs[word])
            vocab_emb.append(vec)
    return new_vocab_list, np.array(vocab_emb)

def generate_corpusdata(corpus, vocab, wordembs):
    rev_vocab = revdict(vocab)
    corpus_data = []
    n_unknown = 0
    for doc in corpus:
        word_vecs = []
        for word_idx, count in doc:
            if word_idx in rev_vocab and rev_vocab[word_idx] in wordembs:
                # normalize
                vec = norm(wordembs[rev_vocab[word_idx]])
                word_vecs.extend([vec for i in range(count)])
            else:
                n_unknown += count
        corpus_data.append(np.array(word_vecs))
    if n_unknown > 0:
        print "# of unknown words in corpus (including duplicates): %s" % n_unknown
    return corpus_data

def get_components(vocab, vocab_emb, gmm, ntopw=10):
    components = []
    vocab_logprobs = estimate_log_gaussian_prob(vocab_emb, gmm.means_, gmm.precisions_cholesky_, gmm.covariance_type)
    vocab_probs = np.exp(vocab_logprobs)
    vocab_probs /= np.sum(vocab_probs, axis=0)
    for idx in range(gmm.n_components):
        top_prob_words = sorted(zip(vocab, vocab_probs[:, idx].tolist()), key=lambda d:d[1], reverse=True)[:ntopw]
        components.append(top_prob_words)
    return components

def print_components(components):
    for i in range(len(components)):
        str_topic = ' + '.join(['%s * %s' % (prob, word) for word, prob in components[i]])
        print 'topic %s:' % i
        print str_topic
        print

def main():
    usage = 'python gmm4topics.py [in_corpus] [in_vocab] [in_wordembs]'
    try:
        in_corpus = sys.argv[1]
        in_vocab = sys.argv[2]
        in_wordembs = sys.argv[3]
    except Exception as e:
        print e
        sys.exit()

    print "running %s" % " ".join(sys.argv)
    corpus = load_corpus(in_corpus)
    vocab = load_vocab(in_vocab)
    wordembs = load_wordembs(in_wordembs, 'txt', vocab=vocab.keys())
    corpus_data = generate_corpusdata(corpus, vocab, wordembs)

    # Fit a Gaussian mixture with EM using five components
    n_components = 10
    covar_type = 'diag'
    gmm = GaussianMixture(n_components=n_components, covariance_type=covar_type, tol=1e-6, max_iter=1000, n_init=1, verbose=2).fit(corpus_data)

    vocab_list, vocab_emb = get_vocabembs(vocab.keys(), wordembs)
    components = get_components(vocab_list, vocab_emb, gmm, ntopw=10)
    print_components(components)
    # print "means: %s" % gmm.means_
    # print "covariances: %s" % gmm.covariances_
    # print "system output weights:"
    # print gmm.weights_


if __name__ == '__main__':
    main()
