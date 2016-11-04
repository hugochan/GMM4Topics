'''
Created on Nov, 2016

@author: hugo

'''

import sys
import json
import numpy as np
from scipy import linalg
from gensim.models import word2vec

sys.path.append("../")
from gaussian_mixture import *

# def estimate_log_gaussian_prob(X, means, precisions_chol, covariance_type):
#     """Estimate the log Gaussian probability.

#     Parameters
#     ----------
#     X : array-like, shape (n_samples, n_features)

#     means : array-like, shape (n_components, n_features)

#     precisions_chol : array-like,
#         Cholesky decompositions of the precision matrices.
#         'full' : shape of (n_components, n_features, n_features)
#         'tied' : shape of (n_features, n_features)
#         'diag' : shape of (n_components, n_features)
#         'spherical' : shape of (n_components,)

#     covariance_type : {'full', 'tied', 'diag', 'spherical'}

#     Returns
#     -------
#     log_prob : array, shape (n_samples, n_components)
#     """
#     n_samples, n_features = X.shape
#     n_components, _ = means.shape
#     # det(precision_chol) is half of det(precision)
#     log_det = _compute_log_det_cholesky(
#         precisions_chol, covariance_type, n_features)

#     if covariance_type == 'full':
#         log_prob = np.empty((n_samples, n_components))
#         for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
#             y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
#             log_prob[:, k] = np.sum(np.square(y), axis=1)

#     elif covariance_type == 'tied':
#         log_prob = np.empty((n_samples, n_components))
#         for k, mu in enumerate(means):
#             y = np.dot(X, precisions_chol) - np.dot(mu, precisions_chol)
#             log_prob[:, k] = np.sum(np.square(y), axis=1)

#     elif covariance_type == 'diag':
#         precisions = precisions_chol ** 2
#         log_prob = (np.sum((means ** 2 * precisions), 1) -
#                     2. * np.dot(X, (means * precisions).T) +
#                     np.dot(X ** 2, precisions.T))

#     elif covariance_type == 'spherical':
#         precisions = precisions_chol ** 2
#         log_prob = (np.sum(means ** 2, 1) * precisions -
#                     2 * np.dot(X, means.T * precisions) +
#                     np.outer(row_norms(X, squared=True), precisions))
#     return -.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det


def revdict(d):
    """
    Reverse a dictionary mapping.
    When two keys map to the same value, only one of them will be kept in the
    result (which one is kept is arbitrary).
    """
    return dict((v, k) for (k, v) in d.iteritems())

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

def load_wordembs(in_model):
    return word2vec.Word2Vec.load_word2vec_format(in_model, binary=True)

def get_vocabembs(vocab_list, wordembs):
    new_vocab_list = []
    vocab_emb = []
    for word in vocab_list:
        if word in wordembs:
            new_vocab_list.append(word)
            vocab_emb.append(wordembs[word])
    return new_vocab_list, np.array(vocab_emb)

def generate_corpusdata(corpus, vocab, wordembs):
    rev_vocab = revdict(vocab)
    corpus_data = []
    n_unknown = 0
    for doc in corpus:
        word_vecs = []
        for word_idx, count in doc:
            if word_idx in rev_vocab and rev_vocab[word_idx] in wordembs:
                vec = wordembs[rev_vocab[word_idx]]
                word_vecs.extend([vec for i in range(count)])
            else:
                n_unknown += count
        corpus_data.append(np.array(word_vecs))
    if n_unknown > 0:
        print "# of unknown words in corpus (including duplicates): %s" % n_unknown
    return corpus_data

def show_components(vocab, vocab_emb, gmm, ntopw=10):
    vocab_logprobs = estimate_log_gaussian_prob(vocab_emb, gmm.means_, gmm.precisions_cholesky_, gmm.covariance_type)
    vocab_probs = np.exp(vocab_logprobs)
    vocab_probs /= np.sum(vocab_probs, axis=0)
    for idx in range(gmm.n_components):
        top_prob_words = sorted(zip(vocab, vocab_probs[:, idx].tolist()), key=lambda d:d[1], reverse=True)[:ntopw]
        str_topic = ' + '.join(['%s * %s' % (prob, word) for word, prob in top_prob_words])
        print str_topic


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
    wordembs = load_wordembs(in_wordembs)
    corpus_data = generate_corpusdata(corpus, vocab, wordembs)
    # import pdb;pdb.set_trace()

    # Fit a Gaussian mixture with EM using five components
    n_components = 2
    covar_type = 'diag'
    gmm = GaussianMixture(n_components=n_components, covariance_type=covar_type, tol=1e-6, max_iter=1000, n_init=1, verbose=2).fit(corpus_data)

    vocab_list, vocab_emb = get_vocabembs(vocab.keys(), wordembs)
    # import pdb;pdb.set_trace()
    show_components(vocab_list, vocab_emb, gmm, ntopw=10)
    # print "means: %s" % gmm.means_
    # print "covariances: %s" % gmm.covariances_
    # print "system output weights:"
    # print gmm.weights_


if __name__ == '__main__':
    main()
