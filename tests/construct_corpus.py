'''
Created on Nov, 2016

@author: hugo

This program generates the input corpus data for GMMs model.

The words of each document are assumed exchangeable. Thus, each document
is succinctly represented as a sparse vector of word counts. The data is a file
where each line (which represents a doc) is of the form:

     [term_1]:[count] [term_2]:[count] ... [term_M]:[count]

where [M] is the number of unique terms in the document; the [count] associated
with each term is how many times that term appeared in the document.
Note that [term_1] is an integer which indexes the term; it is not a string.

'''
import os
import sys
import json
from collections import defaultdict
import chardet
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

word_tokenizer = RegexpTokenizer(r'\w+')
cached_stop_words = stopwords.words("english")

def save_json(data, file):
    try:
        with open(file, 'w') as outfile:
            json.dump(data, outfile)
    except Exception as e:
        print e

def save_corpus(out_corpus, doc_word_freq, vocab_dict):
    try:
        with open(out_corpus, 'w') as fp:
            for _, val in doc_word_freq.iteritems():
                word_count = {}
                for word, freq in val.iteritems():
                    try:
                        word_count[vocab_dict[word]] = freq
                    except: # word is not in vocab, i.e., this word should be filtered out
                        pass
                fp.write('%s\n' % ' '.join(['%s:%s' % (idx, count) for idx, count in word_count.iteritems()]))
    except Exception as e:
        print e
        sys.exit()
    else:
        fp.close()

def load_data(corpus_path):
    word_freq = defaultdict(lambda: 0) # count the number of times a word appears in a corpus
    doc_word_freq = defaultdict(dict) # count the number of times a word appears in a doc
    files = (filename for filename in os.listdir(corpus_path) if os.path.isfile(os.path.join(corpus_path, filename)))
    for filename in files:
        if filename[0] == '.':
            continue
        try:
            with open(os.path.join(corpus_path, filename), 'r') as fp:
                text = fp.read().lower()
                words = word_tokenizer.tokenize(text)
                words = [word for word in words if word not in cached_stop_words]

                for i in range(len(words)):
                    # doc-word frequency
                    try:
                        doc_word_freq[filename][words[i]] += 1
                    except:
                        doc_word_freq[filename][words[i]] = 1
                    # word frequency
                    word_freq[words[i]] += 1
        except Exception as e:
            print e
            sys.exit()
        else:
            fp.close()
    return word_freq, doc_word_freq

def get_vocab_dict(word_freq, threshold=5):
    idx = 0
    vocab_dict = {}
    for word, freq in word_freq.iteritems():
        if freq < threshold:
            continue
        vocab_dict[word] = idx
        idx += 1
    return vocab_dict

# def get_low_freq_words(word_freq, threshold=5):
#     return [word for word, freq in word_freq.iteritems() if freq < threshold]


if __name__ == "__main__":
    usage = 'python construct_corpus.py [corpus_path] [out_vocab] [out_corpus]'
    try:
        corpus_path = sys.argv[1]
        out_vocab = sys.argv[2]
        out_corpus = sys.argv[3]
    except Exception as e:
        print e
        sys.exit()

    word_freq, doc_word_freq = load_data(corpus_path)
    print 'finished loading'
    # filter_list = get_low_freq_words(word_freq, threshold=2)
    vocab_dict = get_vocab_dict(word_freq, threshold=5)
    save_corpus(out_corpus, doc_word_freq, vocab_dict)
    save_json(vocab_dict, out_vocab)
