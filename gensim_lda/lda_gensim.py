#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import sys
import logging
import os.path
import unittest
import tempfile
import itertools

import numpy

from gensim.utils import to_unicode
from gensim.interfaces import TransformedCorpus
from gensim.corpora import (bleicorpus, mmcorpus, lowcorpus, svmlightcorpus,
                            ucicorpus, malletcorpus, textcorpus, indexedcorpus, dictionary)
from gensim.models import (tfidfmodel,word2vec,ldamodel)

print 'start'
train_set=[]
for line in open('articles.txt'):
	items = line.strip().split('\t', 1)
	if len(items) < 2:
		continue
	words = items[1].strip().split(' ')
	train_set.append(words)

print 'construct dict'
dic = dictionary.Dictionary(train_set)
print 'doc2bow'
corpus = [dic.doc2bow(text) for text in train_set]
print 'ifidf'
tfidf = tfidfmodel.TfidfModel(corpus)
print 'ifidf corpus'
corpus_tfidf = tfidf[corpus]
print 'lda model'
lda = ldamodel.LdaModel(corpus_tfidf, id2word = dic, num_topics = 1000,  iterations = 1300, alpha = 0.15, eta = 0.01)
print 'corpus_tfidf'
corpus_lda = lda[corpus_tfidf]

lda.save('lda_model')

