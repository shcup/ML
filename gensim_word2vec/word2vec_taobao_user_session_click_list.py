#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import logging
import os.path
import unittest
import tempfile
import itertools

import numpy

from gensim.utils import to_unicode
from gensim.interfaces import TransformedCorpus
from gensim.corpora import (bleicorpus, mmcorpus, lowcorpus, svmlightcorpus,
                            ucicorpus, malletcorpus, textcorpus, indexedcorpus)
from gensim.models import (tfidfmodel,word2vec)

class MySentences(object):
	def __init__(self, dirname):
		self.dirname = dirname

	def __iter__(self):
		for fname in os.listdir(self.dirname):
			for line in open(os.path.join(self.dirname, fname)):
				yield line.split()

sentences = MySentences('/home/haochuan.shc/data/tools/gensim/sdmo_sentence') # a memory-friendly iterator
model = word2vec.Word2Vec(sentences, size=100, window=5, min_count=5, workers=48)
model.save('word2vec_words')

model = word2vec.Word2Vec.load('word2vec_words')
output = open('output_word2vec.txt', 'w')
count = len(model.wv.index2word)
idx = 0
while idx < count:
	tag = model.wv.index2word[idx]
	idx = idx + 1
	res=[]
	for f in model.wv[tag]:
		res.append(str(f))
	output.write(tag + "\t" + " ".join(res) + "\n")

output.close()


#model.wv.similar_by_word('661559176')
#model.wv.similarity
