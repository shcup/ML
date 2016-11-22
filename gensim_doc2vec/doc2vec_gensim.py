#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import logging
import sys
import os.path
import unittest
import tempfile
import itertools

import numpy

from gensim.utils import to_unicode
from gensim.interfaces import TransformedCorpus
from gensim.corpora import (bleicorpus, mmcorpus, lowcorpus, svmlightcorpus,
                            ucicorpus, malletcorpus, textcorpus, indexedcorpus)
from gensim.models import (tfidfmodel,word2vec,doc2vec)

tag=sys.argv[1]
if len(tag) == 0:
	sys.exit(1)

class MyTagDocument(object):
	def __init__(self, filename):
		self.filename = filename

	def __iter__(self):
		for line in open(self.filename):
			items = line.split('\t', 1)
                    	yield doc2vec.TaggedDocument(to_unicode(items[1]).split(), [items[0]])


documents=MyTagDocument('./data/articles.txt.' + tag)

model=doc2vec.Doc2Vec(documents, size=100, window=8, min_count=5, workers=8)
model.save('doc2vec')

output = open('output_doc2vec.txt.' + tag, 'w')
count = model.docvecs.count
idx = 0
while idx < count:
	tag = model.docvecs.index_to_doctag(idx)
	idx = idx + 1
	res=[]
	for f in model.docvecs[tag]:
		res.append(str(f))
	output.write(tag + "\tdoc2vec\t" + " ".join(res) + "\n")

output.close()

#output_similarity = open('output_doc2vec_similarity.txt', 'w')
#count = model.docvecs.count
#idx = 0
#while idx < count:
#	tag = model.docvecs.index_to_doctag(idx)
#	idx = idx + 1
#	res = []
#	for d in model.docvecs.most_similar(tag):
#		res.append(str(d))
#	output_similarity.write(tag + "\t" + " ".join(res) + "\n")
#output_similarity.close()


