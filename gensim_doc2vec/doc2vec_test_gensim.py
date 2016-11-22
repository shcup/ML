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
from gensim.models import (tfidfmodel,word2vec,doc2vec)


from multiprocessing import Process
from multiprocessing import Pool
import os

def run_proc(start, end):

	print 'Run task %d %d (%s)...' % (start, end, os.getpid())
	
	model=doc2vec.Doc2Vec.load('doc2vec.20161101_141701_24473');	
	output_similarity= open('output_doc2vec_similarity.txt.' + str(start), 'w')
	count = model.docvecs.count
	idx = start
	while idx <= end:
		tag = model.docvecs.index_to_doctag(idx)
		idx = idx + 1
		res = []
		for d in model.docvecs.most_similar(tag):
			res.append(str(d))
		output_similarity.write(str(idx) + '\t' + tag + "\t" + " ".join(res) + "\n")
	output_similarity.close()

def multi_proc_start():
	model=doc2vec.Doc2Vec.load('doc2vec.20161101_141701_24473');
	length=model.docvecs.count
	number_worker=10

	length_each=length/number_worker

	print 'Parent process %s.' % os.getpid()
	p = Pool(10)

	start=0
	idx=0
	while idx < number_worker:
		if (idx == number_worker -1):
			p.apply_async(run_proc, args=(start,length - 1,))
			#print str(start) + "\t" + str(length - 1)
		else:
			end = start + length_each - 1
			if end >= length:
				end = length - 1
			p.apply_async(run_proc, args=(start,end,))
			#print str(start) + "\t" + str(end)

		start = end + 1	
		idx = idx + 1

	p.close()
	p.join()

def TestForOutput():
	output_similarityu= open('output_doc2vec_similarity.txt', 'w')
	count = model.docvecs.count
	idx = 0
	while idx < count:
		tag = model.docvecs.index_to_doctag(idx)
		idx = idx + 1
		res = []
		for d in model.docvecs.most_similar(tag):
			res.append(str(d))
		output_similarity.write(tag + "\t" + " ".join(res) + "\n")
	output_similarity.close()



	output = open('output_doc2vec.txt', 'w')
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



