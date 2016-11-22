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
from gensim.models import (tfidfmodel,word2vec,ldamodel, phrases)

class MySentences(object):
	def __init__(self, dirname):
		self.dirname = dirname

	def __iter__(self):
		for fname in os.listdir(self.dirname):
			for line in open(os.path.join(self.dirname, fname)):
				yield line.split()

sentences = MySentences('./sentences') # a memory-friendly iterator
bigram=phrases.Phrases(sentences)
bigram.save('./sentences_bigram')

