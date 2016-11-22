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

gram=phrases.Phrases.load('sentences_bigram')

