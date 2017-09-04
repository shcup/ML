#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import logging
import os.path
import unittest
import tempfile
import itertools
import sys

import numpy as np

def loadCFData(file_name):
  cf = {}
  for line in open(file_name):
    k,v = line.strip().split('\t', 1)
    sp = v.split(' ')
    table = {}
    for item in sp:
      key, value = time.split(':', 1)
      table[key]=float(value)
    cf[k] = table

   return cf
 
 def getCFScore(id1, id2, cf):
  score = 0.0
  if id1 in cf:
    t = cf[id1]
    if id2 in t:
      score = t[id2]
  return score

 def process_word2vec_sum(file_name, cf):
  for line in open(file_name):
    sp = line.strip().split('\t')
    score = 0.0
    if len(sp) == 3:
      seller_id = sp[0]
      if len(sp[1]) > 0:
        leaf_seed, top_seed = sp[1].split('|', 1)
        if len(leaf_seed) > 0:
          for his in leaf_seed.split(';'):
            id = his.split(',')[0]
            s = getCFScore(his, seller_id, cf)
            score = score + s
        elif len(top_seed) > 0:
          for his in top_seed.split(';'):
            id = his.split(',')[0]
            s = getCFScore(his, seller_id, cf)
            score = score + s
 
      print str(score) + '\t' + sp[2] 


