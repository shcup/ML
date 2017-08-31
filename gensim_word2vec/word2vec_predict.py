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
from scipy.linalg.misc import norm

def loadFile2Numpy(file_name):
  cnt = 0
  seller_idx = {}
  vectors = np.empty((4997933, 100))
  norms = np.empty(4997933)
  for line in open(file_name):
    k,v = line.strip().split('\t')
    vec=np.fromstring(v, dtype=np.float32, sep=' ') 
    seller_idx[k]=cnt
    vectors[cnt]=vec
    norms[cnt]=norm(vec)
    cnt = cnt + 1
  print "finish to load the np vector"
  return seller_idx, vectors, norms

def consin(seller1, seller2, seller_idx, vectors, norms):
  if seller1 not in seller_idx or seller2 not in seller_idx:
    return 0.0

  idx1 = seller_idx[seller1]
  idx2 = seller_idx[seller2]
  vec1 = vectors[idx1]
  vec2 = vectors[idx2]
  n1 = norms[idx1]
  n2 = norms[idx2]
  return np.dot(vec1,vec2)/(n1 * n2)

def process_word2vec(file_name, seller_idx, vectors, norms):
  for line in open(file_name):
    sp = line.strip().split('\t')
    score = 0.0
    if len(sp) == 3:
      seller_id = sp[0]
      if len(sp[1]) > 0:
        leaf_seed, top_seed = sp[1].split('|', 1)
        if len(leaf_seed) > 0:
          his = leaf_seed.split(';')[0].split(',')[0]
          score = consin(seller_id, his, seller_idx, vectors, norms)
        elif len(top_seed) > 0:
          his = top_seed.split(';')[0].split(',')[0]
          score = consin(seller_id, his, seller_idx, vectors, norms)

      print str(score) + '\t' + sp[2] 

def process_word2vec_max(file_name, seller_idx, vectors, norms):
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
            s = consin(seller_id, id, seller_idx, vectors, norms)
            if s > score:
              score = s
        
        if len(top_seed) > 0:
          for his in top_seed.split(';'):
            id = his.split(',')[0]
            s = consin(seller_id, id, seller_idx, vectors, norms)
            if s > score:
              score = s
 
      print str(score) + '\t' + sp[2] 

def process_word2vec_avg(file_name, seller_idx, vectors, norms):
  for line in open(file_name):
    sp = line.strip().split('\t')
    score = 0.0
    cnt = 0
    if len(sp) == 3:
      seller_id = sp[0]
      if len(sp[1]) > 0:
        leaf_seed, top_seed = sp[1].split('|', 1)
        if len(leaf_seed) > 0:
          for his in leaf_seed.split(';'):
            id = his.split(',')[0]
            s = consin(seller_id, id, seller_idx, vectors, norms)
            score = s + score 
            cnt = cnt + 1
        elif len(top_seed) > 0:
          for his in top_seed.split(';'):
            id = his.split(',')[0]
            s = consin(seller_id, id, seller_idx, vectors, norms)
            score = s + score
            cnt = cnt + 1
 
      print str(score/cnt) + '\t' + sp[2] 


seller_idx, vectors, norms = loadFile2Numpy('output_word2vec.txt')
process_word2vec_max('shop_s2s_sample_data.txt', seller_idx, vectors, norms)
    
