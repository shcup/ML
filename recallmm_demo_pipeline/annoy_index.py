#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import sys
from annoy import AnnoyIndex

reload(sys)
sys.setdefaultencoding("utf-8")

def LoadVector(filename, vec_len):
  ids =[]
  vecs =[]
  for line in open(filename):
    sp = line.strip().split('\t')
    oid = sp[0]
    vec = vec_len * [0]
    idx = 0
    for f in sp[1].strip().split(','):
      vec[idx] = float(f)
      idx = idx + 1
    vecs.append(vec)
    ids.append(oid)
  
  return ids, vecs

def AnnoyIndexBuild(ids, vecs, vec_len):
  t = AnnoyIndex(vec_len, metric='euclidean')
  idx = 0
  for vec in vecs:
    t.add_item(idx, vec)
    idx = idx+ 1
  t.build(100)
  t.save("item_title_vec.ann")


def AnnoyInfer(filename, ids, vec_len):

  u = AnnoyIndex(vec_len, metric='euclidean')
  u.load('item_title_vec.ann') 
  for line in open(filename):
    sp = line.strip().split('\t')
    vec = vec_len * [0]
    idx = 0
    for f in sp[2].strip().split(','):
      vec[idx] = float(f)
      idx = idx + 1

    list, score = u.get_nns_by_vector(vec, 100, 1000, include_distances=True)
    res = []
    j = 0
    length = len(list)
    while (j < length):
      print (sp[0] + "\t" + ids[list[j]] + "\t" + str(score[j]))
      j = j + 1

item_vector = sys.argv[2]
query_vector = sys.argv[1]
length = int(sys.argv[3])

ids, vecs = LoadVector(item_vector, length)
#AnnoyIndexBuild(ids, vecs, length)
AnnoyInfer(query_vector, ids, length)
