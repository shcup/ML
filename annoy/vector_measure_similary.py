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
    oid, vec_list = line.strip().split('\t', 1)
    values = vec_list.strip().split(' ')
    vec = vec_len * [0]
    idx = 0
    for f in values:
      vec[idx] = float(f)
      idx = idx + 1
    vecs.append(vec)
    ids.append(oid)
  
  return ids, vecs

def LoadLDA():
  vecs = []
  docs = []
  for line in sys.stdin:
    docid,tab,topic_list = line.strip().split('\t', 2) 
    topics = topic_list.strip().split(' ')
    vec = 1000 * [0]
    for topic in topics:
      topicid,value = topic.split(':')
      vec[int(topicid)] = int(value)
    vecs.append(vec)
    docs.append(docid)
  return docs, vecs

def KNN_Any2Vec(tag, filename, length):
  ids, vecs = LoadVector(filename, length)
  #t = AnnoyIndex(100)
  #idx = 0
  #for vec in vecs:
  #  t.add_item(idx, vec)
  #  idx = idx + 1
  #t.build(100)
  #t.save(tag + '_2vec.ann')

  u = AnnoyIndex(100)
  u.load(tag + '_2vec.ann') 
  idx = 0
  for id in ids:
    list, score = u.get_nns_by_item(idx, 20, 2000, include_distances=True)
    res = []
    j = 0
    length = len(list)
    while (j < length):
      if vecs[list[j]] != id :
        res.append(ids[list[j]] + "," + str(2.0 - score[j]))
      j = j + 1
    idx = idx + 1
    print (id + "\t" + "\t".join(res))

tag = sys.argv[1]
filename = sys.argv[2]
length = int(sys.argv[3])

KNN_Any2Vec(tag, filename, length)



