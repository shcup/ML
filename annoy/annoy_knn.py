#!/usr/bin/env python                                                                                                           
# -*- coding:utf-8 -*-

import sys
from annoy import AnnoyIndex


reload(sys)
sys.setdefaultencoding("utf-8")


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

def LoadDoc2Vec():
  vecs = []
  docs = []
  for line in sys.stdin:
    docid,tag,doc2vec = line.strip().split('\t', 2)
    values = doc2vec.strip().split(' ')
    vec = 100 * [0.0]
    idx = 0
    for value in values:
      vec[idx] = float(value)
      idx = idx + 1
    vecs.append(vec)
    docs.append(docid)
  return docs, vecs


def KNN_LDA():

  docs, vecs = LoadLDA()
  t = AnnoyIndex(1000)
  idx = 0
  for vec in vecs:
    t.add_item(idx, vec)
    idx = idx + 1

  t.build(10) # 1000 trees
  t.save('lda.ann')

  u = AnnoyIndex(1000)
  u.load('lda.ann') # super fast, will just mmap the file
  idx = 0
  for id in docs:
    list,score = u.get_nns_by_item(idx, 20, include_distances=True)
    res = []
    j = 0
    length = len(list)
    while (j < length):
      if docs[list[j]] != id :
        res.append(docs[list[j]] + "," + str(2.0 - score[j]))
      j = j + 1
    idx = idx + 1
    print (id + "\t" + "\t".join(res))

def KNN_Doc2Vec():
  docs, vecs = LoadDoc2Vec()
  t = AnnoyIndex(100)
  idx = 0
  for vec in vecs:
    t.add_item(idx, vec)
    idx = idx + 1
  t.build(100)
  t.save('doc2vec.ann')

  u = AnnoyIndex(100)
  u.load('doc2vec.ann') 
  idx = 0
  for id in docs:
    list, score = u.get_nns_by_item(idx, 20, 10000, include_distances=True)
    res = []
    j = 0
    length = len(list)
    while (j < length):
      if docs[list[j]] != id :
        res.append(docs[list[j]] + "," + str(2.0 - score[j]))
      j = j + 1
    idx = idx + 1
    print (id + "\t" + "\t".join(res))

target = sys.argv[1]
if target == 'LDA':
  KNN_LDA()
elif target == 'Doc2Vec':
  KNN_Doc2Vec()



