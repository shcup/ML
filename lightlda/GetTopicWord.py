#!/usr/bin/env python                                                                                                           
# -*- coding:utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

def cmp(x, y):
  if x[1] < y[1]:
    return 1
  elif x[1] > y[1]:
    return -1
  else:
    return 0

word_list=[]
for word in open(sys.argv[1], 'r'):
  word_list.append(word.strip())

topic_dict={}

i=0
for line in sys.stdin:
  line_array = line.strip().split(' ')
  word_idx = int(line_array[0])
  topics = line_array[1:]
  for topic in topics:
    if len(topic) == 0:
      continue
    topic_word, count = topic.rsplit(':', 1)
    if topic_word not in topic_dict:
      topic_dict[topic_word]=[]
    topic_dict[topic_word].append([word_idx, int(count)])
  
for k,v in topic_dict.items():
  v.sort(cmp)
  res=[]
  for kv in v:
    res.append(word_list[kv[0]]+':'+str(kv[1]))
  print k + "\tlda\t" + ' '.join(res) 
    
