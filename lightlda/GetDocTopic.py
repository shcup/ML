#!/usr/bin/env python                                                                                                           
# -*- coding:utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

def cmp(x, y):
  if x[1] < y[1]:
    return -1
  elif x[1] > y[1]:
    return 1

doc_list=[]
for docid in open(sys.argv[1], 'r'):
  doc_list.append(docid.strip())

i=0
for line in sys.stdin:
  topicid_id=[]
  line_array = line.strip().split(' ')
  doc_idx = int(line_array[0])
  topics = line_array[1:]
  for topic in topics:
    if len(topic) == 0:
      continue
    topic_id, count = topic.rsplit(':', 1)
    topicid_id.append([topic_id, int(count)])
  
  #total = sum([int(x[1]) for x in topicid_id]) 
  topicid_id = sorted(topicid_id, key=lambda x:x[1], reverse=True)
  idx = 0
  top = int(topicid_id[0][1])
  while (1):
    if (idx >= len(topicid_id)):
      break;
    value = int(topicid_id[idx][1])
    if (value <= 1 or value / float(top) < 0.2):
      break;
    idx = idx + 1
 
  if (idx <= 1):
    continue
  i = 0
  res = [x[0] + ':' + str(x[1]) for x in topicid_id[:idx]]
  print "212_"+doc_list[doc_idx]+'\tlda\t'+' '.join(res)
