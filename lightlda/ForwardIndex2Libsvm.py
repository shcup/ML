#!/usr/bin/env python                                                                                                           
# -*- coding:utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

doclist=[]
doclist_idx={}
docidx=0

wordlist=[]
wordlist_idx={}
word_count={}
wordidx=0

for line in sys.stdin:
  document_forward = line.strip().split('\t', 2)
  if (len(document_forward) == 3):
    docid, wordcount, words = document_forward
  else:
    continue
  
  cur_docidx=0
  if docid in doclist_idx:
    cur_docidx = doclist_idx[docid]
  else:
    doclist.append(docid)
    doclist_idx[docid]=docidx
    cur_docidx = docidx
    docidx += 1
    
  items = words.strip(',').split(',')
  doc_words = []
  for item in items:
    key,split,value = item.rpartition(':')
    if (key == '\t' or key == '|' or key == '.'):
        continue
    
    
    if key in wordlist_idx:
      doc_words.append(str(wordlist_idx[key])+':'+value)
      word_count[key] += int(value)
    else:
      wordlist.append(key)
      wordlist_idx[key]=wordidx
      word_count[key] = int(value)
      doc_words.append(str(wordidx)+':'+value);
      wordidx += 1

  #print doc_words
  print str(cur_docidx) + "\t" + ' '.join(doc_words)


word_list_f = open(sys.argv[1], 'w')
doc_list_f = open(sys.argv[2], 'w')
word_id_dict_f = open(sys.argv[3], 'w')

for doc in doclist:
  doc_list_f.write(doc+'\n')

i=0
for word in wordlist:
  word_list_f.write(word+'\n')
  word_id_dict_f.write('\t'.join([str(i),word,str(word_count[word])])+'\n')
  i += 1


