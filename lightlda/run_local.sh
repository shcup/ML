#!/bin/bash

cd `dirname $0`
source /etc/profile

if [ -f occupy.lock ]
then 
  exit 1
else
  touch occupy.lock
fi

local_forward_index=input/forward_index.dat
local_word_list=input/word.list
local_doc_list=input/doc.list
local_svm=input/train.libsvm
local_word_id_dict=input/word.id.dict
local_output=output

cat $local_forward_index | python ./ForwardIndex2Libsvm.py $local_word_list $local_doc_list $local_word_id_dict > $local_svm

./dump_binary $local_svm  $local_word_id_dict $local_output 0

num_vocabs=`cat $local_word_list | wc -l`
num_document=`cat $local_doc_list | wc -l`
max_num_document=`expr $num_document + 1`
echo 'num_vocabs =' $num_vocabs
echo 'num_document =' $num_document
echo 'max_num_document =' $max_num_document

./lightlda -num_vocabs $num_vocabs -num_topics 1000 -num_iterations 1200 -alpha 0.15 -beta 0.01 -mh_steps 2 -num_local_workers 10 -num_blocks 1 -max_num_document $max_num_document -input_dir $local_output -data_capacity 5000

cat doc_topic.0 | python ./GetDocTopic.py $local_doc_list > docid_topic

cat docid_topic | python ../annoy/annoy_knn.py  LDA > lda_similarity.txt

rm -f occupy.lock
