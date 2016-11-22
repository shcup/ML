#!/bin/bash

cd `dirname $0`
source /etc/profile

tag=
if [ $# -ge 1 ]
then 
   tag=$1
else
  exit 1
fi

if [ -f occupy.lock ]
then 
  exit 1
else
  touch occupy.lock
fi

python doc2vec_gensim.py $tag

cat output_doc2vec.txt.$tag | python ../annoy/annoy_knn.py Doc2Vec > doc2vec_similarity.txt

rm -f occupy.lock
