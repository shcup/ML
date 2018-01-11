#!/usr/bin

#odpscmd -e "tunnel download -fd '\t' recallmm_query_demo recallmm_query_demo.txt"
#odpscmd -e "tunnel download -fd '\t' recallmm_item_demo recallmm_item_demo.txt"

python annoy_index.py recallmm_query_demo.txt recallmm_item_demo.txt 128 > relevence.txt

# create table query_table(query varchar(256) PRIMARY KEY NOT NULL, querycate varchar(256));
# create table item_table(id INT PRIMARY KEY NOT NULL, title varchar(256), pic varchar(128), cate varchar(64), subtitle varchar(256));
# create table relevence(query varchar(256) NOT NULL, list varchar(10240), PRIMARY KEY(query));

cat recallmm_query_demo.txt | awk -F'\t' '{print $1"\t"$3}' > query_dump
cat recallmm_item_demo.txt | awk -F'\t' '{print $1"\t"$3"\t"$4"\t"$5"\t"$6}' > item_dump