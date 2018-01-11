#!/usr/bin

#odpscmd -e "tunnel download -fd '\t' recallmm_query_demo recallmm_query_demo.txt"
#odpscmd -e "tunnel download -fd '\t' recallmm_item_demo recallmm_item_demo.txt"

python annoy_index.py recallmm_query_demo.txt recallmm_item_demo.txt 128 > relevence.txt
