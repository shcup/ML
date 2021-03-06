#!/usr/bin/env bash
#!/usr/bin/env bash
#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

myshuf() {
  perl -MList::Util=shuffle -e 'print shuffle(<>);' "$@";
}

normalize_text() {
  tr '[:upper:]' '[:lower:]' | sed -e 's/^/__label__/g' | \
    sed -e "s/'/ ' /g" -e 's/"//g' -e 's/\./ \. /g' -e 's/<br \/>/ /g' \
        -e 's/,/ , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' \
        -e 's/\?/ \? /g' -e 's/\;/ /g' -e 's/\:/ /g' | tr -s " " | myshuf
}

RESULTDIR=F_result/OCFSecond_result
DATADIR=F_train/OCFSecond_train

mkdir -p "${RESULTDIR}"
mkdir -p "${DATADIR}"

#if [ ! -f "${DATADIR}/india.train" ]
#then
#  wget -c "https://github.com/le-scientifique/torchDatasets/raw/master/dbpedia_csv.tar.gz" -O "${DATADIR}/dbpedia_csv.tar.gz"
#  tar -xzvf "${DATADIR}/dbpedia_csv.tar.gz" -C "${DATADIR}"
#  cat "${DATADIR}/dbpedia_csv/train.csv" | normalize_text > "${DATADIR}/dbpedia.train"
#  cat "${DATADIR}/dbpedia_csv/test.csv" | normalize_text > "${DATADIR}/dbpedia.test"
#fi

make
#./fasttext supervised -input "${DATADIR}/india.train" -output "${RESULTDIR}/india" -dim 10 -lr 0.1 -wordNgrams 3 -minCount 1 -bucket 10000000 -epoch 5 -thread 4

#./fasttext test "${RESULTDIR}/india.bin" "${DATADIR}/india.test"
./fasttext test "${RESULTDIR}/india.bin" "${DATADIR}/india.nomatch.test"
#./fasttext predict "${RESULTDIR}/india.bin" "${DATADIR}/india.test" > "${RESULTDIR}/india.test.predict"
./fasttext predict "${RESULTDIR}/india.bin" "${DATADIR}/india.nomatch.test" > "${RESULTDIR}/india.nomatch.test.predict"
