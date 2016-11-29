#!/bin/bash
urldir="../../nomatchdata/NF_nomatchdata"
dir="."
 cat $dir/india.nomatch.test.predict | awk -F "\t" '{
  
              if($0 == "__label__1") {print "Football"}
              else if ($0 == "__label__2") {print "Badminton"}
              else if ($0 == "__label__3"){print "Cricket"}
              else if ($0 == "__label__4") {print "Boxing"}
              else if($0 == "__label__5") {print "F1"}
              else if($0 == "__label__6") {print "Hockey"}
              else if($0 == "__label__7") {print "Tennis"}
              else if($0 == "__label__8") {print "Other Sports"}
              else if($0 == "__label__9") {print "Big Events"}
                              }' >$dir/second_category

cat $urldir/sports_data | awk -F "\t" '{print $2}' >$dir/url
cat $urldir/sports_data | awk -F "\t" '{print $3}' >$dir/id

