#!/bin/bash
urldir="../second_train"
dir="."
 cat $dir/india.test.predict | awk -F "\t" '{
  
              if($0 == "__label__1") {print "Lifestyle"}
              else if ($0 == "__label__2") {print "Auto"}
              else if ($0 == "__label__3"){print "Sports"}
              else if ($0 == "__label__4") {print "India"}
              else if($0 == "__label__5") {print "Sci-Tech"}
              else if($0 == "__label__6") {print "Business"}
              else if($0 == "__label__7") {print "Education"}
              else if($0 == "__label__8") {print "Entertainment"}
              else if($0 == "__label__9") {print "World"}
                              }' >category

cat $urldir/NoMatchdata | awk -F "\t" '{print $2}' >$dir/url
