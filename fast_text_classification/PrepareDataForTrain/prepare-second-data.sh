#! /bin/bash

#__label__ ,(title), body

dir="../../F_data/NFSecond_data"
testdir="."
cat $dir/part* | awk -F "\t" '{a[$3]++}END{for(i in a) print i"="a[i]}'

cat $dir/part* | awk -F "\t" '{

            if($3 == "Football,") {print "__label__1 , \t" $0}
            else if ($3 == "Badminton,") {print "__label__2 , \t" $0}
            else if ($3 == "Cricket,"){print "__label__3 , \t" $0}
            else if ($3 == "Boxing,") {print "__label__4 , \t" $0}
            else if($3 == "F1,") {print "__label__5 , \t" $0}
            else if($3 == "Hockey,") {print "__label__6 , \t" $0}
            else if($3 == "Tennis,") {print "__label__7 , \t" $0}
            else if($3 == "Other Sports,") {print "__label__8 , \t" $0}
            else if($3 == "Big Events,") {print "__label__9 , \t" $0}

                            }' >$testdir/traindata

#cat $dir/part* | awk -F "\t" '{if($3 == "Lifestyle") {print "__label__1 , \t" $0}}' >$testdir/lifestyle_data
#cat $dir/part* | awk -F "\t" '{if($3 == "Auto" || $3 == "Autos") {print "__label__2 , \t" $0}}' >$testdir/auto_data
#cat $dir/part* | awk -F "\t" '{if($3 == "Sports" || $3 == "Sport") {print "__label__3 , \t" $0}}'>$testdir/sports_data
#cat $dir/part* | awk -F "\t" '{if($3 == "India") {print "__label__4 , \t" $0}}'>$testdir/india_data
#cat $dir/part* | awk -F "\t" '{if($3 == "Sci-Tech") {print "__label__5 , \t" $0}}'>$testdir/sci-tech_data
#cat $dir/part* | awk -F "\t" '{if($3 == "Business") {print "__label__6 , \t" $0}}'>$testdir/business_data
#cat $dir/part* | awk -F "\t" '{if($3 == "Education") {print "__label__7 , \t" $0}}'>$testdir/education_data
#cat $dir/part* | awk -F "\t" '{if($3 == "Entertainment") {print "__label__8 , \t" $0}}'>$testdir/entertainment_data
#cat $dir/part* | awk -F "\t" '{if($3 == "World") {print "__label__9 , \t" $0}}'>$testdir/world_data

#cat $testdir/*_data  >$testdir/traindata
num=`cat $testdir/traindata |wc -l`
line=`echo "sclae=0; $num/10" | bc`
#sort -R $testdir/traindata > $testdir/trainsort
shuf $testdir/traindata > $testdir/trainsort
split -$line $testdir/trainsort 

cat xaa xab xac xad xae xaf xag xah xaj >$testdir/train_full
cat xai >$testdir/test_full

cat $testdir/train_full | awk -F "\t" '{print $1$5}'>$testdir/india.train
cat $testdir/test_full | awk -F "\t" '{print $1$5}'>$testdir/india.test

cat ../../nomatchdata/sports_data | awk -F "\t" '{print $1",\t"$5}'>$testdir/india.nomatch.test
#cat $dir/part* | awk -F "\t" '{if($3 == "NoMatch"){print "__label__1 , \t" $0}}' > $tetstdir/NoMatchdata
#cat $testdir/NoMatchdata | awk -F "\t" '{print $1$5}'>$testdir/india.nomatch.test
