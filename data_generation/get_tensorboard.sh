#!/bin/bash
tensorboard --logdir=./logs &
sleep 10

NAME_LIST=$(find ./logs -name *.trace.json.gz)

TB_URL1="http://localhost:6006/data/plugin/profile/data?run="
TB_URL3="/train/"
TB_URL5="&tag=tensorflow_stats&host="
TB_URL7="&tqx=out:csv;"
for NAME in $NAME_LIST
do
        TB_URL2=$(echo $NAME | cut -d "/" -f 3)
        TB_URL4=$(echo $NAME | cut -d "/" -f 7)
        TB_URL6=$(echo $NAME | cut -d "/" -f 8 | cut -d "." -f 1)
        TB_URL="$TB_URL1$TB_URL2$TB_URL3$TB_URL4$TB_URL5$TB_URL6$TB_URL7"
        echo $TB_URL
        FILENAME="./tensorstats/$TB_URL2$TB_URL4.csv"
        curl -o $FILENAME $TB_URL
        sleep 1
done
ps -ef | grep tensorboard | grep -v grep | awk '{print $2}' | xargs kill
