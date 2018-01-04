#!/bin/bash

#Top script for My experiment
#Author: Richardfan
#2017.11.1

stage=0

decode_type='Greedy'
data_set='test'
lm_path='./data_prepare/data/lm_train.text'
CONF_FILE='./conf/ctc_model_setting.conf'
LOG_DIR='./log/'

if [ ! -z $1 ]; then
    stage=$1
fi

if [ $stage -le 0 ]; then
    echo ========================================================
    echo "                     Training                         "
    echo ========================================================

    python steps/ctc_train.py --conf $CONF_FILE --log-dir $LOG_DIR || exit 1;
fi

echo ========================================================
echo "                 Greedy Decoding                      "
echo ========================================================

python steps/test.py --conf $CONF_FILE --decode-type $decode_type --map-48-39 ./decode_map_48-39/map_dict.pkl --data-set $data_set --lm-path $lm_path

