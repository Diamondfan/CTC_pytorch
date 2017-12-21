#!/bin/bash

#Top script for My experiment
#Author: Richardfan
#2017.11.1

stage=0

decode_type='Greedy'
data_set='test'
lm_path='./data_prepare/data/lm_train.text'
lstm_ctc_CONF_FILE='./conf/lstm_ctc_setting.conf'
cnn_lstm_ctc_CONF_FILE='./conf/cnn_lstm_ctc_setting.conf'
LOG_DIR='./log/'

if [ ! -z $1 ]; then
    stage=$1
fi

if [ $stage -le 0 ]; then
    echo ========================================================
    echo "                     Training                         "
    echo ========================================================

    python steps/cnn_lstm_ctc.py --conf $cnn_lstm_ctc_CONF_FILE --log-dir $LOG_DIR || exit 1;
    #python steps/lstm_ctc.py --conf $lstm_ctc_CONF_FILE --log-dir $LOG_DIR
fi

echo ========================================================
echo "                 Greedy Decoding                      "
echo ========================================================

python steps/test.py --conf $cnn_lstm_ctc_CONF_FILE --decode-type $decode_type --map-48-39 ./decode_map_48-39/map_dict.pkl --data-set $data_set --lm-path $lm_path
#python steps/test.py --conf $lstm_ctc_CONF_FILE --decode-type 'Greedy' --map-48-39 ./decode_map_48-39/map_dict.pkl --data-set 'test'

