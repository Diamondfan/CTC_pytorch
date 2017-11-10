#!/bin/bash

#Top script for My experiment
#Author: Richardfan
#2017.11.1

lstm_ctc_CONF_FILE='./conf/lstm_ctc_setting.conf'
cnn_lstm_ctc_CONF_FILE='./conf/cnn_lstm_ctc_setting.conf'
LOG_DIR='./log/'

echo ========================================================
echo "                     Training                         "
echo ========================================================

#python steps/cnn_lstm_ctc.py --conf $cnn_lstm_ctc_CONF_FILE --log-dir $LOG_DIR
python steps/lstm_ctc.py --conf $lstm_ctc_CONF_FILE --log-dir $LOG_DIR


echo ========================================================
echo "                 Greedy Decoding                      "
echo ========================================================

python steps/test.py --conf $lstm_ctc_CONF_FILE --decode-type 'Greedy' --map-48-39 ./decode_map_48-39/map_dict.pkl

