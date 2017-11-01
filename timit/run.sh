#!/bin/bash

#Top script for My experiment
#Author: Richardfan
#2017.11.1

lstm_ctc_CONF_FILE='./setting.conf'
lstm_ctc_LOG_FILE='./log/train_lstm_ctc.log'

echo ========================================================
echo "                     Training                         "
echo ========================================================

python steps/lstm_ctc.py --conf $lstm_ctc_CONF_FILE --log $lstm_ctc_LOG_FILE


echo ========================================================
echo "                 Greedy Decoding                      "
echo ========================================================

python steps/test.py --decode-type 'Greedy'

