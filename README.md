# My system for Speech recogniton
- Start time : 2017.8.24
- Author     : Richardfan
- The END-To-END CTC for speech recognition system using pytorch. Now only phone recognition.

## Data
English Corpus: Timit
- Training set: 3696 sentences
- Dev set: 400 sentences
- Test set: 192 sentences

Chinese Corpus: 863 Corpus
- Training set:  
  M50       F50	    A1-A521      AW1-AW129      650 sentences  
  M54       F54	    B522-B1040   BW130-BW259    649 sentences  
  M60       F60	    C1041-C1560  CW260-CW388    649 sentences  
  M64       F64	    D1-D625                     625 sentences  
  Total:5146 sentences  

- Test set:  
  M51       F51     A1-A100         100 sentences  
  M55       F55     B522-B521       100 sentences  
  M61       F61     C1041-C1140     100 sentences  
  M63       F63     D1-D100         100 sentences  
  Total:800 sentences

## Data Prepare:
1. Extract 39dim mfcc and 40dim fbank feature from kaldi. 
2. Use compute-cmvn-stats and apply-cmvn with training data to get the global mean and variance and normalisz the feature. 
3. Rewrite Dataset and dataLoader in torch.nn.dataset to prepare data for training.
	
## Warpctc-pytorch install:
Notice: If use python2, reinstall the pytorch with source code instead of pip.  
Baidu warp-ctc and binding with pytorch. 
```
https://github.com/SeanNaren/warp-ctc/ 
```
This part is to calculate the CTC-Loss.

## Model part:
- RNN + DNN + CTC :  
    Training process is in the file steps/lstm_ctc.py
- CNN + RNN + DNN + CTC  
    Training process is in the file steps/cnn_lstm_ctc.py

## Training:
- initial-lr = 0.001
- decay = 0.5
- wight-decay = 0.005  
Adjust the learning rate if the dev acc is around a specific acc for ten times.  
Adjust time is set to 8.  
Optimizer is nn.optimizer.Adam with weigth decay 0.001 

## Decoder:
### Greedy decoder:
Take the max prob of outputs as the result and get the path. Calculate the WER and CER by used the function of the class.
### Beam decoder:
Use python implemention of Beamsearch. The basic code is from https://github.com/githubharald/CTCDecoder  
I fix it to support phoneme for batch decode.  
It can add phoneme-level LM which is not finished.  
Beamsearch can improve about 0.2% of phonome accuracy.  

## Test:
The same as training part with decoder but the CTC loss

## ToDo:
Get better model for speech recognition  
Combine with LM  
Beam search with word-level LM  
