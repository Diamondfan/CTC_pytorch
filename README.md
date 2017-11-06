# My system for Speech recogniton
Start time : 2017.8.24
Author     : Richardfan
The END-To-END CTC for speech recognition system using pytorch. Now only phone recognition.

## Data
English Corpus: Timit
- Training set: 3696 sentences
- Dev set: 400 sentences
- Test set: 192 sentences

Chinese Corpus: 863 Corpus
- Training set:
  M50		F50		A1-A521
				AW1-AW129		650 sentences
  M54		F54		B522-B1040
				BW130-BW259		649 sentences
  M60		F60		C1041-C1560
				CW260-CW388  		649 sentences
  M64		F64		D1-D625         	625 sentences
  Total:5146 sentences

- Test set:
  M51		F51		A1-A100			100 sentences
  M55		F55		B522-B521		100 sentences
  M61		F61		C1041-C1140		100 sentences
  M63		F63		D1-D100         	100 sentences
  Total:800 sentences

## Data Prepare:
Extract 39dim mfcc and 40dim fbank feature from kaldi. Use compute-cmvn-stats and apply-cmvn with training data to get the global mean and variance and normalisz the feature. Rewrite Dataset and dataLoader in torch.nn.dataset to prepare data for training.
	
## Warpctc-pytorch install:
Notice: If use python2, reinstall the pytorch with source code instead of pip. Baidu warp-ctc and binding with pytorch. 
https://github.com/SeanNaren/warp-ctc/ This part is to calculate the CTC-Loss.

## Model part:
1 RNN + Batch + N RNN + Batch + 1 Linear , defined with class CTC-RNN. Add space ' and blank to 26 characters, so 29 categories. When the label is phone, in TIMIT, there are 48 labels and 66 lables in 863 Chinese corpus. Hidden-size for LSTM is 256  which can be set.
Update: 10.26
Add CNN before RNN:
USE (11, 21)kernel, stride=(2, 2)
The first stide 2 can decrease the frame for the input for RNN, and training become faste and with the same result

## Training:
- initial-lr = 0.001
- decay = 0.5
- wight-decay = 0.005
Adjust the learning rate if the dev acc doesn't increase.
Optimizer is nn.optimizer.Adam with weigth decay 0.0001

## Decoder:
### Greedy decoder:
Take the max prob of outputs as the result and get the path. Calculate the WER and CER by used the function of the class.
### Beam decoder:
python 2 has a problem with package pytorch-ctc and not solved. The package is used for beam search decode
https://github.com/ryanleary/pytorch-ctc
It can add LM with decode. With no LM, the CER can increase 0.2%
No test with LM.

## Test:
The same as training part with decoder but the CTC loss

## ToDo:
data preparation for 863 corpus, little different from English one
combine with LM

