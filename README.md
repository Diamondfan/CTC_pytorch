## Update:
Update to pytorch1.2 and python3.

# CTC-based Automatic Speech Recogniton
This is a CTC-based speech recognition system with pytorch.

At present, the system only supports phoneme recognition.  

You can also do it at word-level and may get a high error rate.

Another way is to decode with a lexcion and word-level language model using WFST which is not included in this system.

## Data
English Corpus: Timit
- Training set: 3696 sentences(exclude SA utterance)
- Dev set: 400 sentences
- Test set: 192 sentences

Chinese Corpus: 863 Corpus
- Training set:
  
|  Speaker |          UtterId         |   Utterances  |  
|   :-:    |           :-:            |      :-:      |  
| M50, F50 |   A1-A521, AW1-AW129     | 650 sentences |    
| M54, F54 | B522-B1040,BW130-BW259   | 649 sentences |   
| M60, F60 | C1041-C1560  CW260-CW388 | 649 sentences |   
| M64, F64 |         D1-D625          | 625 sentences |  
|   All    |                          |5146 sentences |   

- Test set:  

|  Speaker |   UtterId   |   Utterances  |  
|   :-:    |     :-:     |      :-:      |
| M51, F51 |   A1-A100   | 100 sentences | 
| M55, F55 |  B522-B521  | 100 sentences | 
| M61, F61 | C1041-C1140 | 100 sentences | 
| M63, F63 |   D1-D100   | 100 sentences | 
|   All    |             | 800 sentences |

## Install
- Install [Pytorch](http://pytorch.org/)
- ~~Install [warp-ctc](https://github.com/SeanNaren/warp-ctc) and bind it to pytorch.~~  
    ~~Notice: If use python2, reinstall the pytorch with source code instead of pip.~~
    Use pytorch1.2 built-in CTC function(nn.CTCLoss) Now.
- Install [Kaldi](https://github.com/kaldi-asr/kaldi). We use kaldi to extract mfcc and fbank.
- Install pytorch [torchaudio](https://github.com/pytorch/audio.git)(This is needed when using waveform as input).
- ~~Install [KenLM](https://github.com/kpu/kenlm). Training n-gram Languange Model if needed~~.
    Use Irstlm in kaldi tools instead.
- Install and start visdom
```
pip3 install visdom
python -m visdom.server
```
- Install other python packages
```
pip install -r requirements.txt
```

## Usage
1. Install all the packages according to the Install part.  
2. Revise the top script run.sh.  
4. Open the config file to revise the super-parameters about everything.  
5. Run the top script with four conditions
```bash
bash run.sh    data_prepare + AM training + LM training + testing
bash run.sh 1  AM training + LM training + testing
bash run.sh 2  LM training + testing
bash run.sh 3  testing
```
RNN LM training is not implemented yet. They are added to the todo-list.  

## Data Prepare
1. Extract 39dim mfcc and 40dim fbank feature from kaldi. 
2. Use compute-cmvn-stats and apply-cmvn with training data to get the global mean and variance and normalize the feature. 
3. Rewrite Dataset and dataLoader in torch.nn.dataset to prepare data for training. You can find them in the steps/dataloader.py.

## Model
- RNN + DNN + CTC 
    RNN here can be replaced by nn.LSTM and nn.GRU
- CNN + RNN + DNN + CTC  
    CNN is use to reduce the variety of spectrum which can be caused by the speaker and environment difference.
- How to choose  
    Use add_cnn to choose one of two models. If add_cnn is True, then CNN+RNN+DNN+CTC will be chosen.

## Training:
- initial-lr = 0.001
- decay = 0.5
- wight-decay = 0.005   

Adjust the learning rate if the dev loss is around a specific loss for ten times.  
Times of adjusting learning rate is 8 which can be alter in steps/train_ctc.py(line367).  
Optimizer is nn.optimizer.Adam with weigth decay 0.005 

## Decoder
### Greedy decoder:
Take the max prob of outputs as the result and get the path.  
Calculate the WER and CER by used the function of the class.
### Beam decoder:
Implemented with python. [Original Code](https://github.com/githubharald/CTCDecoder)  
I fix it to support phoneme for batch decode.    
Beamsearch can improve about 0.2% of phonome accuracy.  
Phoneme-level language model is inserted to beam search decoder now.  

## ToDo
- Combine with RNN-LM  
- Beam search with RNN-LM  
- The code in 863_corpus is a mess. Need arranged.

