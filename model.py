#!/usr/bin/python
#encoding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

support_rnn = {'lstm': nn.LSTM, 'rnn': nn.RNN, 'gru': nn.GRU}
USE_CUDA = True

class SequenceWise(nn.Module):
    def __init__(self, module):
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        x, batch_size_len = x.data, x.batch_sizes
        #x.data:    sum(x_len) * num_features
        x = self.module(x)
        x = nn.utils.rnn.PackedSequence(x, batch_size_len)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr

class InferenceBatchLogSoftmax(nn.Module):
    def forward(self, x):
        x, batch_seq = nn.utils.rnn.pad_packed_sequence(x,batch_first=False)
        #x:    seq_len * batch_size * num

        if not self.training:
            batch_size = x.size()[0]
            return torch.stack([F.log_softmax(x[i]) for i in range(batch_size)], 0)
        else:
            return x

class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, 
            bidirectional=False, batch_norm=True):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=False)
        
    def forward(self, x):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x, _ = self.rnn(x)
        self.rnn.flatten_parameters()
        return x
    

class CTC_RNN(nn.Module):
    def __init__(self, rnn_input_size=40, rnn_hidden_size=768, rnn_layers=5,
            rnn_type=nn.LSTM, bidirectional=True, 
            batch_norm=True, num_class=28):
        super(CTC_RNN, self).__init__()
        self.rnn_input_size = rnn_input_size
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_layers = rnn_layers
        self.rnn_type = rnn_type
        self.num_directions = 2 if bidirectional else 1
        
        rnns = []
        rnn = BatchRNN(input_size=rnn_input_size, hidden_size=rnn_hidden_size, 
                        rnn_type=rnn_type, bidirectional=bidirectional, 
                        batch_norm=False)
        rnns.append(('0', rnn))
        for i in range(rnn_layers-1):
            rnn = BatchRNN(input_size=self.num_directions*rnn_hidden_size, 
                    hidden_size=rnn_hidden_size, rnn_type=rnn_type, 
                    bidirectional=bidirectional, batch_norm = batch_norm)
            rnns.append(('%d' % (i+1), rnn))

        self.rnns = nn.Sequential(OrderedDict(rnns))

        if batch_norm :
            fc = nn.Sequential(nn.BatchNorm1d(self.num_directions*rnn_hidden_size),
                        nn.Linear(self.num_directions*rnn_hidden_size, num_class+1, bias=False))
        else:
            fc = nn.Linear(self.num_directions*rnn_hidden_size, num_class+1, bias=False)
        
        self.fc = SequenceWise(fc)
        self.inference_log_softmax = InferenceBatchLogSoftmax()
    
    def forward(self, x):
        #x: packed padded sequence
        #x.data:           means the origin data
        #x.batch_sizes:    the batch_size of each frames
        #x_len:            type:list not torch.IntTensor
        #print(x)
        x = self.rnns(x)
        #print(x)
        x = self.fc(x)

        x = self.inference_log_softmax(x)

        return x

