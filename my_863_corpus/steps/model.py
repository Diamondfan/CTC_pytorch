#!/usr/bin/python
#encoding=utf-8

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

__author__ = "Richardfan"

support_rnn = {'lstm': nn.LSTM, 'rnn': nn.RNN, 'gru': nn.GRU}
USE_CUDA = True

def position_encoding_init(n_position, d_pos_vec):
    position_enc = np.array([
        [pos / np.power(10000, 2*i / d_pos_vec) for i in range(d_pos_vec)]
        if pos !=0 else np.zeros(d_pos_vec) for pos in range(n_position)]
        )

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  #dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  #dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)

class Encoder(nn.Module):
    def __init__(self, n_position, d_word_vec=512):
        super(Encoder, self).__init__()
        self.position_enc = nn.Embedding(n_position, d_word_vec, padding_idx=0)
        self.position_enc.weight.data = position_encoding_init(n_position, d_word_vec)

    def forward(self, src_pos):
        enc_input = self.position_enc(src_pos)

        return enc_input

class SequenceWise(nn.Module):
    def __init__(self, module):
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        try:
            x, batch_size_len = x.data, x.batch_sizes
            #print(x)
            #x.data:    sum(x_len) * num_features
            x = self.module(x)
            x = nn.utils.rnn.PackedSequence(x, batch_size_len)
        except:
            t, n = x.size(0), x.size(1)
            x = x.view(t*n, -1)
            #print(x)
            #x :    sum(x_len) * num_features
            x = self.module(x)
            x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr

class InferenceBatchLogSoftmax(nn.Module):
    def forward(self, x):
        #x:    seq_len * batch_size * num

        if not self.training:
            seq_len = x.size()[0]
            return torch.stack([F.log_softmax(x[i]) for i in range(seq_len)], 0)
        else:
            return x

class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, 
            bidirectional=False, batch_norm=True, dropout = 0.1):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, dropout = dropout, bias=False)
        
    def forward(self, x):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x, _ = self.rnn(x)
        self.rnn.flatten_parameters()
        return x
    

class CTC_RNN(nn.Module):
    def __init__(self, rnn_input_size=40, rnn_hidden_size=768, rnn_layers=5,
            rnn_type=nn.LSTM, bidirectional=True, 
            batch_norm=True, num_class=28, drop_out = 0.1):
        super(CTC_RNN, self).__init__()
        self.rnn_input_size = rnn_input_size
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_layers = rnn_layers
        self.rnn_type = rnn_type
        self.num_class = num_class
        self.num_directions = 2 if bidirectional else 1
        self.name = 'CTC_RNN'
        self._drop_out = drop_out
        
        rnns = []
        rnn = BatchRNN(input_size=rnn_input_size, hidden_size=rnn_hidden_size, 
                        rnn_type=rnn_type, bidirectional=bidirectional, 
                        batch_norm=False)
        rnns.append(('0', rnn))
        for i in range(rnn_layers-1):
            rnn = BatchRNN(input_size=self.num_directions*rnn_hidden_size, 
                    hidden_size=rnn_hidden_size, rnn_type=rnn_type, 
                    bidirectional=bidirectional, dropout=drop_out, batch_norm=batch_norm)
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
        x = self.rnns(x)
        #print(x)
        x = self.fc(x)
        
        x, batch_seq = nn.utils.rnn.pad_packed_sequence(x,batch_first=False)
        x = self.inference_log_softmax(x)

        return x

    @staticmethod
    def save_package(model, optimizer=None, decoder=None, epoch=None, loss_results=None, training_cer_results=None, dev_cer_results=None):
        package = {
                'input_size': model.rnn_input_size,
                'hidden_size': model.rnn_hidden_size,
                'rnn_layers': model.rnn_layers,
                'rnn_type': model.rnn_type,
                'num_class': model.num_class,
                'bidirectional': model.num_directions,
                '_drop_out' : model._drop_out,
                'name': model.name,
                'state_dict': model.state_dict()
                }
        if optimizer is not None:
            package['optim_dict'] = optimizer.state_dict()
        if decoder is not None:
            package['decoder'] = decoder
        if epoch is not None:
            package['epoch'] = epoch
        if loss_results is not None:
            package['loss_results'] = loss_results
            package['training_cer_results'] = training_cer_results
            package['dev_cer_results'] = dev_cer_results
        return package

class CNN_LSTM_CTC(nn.Module):
    def __init__(self, rnn_input_size=201, rnn_hidden_size=256, rnn_layers=4,
                    rnn_type=nn.LSTM, bidirectional=True, 
                    batch_norm=True, num_class=48, drop_out=0.1):
        super(CNN_LSTM_CTC, self).__init__()
        self.rnn_input_size = rnn_input_size
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_layers = rnn_layers
        self.rnn_type = rnn_type
        self.num_class = num_class
        self.num_directions = 2 if bidirectional else 1
        self._drop_out = drop_out
        self.name = 'CNN_LSTM_CTC'
        
        self.conv = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(11, 5), stride=(2, 2)),
                nn.BatchNorm2d(16),
                nn.Hardtanh(0, 20, inplace=True),
                #nn.Conv2d(32, 32, kernel_size=(11, 21), stride=(1, 2)),
                #nn.BatchNorm2d(32),
                #nn.Hardtanh(0, 20, inplace=True)
                )
        
        rnn_input_size = int(math.floor(rnn_input_size-5)/2+1)
        #rnn_input_size = int(math.floor(rnn_input_size-21)/2+1)
        rnn_input_size *= 16

        rnns = []
        rnn = BatchRNN(input_size=rnn_input_size, hidden_size=rnn_hidden_size, 
                        rnn_type=rnn_type, bidirectional=bidirectional, 
                        batch_norm=False)
        
        rnns.append(('0', rnn))
        for i in range(rnn_layers-1):
            rnn = BatchRNN(input_size=self.num_directions*rnn_hidden_size, 
                    hidden_size=rnn_hidden_size, rnn_type=rnn_type, 
                    bidirectional=bidirectional, dropout = drop_out, batch_norm = batch_norm)
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
        #x: batch_size * 1 * max_seq_length * feat_size
        x = self.conv(x)
        x = x.transpose(2, 3).contiguous()
        sizes = x.size()

        x = x.view(sizes[0], sizes[1]*sizes[2], sizes[3])
        x = x.transpose(1,2).transpose(0,1).contiguous()
        
        x = self.rnns(x)
        #print(x)
        
        x = self.fc(x)

        x = self.inference_log_softmax(x)

        return x

    @staticmethod
    def save_package(model, optimizer=None, decoder=None, epoch=None, loss_results=None, training_cer_results=None, dev_cer_results=None):
        package = {
                'input_size': model.rnn_input_size,
                'hidden_size': model.rnn_hidden_size,
                'rnn_layers': model.rnn_layers,
                'rnn_type': model.rnn_type,
                'num_class': model.num_class,
                'bidirectional': model.num_directions,
                '_drop_out': model._drop_out,
                'name': model.name,
                'state_dict': model.state_dict()
                }
        if optimizer is not None:
            package['optim_dict'] = optimizer.state_dict()
        if decoder is not None:
            package['decoder'] = decoder
        if epoch is not None:
            package['epoch'] = epoch
        if loss_results is not None:
            package['loss_results'] = loss_results
            package['training_cer_results'] = training_cer_results
            package['dev_cer_results'] = dev_cer_results
        return package

def xavier_uniform_init(m):
    for param in m.parameters():
        if param.data.ndimension() > 1:
            nn.init.xavier_uniform(param.data)

if __name__ == '__main__':
    #model = CNN_LSTM_CTC(rnn_input_size=201, rnn_hidden_size=256, rnn_layers=4, 
    #                    rnn_type=nn.LSTM, bidirectional=True, batch_norm=True, 
    #                    num_class=48, drop_out=0)
    #model.apply(xavier_uniform_init) 
    encoder = Encoder(11, 20)
    src_pos = [[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,0,0,0,0,0]]
    src_pos = torch.LongTensor(src_pos)
    src_pos = torch.autograd.Variable(src_pos)
    print(src_pos)
    out = encoder(src_pos)
    print(out)

