#!/usr/bin/python
#encoding=utf-8

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

__author__ = "Richardfan"

class SequenceWise(nn.Module):
    def __init__(self, module):
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        '''
        two kinds of inputs: 
            when add cnn, the inputs are regular matrix
            when only use lstm, the inputs are PackedSequence
        '''
        try:
            x, batch_size_len = x.data, x.batch_sizes
            #x.data:    sum(x_len) * num_features
            x = self.module(x)
            x = nn.utils.rnn.PackedSequence(x, batch_size_len)
        except:
            t, n = x.size(0), x.size(1)
            x = x.view(t*n, -1)
            #x :    sum(x_len) * num_features
            x = self.module(x)            
            x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr

class BatchSoftmax(nn.Module):
    '''
    The layer to add softmax for a sequence, which is the output of rnn
    Which state use its own softmax, and concat the result
    '''
    def forward(self, x):
        #x:    seq_len * batch_size * num
        if not self.training:
            seq_len = x.size()[0]
            return torch.stack([F.softmax(x[i], dim=1) for i in range(seq_len)], 0)
        else:
            return x

class BatchRNN(nn.Module):
    """
    Add BatchNorm before rnn to generate a batchrnn layer
    """
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, 
                    bidirectional=False, batch_norm=True, dropout=0.1):
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

class LayerCNN(nn.Module):
    """
    One CNN layer include conv2d, batchnorm, activation and maxpooling
    """
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, pooling_size=None, 
                        activation_function=nn.ReLU, batch_norm=True):
        super(LayerCNN, self).__init__()
        try:
            if len(kernel_size) == 2:
                self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
                self.batch_norm = nn.BatchNorm2d(out_channel) if batch_norm else None
        except:
            self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
            self.batch_norm = nn.BatchNorm1d(out_channel) if batch_norm else None
        self.activation = activation_function(inplace=True)
        if pooling_size is not None:
            try:
                if len(kernel_size) == 2:
                    self.pooling = nn.MaxPool2d(pooling_size)
            except:
                self.pooling = nn.MaxPool1d(pooling_size)
        else:
            self.pooling = None
        
    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = self.activation(x)
        if self.pooling is not None:
            x = self.pooling(x)
        return x

class CTC_Model(nn.Module):
    def __init__(self, rnn_param=None, add_cnn=False, cnn_param=None, num_class=48, drop_out=0.1):
        """
        rnn_param:    the dict of rnn parameters               type dict
            rnn_param = {"rnn_input_size":201, "rnn_hidden_size":256, ....}
        add_cnn  :    whether add cnn in the model             type bool 
        cnn_param:    the cnn parameters, only support Conv2d  type list
            cnn_param = {"layer":[[(in_channel, out_channel), (kernel_size), (stride), (padding), (pooling_size)],...], 
                            "batch_norm":True, "activate_function":nn.ReLU}
        num_class:    the number of units, add one for blank to be the classes to classify
        drop_out :    drop_out paramteter for all place where need drop_out
        """
        super(CTC_Model, self).__init__()
        if rnn_param is None or type(rnn_param) != dict:
            raise ValueError("rnn_param need to be a dict to contain all params of rnn!")
        self.rnn_param = rnn_param
        self.add_cnn = add_cnn
        self.cnn_param = cnn_param
        self.num_class = num_class
        self.num_directions = 2 if rnn_param["bidirectional"] else 1
        self._drop_out = drop_out
        
        if add_cnn:
            cnns = []
            activation = cnn_param["activate_function"]
            batch_norm = cnn_param["batch_norm"]
            rnn_input_size = rnn_param["rnn_input_size"]
            cnn_layers = len(cnn_param["layer"])
            for layer in range(cnn_layers):
                in_channel = cnn_param["layer"][layer][0][0]
                out_channel = cnn_param["layer"][layer][0][1]
                kernel_size = cnn_param["layer"][layer][1]
                stride = cnn_param["layer"][layer][2]
                padding = cnn_param["layer"][layer][3]
                pooling_size = cnn_param["layer"][layer][4]
                
                cnn = LayerCNN(in_channel, out_channel, kernel_size, stride, padding, pooling_size, 
                                activation_function=activation, batch_norm=batch_norm)
                cnns.append(('%d' % layer, cnn))
               
                try:
                    rnn_input_size = int(math.floor((rnn_input_size+2*padding[1]-kernel_size[1])/stride[1])+1)
                except:
                    rnn_input_size = rnn_input_size
            self.conv = nn.Sequential(OrderedDict(cnns))
            #change the input of rnn, adjust the feature length and adjust the seq_len in dataloader
            rnn_input_size *= out_channel
        else:
            rnn_input_size = rnn_param["rnn_input_size"]
        rnns = []
        
        rnn_hidden_size = rnn_param["rnn_hidden_size"]
        rnn_type = rnn_param["rnn_type"]
        rnn_layers = rnn_param["rnn_layers"]
        bidirectional = rnn_param["bidirectional"]
        batch_norm = rnn_param["batch_norm"]
        
        rnn = BatchRNN(input_size=rnn_input_size, hidden_size=rnn_hidden_size, 
                        rnn_type=rnn_type, bidirectional=bidirectional, dropout=drop_out,
                        batch_norm=False)
        
        rnns.append(('0', rnn))
        for i in range(rnn_layers-1):
            rnn = BatchRNN(input_size=self.num_directions*rnn_hidden_size, 
                            hidden_size=rnn_hidden_size, rnn_type=rnn_type, 
                            bidirectional=bidirectional, dropout=drop_out, batch_norm=batch_norm)
            rnns.append(('%d' % (i+1), rnn))

        self.rnns = nn.Sequential(OrderedDict(rnns))

        if batch_norm:
            fc = nn.Sequential(nn.BatchNorm1d(self.num_directions*rnn_hidden_size),
                                nn.Linear(self.num_directions*rnn_hidden_size, num_class+1, bias=False),)
        else:
            fc = nn.Linear(self.num_directions*rnn_hidden_size, num_class+1, bias=False)
        
        self.fc = SequenceWise(fc)
        self.inference_softmax = BatchSoftmax()
    
    def forward(self, x, visualize=False, dev=False):
        #x: batch_size * 1 * max_seq_length * feat_size
        if visualize:
            visual = [x]
        
        if self.add_cnn:
            x = self.conv(x)
            
            if visualize:
                visual.append(x)
            
            x = x.transpose(1, 2).contiguous()
            sizes = x.size()
            if len(sizes) > 3:
                x = x.view(sizes[0], sizes[1], sizes[2]*sizes[3])
            
            x = x.transpose(0,1).contiguous()

            if visualize:
                visual.append(x)

            x = self.rnns(x)
            x = self.fc(x)
            out = self.inference_softmax(x)
            
            if visualize:
                visual.append(out)
                return out, visual
            if dev:
                return x, out
            return out
        else:   
            x = self.rnns(x)
            x = self.fc(x)
            x, batch_seq = nn.utils.rnn.pad_packed_sequence(x, batch_first=False)
            
            out = self.inference_softmax(x)
            if visualize:
                visual.append(out)
                return out, visual
            if dev:
                return x, out
            return out

    @staticmethod
    def save_package(model, optimizer=None, decoder=None, epoch=None, loss_results=None, dev_loss_results=None, dev_cer_results=None):
        package = {
                'rnn_param': model.rnn_param,
                'add_cnn': model.add_cnn,
                'cnn_param': model.cnn_param,
                'num_class': model.num_class,
                '_drop_out': model._drop_out,
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
            package['dev_loss_results'] = dev_loss_results
            package['dev_cer_results'] = dev_cer_results
        return package

if __name__ == '__main__':
    model = CTC_Model(add_cnn=True, cnn_param={"batch_norm":True, "activativate_function":nn.ReLU, "layer":[[(1,32), (3,41), (1,2), (0,0), None],
                            [(32,32), (3,21), (2,2), (0,0), None]]}, num_class=48, drop_out=0)
    for idx, m in CTC_Model.modules():
        print(idx, m) 
    
