#!/usr/bin/python
#encoding=utf-8

from data_prepare.data_loader import myDataset
from data_prepare.data_loader import myDataLoader
from model import *
from ctcDecoder import Decoder, BeamDecoder
import torch
import torch.nn as nn
from torch.autograd import Variable
import time

def test(model_path):
    input_size = 39
    layers = 4
    hidden_size = 256
    decoder_type = 'Greedy'
    labels = "#'acbedgfihkjmlonqpsrutwvyxz_"
    
    model = CTC_RNN(rnn_input_size=input_size, rnn_hidden_size=hidden_size, rnn_layers=layers,
                    rnn_type=nn.LSTM, bidirectional=True, batch_norm=True, num_class=28)
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    if USE_CUDA:
        model = model.cuda()

    if decoder_type == 'Greedy':
        decoder  = Decoder(labels, space_idx=28, blank_index=0)
    else:
        decoder = BeamDecoder(labels, space_idx=28, blank_index=0)

    test_dataset = myDataset(data_set='train', n_mfcc=39)
    test_loader = myDataLoader(test_dataset, batch_size=8, shuffle=False,
                    num_workers=4, pin_memory=False)
    total_wer = 0
    total_cer = 0
    total_tokens = 0
    for data in test_loader:
        inputs, target, input_sizes, input_size_list, target_sizes = data 
        inputs = inputs.transpose(0,1)
        inputs = Variable(inputs, volatile=True, requires_grad=False)
        if USE_CUDA:
            inputs = inputs.cuda()
        
        inputs = nn.utils.rnn.pack_padded_sequence(inputs, input_size_list)
        probs = model(inputs)
        probs = probs.data.cpu()
        
        decoded = decoder.greedy_decoder(probs, input_size_list)
        targets = decoder._unflatten_targets(target, target_sizes)
        labels = decoder._process_strings(decoder._convert_to_strings(targets))
        for x in range(len(labels)):
            print("origin: "+ labels[x])
            print("decoded: "+ decoded[x])
        cer, wer = decoder.phone_word_error(probs, input_size_list, target, target_sizes)
        total_cer += cer
        total_wer += wer
        total_tokens += sum(target_sizes)
    CER = (float(total_cer) / total_tokens)*100
    WER = (float(total_wer) / decoder.num_word)*100
    print("Character error rate on test set: %.4f" % CER)
    print("Word error rate on test set: %.4f" % WER)

if __name__ == "__main__":
    test(model_path = './log/best_model_cv66.8457241083.pkl')
    
