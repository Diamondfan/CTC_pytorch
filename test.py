#!/usr/bin/python
#encoding=utf-8

from data_prepare.data_loader import myDataset
from data_prepare.data_loader import myDataLoader
from model import *
from ctcDecoder import GreedyDecoder, BeamDecoder
import torch
import torch.nn as nn
from torch.autograd import Variable
import time

def test(model_path):
    package = torch.load(model_path)

    input_size = package['input_size']
    layers = package['rnn_layers']
    hidden_size = package['hidden_size']
    rnn_type = package['rnn_type']
    num_class = package["num_class"]
    feature_type = package['feature_type']
    n_feats = package['n_feats']
    out_type = package['out_type']

    decoder_type = 'Greedy' 

    model = CTC_RNN(rnn_input_size=input_size, rnn_hidden_size=hidden_size, rnn_layers=layers,
                    rnn_type=rnn_type, bidirectional=True, batch_norm=True, num_class=num_class)
    
    model.load_state_dict(package['state_dict'])
    model.eval()
    
    if USE_CUDA:
        model = model.cuda()

    test_dataset = myDataset(data_set='test', feature_type=feature_type, out_type=out_type, n_feats=n_feats)
    test_loader = myDataLoader(test_dataset, batch_size=8, shuffle=False,
                    num_workers=4, pin_memory=False)
    
    if decoder_type == 'Greedy':
        decoder  = GreedyDecoder(test_dataset.int2phone, space_idx=-1, blank_index=0)
    else:
        decoder = BeamDecoder(test_dataset.int2phone, top_paths=2, beam_width=20, space_idx=-1, blank_index=0, 
                                lm_path='./bigram.binary', dict_path='./dict.txt', 
                                trie_path='./trie', lm_alpha=4, lm_beta1=1, lm_beta2=1)

    
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
        
        decoded = decoder.decode(probs, input_size_list)
        targets = decoder._unflatten_targets(target, target_sizes)
        labels = decoder._process_strings(decoder._convert_to_strings(targets))
        for x in range(len(labels)):
            print("origin: "+ labels[x])
            print("decoded: "+ decoded[x])
        cer, wer = decoder.phone_word_error(probs, input_size_list, target, target_sizes)
        total_cer += cer
        total_wer += wer
    CER = (1 - float(total_cer) / decoder.num_char)*100
    WER = (1 - float(total_wer) / decoder.num_word)*100
    print("Character error rate on test set: %.4f" % CER)
    print("Word error rate on test set: %.4f" % WER)

if __name__ == "__main__":
    test('./log/exp_4lstm_256hidden_5lepoch_mfcc/best_model_cv78.8005578801.pkl')
    
