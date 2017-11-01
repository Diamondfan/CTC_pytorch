#!/usr/bin/python
#encoding=utf-8

from data_loader import myDataset
from data_loader import myDataLoader, myCNNDataLoader
from model import *
from ctcDecoder import GreedyDecoder, BeamDecoder
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--decode-type', dest='decode_type', default='Greedy', help='Decoder for test. GreadyDecoder or Beam search Decoder')

def test(model_path):
    args = parser.parse_args()
    package = torch.load(model_path)
    input_size = package['input_size']
    layers = package['rnn_layers']
    hidden_size = package['hidden_size']
    rnn_type = package['rnn_type']
    num_class = package["num_class"]
    feature_type = package['epoch']['feature_type']
    n_feats = package['epoch']['n_feats']
    out_type = package['epoch']['out_type']
    model_type = package['name']
    drop_out = package['_drop_out']
    #weight_decay = package['epoch']['weight_decay']
    #print(weight_decay)

    decoder_type =  args.decode_type

    test_dataset = myDataset(data_set='test', feature_type=feature_type, out_type=out_type, n_feats=n_feats)
    
    if model_type == 'CNN_LSTM_CTC':
        model = CNN_LSTM_CTC(rnn_input_size=input_size, rnn_hidden_size=hidden_size, rnn_layers=layers, 
                    rnn_type=rnn_type, bidirectional=True, batch_norm=True, num_class=num_class, drop_out=drop_out)
        test_loader = myCNNDataLoader(test_dataset, batch_size=8, shuffle=False,
                    num_workers=4, pin_memory=False)
    else:
        model = CTC_RNN(rnn_input_size=input_size, rnn_hidden_size=hidden_size, rnn_layers=layers,
                    rnn_type=rnn_type, bidirectional=True, batch_norm=True, num_class=num_class, drop_out=drop_out)
        test_loader = myDataLoader(test_dataset, batch_size=8, shuffle=False,
                    num_workers=4, pin_memory=False)
    
    model.load_state_dict(package['state_dict'])
    model.eval()
    
    if USE_CUDA:
        model = model.cuda()

    if decoder_type == 'Greedy':
        decoder  = GreedyDecoder(test_dataset.int2phone, space_idx=-1, blank_index=0)
    else:
        decoder = BeamDecoder(test_dataset.int2phone, top_paths=1, beam_width=20, blank_index=0, space_idx=-1,
                                lm_path='./data_prepare/bigram.binary', dict_path='./data_prepare/dict.txt', 
                                trie_path='./data_prepare/trie', lm_alpha=10, lm_beta1=1, lm_beta2=1)    
    total_wer = 0
    total_cer = 0
    total_tokens = 0
    for data in test_loader:
        inputs, target, input_sizes, input_size_list, target_sizes = data 
        if model.name == 'CTC_RNN':
            inputs = inputs.transpose(0,1)
        inputs = Variable(inputs, volatile=True, requires_grad=False)
        if USE_CUDA:
            inputs = inputs.cuda()
        
        if model.name == 'CTC_RNN':
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
    test('./log/exp_4lstm_256hidden_5lepoch_fbank40/exp2_77.3112/best_model_cv79.165836488.pkl')
    #test('./log/best_model_cv77.3925748821.pkl')
