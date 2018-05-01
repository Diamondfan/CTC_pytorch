#!/usr/bin/python
#encoding=utf-8

import time
import torch
import argparse
import ConfigParser
import torch.nn as nn
from torch.autograd import Variable

from ctc_model import *
from ctcDecoder import GreedyDecoder, BeamDecoder
from data_loader import SpeechDataset, SpeechDataLoader, SpeechCNNDataLoader


parser = argparse.ArgumentParser()
parser.add_argument('--conf', help='conf file for training')
parser.add_argument('--model-path', dest='model_path', help='Model file to decode for test')
parser.add_argument('--map-48-39', dest='map_48_39', help='map 48 phones to 39 when test')
parser.add_argument('--lm-path', dest='lm_path', default=None, help='phoneme-level training data for lm')

def test():
    args = parser.parse_args()
    if args.model_path is not None:
        package = torch.load(args.model_path)
        data_dir = '../data_prepare/data'
    else:
        cf = ConfigParser.ConfigParser()
        cf.read(args.conf)
        model_path = cf.get('Model', 'model_file')
        data_dir = cf.get('Data', 'data_dir')
        
        package = torch.load(model_path)
    
    rnn_param = package["rnn_param"]
    add_cnn = package["add_cnn"]
    cnn_param = package["cnn_param"]
    num_class = package["num_class"]
    feature_type = package['epoch']['feature_type']
    n_feats = package['epoch']['n_feats']
    out_type = package['epoch']['out_type']
    drop_out = package['_drop_out']
    try:
        mel = package['epoch']['mel']
    except:
        mel = False

    USE_CUDA = cf.getboolean('Training', 'use_cuda')
    beam_width = cf.getint('Decode', 'beam_width')
    lm_alpha = cf.getfloat('Decode', 'lm_alpha')
    decoder_type =  cf.get('Decode', 'decode_type')
    data_set = cf.get('Decode', 'eval_dataset')

    test_dataset = SpeechDataset(data_dir, data_set=data_set, feature_type=feature_type, out_type=out_type, n_feats=n_feats, mel=mel)
    
    model = CTC_Model(rnn_param=rnn_param, add_cnn=add_cnn, cnn_param=cnn_param, num_class=num_class, drop_out=drop_out)
        
    if add_cnn:
        test_loader = SpeechCNNDataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=False)
    else:
        test_loader = SpeechDataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=False)
    
    model.load_state_dict(package['state_dict'])
    model.eval()
    
    if USE_CUDA:
        model = model.cuda()

    if decoder_type == 'Greedy':
        decoder  = GreedyDecoder(test_dataset.int2class, space_idx=-1, blank_index=0)
    else:
        decoder = BeamDecoder(test_dataset.int2class, beam_width=beam_width, blank_index=0, space_idx=-1, lm_path=args.lm_path, lm_alpha=lm_alpha)    
    
    if args.map_48_39 is not None:
        import pickle
        f = open(args.map_48_39, 'rb')
        map_dict = pickle.load(f)
        f.close()
        print(map_dict)
    
    total_wer = 0
    total_cer = 0
    start = time.time()
    for data in test_loader:
        inputs, target, input_sizes, input_size_list, target_sizes = data 
        if not add_cnn:
            inputs = inputs.transpose(0,1)
        inputs = Variable(inputs, volatile=True, requires_grad=False)
        
        if USE_CUDA:
            inputs = inputs.cuda()
        
        if not add_cnn:
            inputs = nn.utils.rnn.pack_padded_sequence(inputs, input_size_list)
        
        probs = model(inputs)
        if add_cnn:
            max_length = probs.size(0)
            input_size_list = [int(x*max_length) for x in input_size_list]

        probs = probs.data.cpu()
        decoded = decoder.decode(probs, input_size_list)
        targets = decoder._unflatten_targets(target, target_sizes)
        labels = decoder._process_strings(decoder._convert_to_strings(targets))
        if args.map_48_39 is not None:
            for x in range(len(labels)):
                label = labels[x].strip().split(' ')
                for i in range(len(label)):
                    label[i] = map_dict[label[i]]
                labels[x] = ' '.join(label)
                decode = decoded[x].strip().split(' ')
                for i in range(len(decode)):
                    decode[i] = map_dict[decode[i]]
                decoded[x] = ' '.join(decode)

        for x in range(len(labels)):
            print("origin : " + labels[x])
            print("decoded: " + decoded[x])
        cer = 0
        wer = 0
        for x in range(len(labels)):
            cer += decoder.cer(decoded[x], labels[x])
            wer += decoder.wer(decoded[x], labels[x])
            decoder.num_word += len(labels[x].split())
            decoder.num_char += len(labels[x])
        total_cer += cer
        total_wer += wer
    CER = (float(total_cer) / decoder.num_char)*100
    WER = (float(total_wer) / decoder.num_word)*100
    print("Character error rate on %s set: %.4f" % (data_set, CER))
    print("Word error rate on %s set: %.4f" % (data_set, WER))
    end = time.time()
    time_used = (end - start) / 60.0
    print("time used for decode %d sentences: %.4f minutes." % (len(test_dataset), time_used))

if __name__ == "__main__":
    test()
