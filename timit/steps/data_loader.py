#!/usr/bin/python
#encoding=utf-8

import os
import sys
import torch
import struct
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils import load_wave, process_label_file, process_map_file, F_Mel

out_map = {"phone":"phn", "char":"wrd"}
audio_conf = {"sample_rate":16000, 'window_size':0.025, 'window_stride':0.01, 'window': 'hamming'}

def read_ark(path):
    path, pos = path.split(':')
    ark_read_buffer = open(path, 'rb')
    ark_read_buffer.seek(int(pos),0)
    header = struct.unpack('<xcccc', ark_read_buffer.read(5))
    #if header[0] != "B":
    #    print("Input .ark file is not binary"); exit(1)

    rows = 0; cols= 0
    m, rows = struct.unpack('<bi', ark_read_buffer.read(5))
    n, cols = struct.unpack('<bi', ark_read_buffer.read(5))

    tmp_mat = np.frombuffer(ark_read_buffer.read(rows * cols * 4), dtype=np.float32)
    utt_mat = np.reshape(tmp_mat, (rows, cols))

    ark_read_buffer.close()        
    return utt_mat

class SpeechDataset(Dataset):
    def __init__(self, data_dir, data_set='train', feature_type='spectrum', out_type='phone', n_feats=39, mel=True):
        self.data_set = data_set
        self.out_type = out_type
        self.feature_type = feature_type
        self.mel = mel
        
        scp_file = os.path.join(data_dir, data_set, feature_type + '.scp')
        label_file = os.path.join(data_dir, data_set, out_map[out_type] + '_text')
        class_file = os.path.join(data_dir, out_type+'_list.txt')
        
        self.class2int, self.int2class = process_map_file(class_file)    
        
        if feature_type == "waveform":
            self.label_dict = process_label_file(label_file, self.out_type, self.class2int)
            self.item = []
            with open(wav_path, 'r') as f:
                for line in f.readlines():
                    utt, path = line.strip().split('\t')
                    self.item.append((path, self.label_dict[utt]))
        else:
            self.process_scp_label(scp_file, label_file)
    
    def process_scp_label(self, scp_file, label_file):
        #read the label file
        label_dict = process_label_file(label_file, self.out_type, self.class2int)
        
        path_dict = {}
        #read the scp file
        with open(scp_file, 'r') as rf:
            for lines in rf.readlines():
                utt, path = lines.strip().split()
                path_dict[utt] = path

        assert len(path_dict) == len(label_dict)

        self.item = []
        for utt in path_dict:
            self.item.append((path_dict[utt], label_dict[utt]))

    def __getitem__(self, idx):
        if self.feature_type == "waveform":
            path, label = self.item[idx]
            return (load_wave(path), label)
        else:
            path, label = self.item[idx]
            feature = torch.FloatTensor(read_ark(path))
            if self.mel:
                return (F_Mel(feature, audio_conf), label)
            else:
                return (feature, label)

    def __len__(self):
        return len(self.item) 

def create_RNN_input(batch):
    def func(p):
        return p[0].size(0)
    
    #sort batch according to the frame nums
    batch = sorted(batch, reverse=True, key=func)
    longest_sample = batch[0][0]
    feat_size = longest_sample.size(1)
    max_length = longest_sample.size(0)
    batch_size = len(batch)
    inputs = torch.zeros(batch_size, max_length, feat_size)
    input_sizes = torch.IntTensor(batch_size)
    target_sizes = torch.IntTensor(batch_size)
    targets = []
    input_size_list = []
    for x in range(batch_size):
        sample = batch[x]
        feature = sample[0]
        label = sample[1]
        seq_length = feature.size(0)
        inputs[x].narrow(0, 0, seq_length).copy_(feature)
        input_sizes[x] = seq_length
        input_size_list.append(seq_length)
        target_sizes[x] = len(label)
        targets.extend(label)
    targets = torch.IntTensor(targets)
    return inputs, targets, input_sizes, input_size_list, target_sizes

def create_CNN_input(batch):
    def func(p):
        return p[0].size(0)
    
    #sort batch according to the frame nums
    batch = sorted(batch, reverse=True, key=func)
    longest_sample = batch[0][0]
    n = longest_sample.size()
    if len(n) > 1:
        feat_size = longest_sample.size(1)
    max_length = longest_sample.size(0)
    batch_size = len(batch)
    if len(n) > 1:
        inputs = torch.zeros(batch_size, 1, max_length, feat_size)
    else:
        inputs = torch.zeros(batch_size, 1, max_length)
    input_percentages = torch.FloatTensor(batch_size)
    target_sizes = torch.IntTensor(batch_size)
    targets = []
    input_percentages_list = []
    for x in range(batch_size):
        sample = batch[x]
        feature = sample[0]
        label = sample[1]
        seq_length = feature.size(0)
        inputs[x][0].narrow(0, 0, seq_length).copy_(feature)
        input_percentages[x] = seq_length / float(max_length)
        input_percentages_list.append(seq_length / float(max_length))
        target_sizes[x] = len(label)
        targets.extend(label)
    targets = torch.IntTensor(targets)
    return inputs, targets, input_percentages, input_percentages_list, target_sizes 

'''
class torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, 
                                                    collate_fn=<function default_collate>, pin_memory=False, drop_last=False)
subclass of DataLoader and rewrite the collate_fn to form batch
'''

class SpeechDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(SpeechDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = create_RNN_input

class SpeechCNNDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(SpeechCNNDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = create_CNN_input

if __name__ == '__main__':
    dev_dataset = SpeechDataset('./data_prepare', data_set='train', feature_type="mfcc", out_type='phone', n_feats=201, mel=False)
    dev_dataloader = SpeechDataLoader(dev_dataset, batch_size=2, shuffle=True)
    
    import visdom
    viz = visdom.Visdom(env='fan')
    for i in range(1):
        show = dev_dataset[i][0].transpose(0, 1)
        text = dev_dataset[i][1]
        for num in range(len(text)):
            text[num] = dev_dataset.int2class[text[num]]
        text = ' '.join(text)
        opts = dict(title=text, xlabel='frame', ylabel='spectrum')
        viz.heatmap(show, opts = opts)
