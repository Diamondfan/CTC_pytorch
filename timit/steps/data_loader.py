#!/usr/bin/python
#encoding=utf-8

import os
import h5py
import numpy as np
import torch
import sys
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import scipy.signal
import math
from utils import parse_audio, process_kaldi_feat, process_label_file, process_map_file, F_Mel

windows = {'hamming':scipy.signal.hamming, 'hann':scipy.signal.hann, 'blackman':scipy.signal.blackman,
            'bartlett':scipy.signal.bartlett}
audio_conf = {"sample_rate":16000, 'window_size':0.025, 'window_stride':0.01, 'window': 'hamming'}


#Override the class of Dataset
#Define my own dataset over timit used the feature extracted by kaldi
class myDataset(Dataset):
    def __init__(self, data_dir, data_set='train', feature_type='spectrum', out_type='phone', n_feats=39, normalize=True, mel=False):
        self.data_set = data_set
        self.out_type = out_type
        self.feature_type = feature_type
        self.normalize = normalize
        self.mel = mel
        h5_file = os.path.join(data_dir, feature_type+'_'+out_type+'_tmp', data_set+'.h5py')
        wav_path = os.path.join(data_dir, 'wav_path', data_set+'.wav.scp')
        mfcc_file = os.path.join(data_dir, "feature_"+feature_type, data_set+'.txt')
        label_file = os.path.join(data_dir,"label_"+out_type, data_set+'.text')
        char_file = os.path.join(data_dir, out_type+'_list.txt')
        if not os.path.exists(h5_file):
            if feature_type != 'spectrum':
                self.n_feats = n_feats
                print("Process %s data in kaldi format..." % data_set)
                self.process_txt(mfcc_file, label_file, char_file, h5_file)
            else:
                print("Extract spectrum with librosa...")
                self.n_feats = int(audio_conf['sample_rate']*audio_conf['window_size']/2+1)
                self.process_audio(wav_path, label_file, char_file, h5_file)
        else:
            if feature_type != "spectrum":
                self.n_feats = n_feats
            else:
                self.n_feats = int(audio_conf["sample_rate"]*audio_conf["window_size"]/2+1)
                #self.n_feats = n_feats
            print("Loading %s data from h5py file..." % data_set)
            self.load_h5py(h5_file)
    
    def process_txt(self, mfcc_file, label_file, char_file, h5_file):
        #read map file
        self.char_map, self.int2phone = process_map_file(char_file)
        
        #read the label file
        label_dict = process_label_file(label_file, self.out_type, self.char_map)
        
        #read the mfcc file
        mfcc_dict = process_kaldi_feat(mfcc_file, self.n_feats)
        
        if len(mfcc_dict) != len(label_dict):
            print("%s data: The num of wav and text are not the same!" % self.data_set)
            sys.exit(1)

        self.features_label = []
        #save the data as h5 file
        f = h5py.File(h5_file, 'w')
        f.create_dataset("phone_map_key", data=self.char_map.keys())
        f.create_dataset("phone_map_value", data = self.char_map.values())
        for utt in mfcc_dict:
            grp = f.create_group(utt)
            self.features_label.append((torch.FloatTensor(np.array(mfcc_dict[utt])), label_dict[utt].tolist()))
            grp.create_dataset('data', data=np.array(mfcc_dict[utt]))
            grp.create_dataset('label', data=label_dict[utt])
        print("Saved the %s data to h5py file" % self.data_set)
            
    def process_audio(self, wav_path, label_file, char_file, h5_file):
        #read map file
        self.char_map, self.int2phone = process_map_file(char_file)
        
        #read the label file
        label_dict = process_label_file(label_file, self.out_type, self.char_map)
        
        #extract spectrum
        spec_dict = dict()
        f = open(wav_path, 'r')
        for line in f.readlines():
            utt, path = line.strip().split()
            spect = parse_audio(path, audio_conf, windows, normalize=self.normalize)
            #print(spect)
            spec_dict[utt] = spect.numpy()
        f.close()
        
        assert len(spec_dict) == len(label_dict)
        
        self.features_label = []
        #save the data as h5 file
        f = h5py.File(h5_file, 'w')
        f.create_dataset("phone_map_key", data=self.char_map.keys())
        f.create_dataset("phone_map_value", data = self.char_map.values())
        for utt in spec_dict:
            grp = f.create_group(utt)
            self.features_label.append((torch.FloatTensor(spec_dict[utt]), label_dict[utt].tolist()))
            grp.create_dataset('data', data=spec_dict[utt])
            grp.create_dataset('label', data=label_dict[utt])
        print("Saved the %s data to h5py file" % self.data_set)

    def load_h5py(self, h5_file):
        self.features_label = []
        f = h5py.File(h5_file, 'r')
        for grp in f:
            if grp != 'phone_map_key' and grp != 'phone_map_value':
                self.features_label.append((torch.FloatTensor(np.asarray(f[grp]['data'])), np.asarray(f[grp]['label']).tolist()))
        self.char_map = dict()
        self.int2phone = dict()
        keys = f['phone_map_key']
        values = f['phone_map_value']
        for i in range(len(keys)):
            self.char_map[str(keys[i])] = values[i]
            self.int2phone[values[i]] = keys[i]
        self.int2phone[0]='#'
        print("Load %d sentences from %s dataset" % (self.__len__(), self.data_set))

    def __getitem__(self, idx):
        if self.mel:
            spect, label = self.features_label[idx]
            return (F_Mel(spect, audio_conf), label)
        else:
            return self.features_label[idx]

    def __len__(self):
        return len(self.features_label) 

def create_RNN_input(batch):
    def func(p):
        return p[0].size(0)
    
    #sort batch according to the frame nums
    batch = sorted(batch, reverse=True, key=func)
    longest_sample = batch[0][0]
    feat_size = longest_sample.size(1)
    #feat_size = 101
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
        #feature = sample[0].transpose(0,1)[:101].transpose(0,1)
        label = sample[1]
        seq_length = feature.size(0)
        inputs[x].narrow(0, 0, seq_length).copy_(feature)
        input_sizes[x] = seq_length
        input_size_list.append(seq_length)
        target_sizes[x] = len(label)
        targets.extend(label)
    targets = torch.IntTensor(targets)
    #src_pos = [[(pos+1) if (w!=[0]*feat_size).any() else 0 for pos, w in enumerate(instance)] for instance in inputs.numpy()]
    #src_pos = torch.LongTensor(np.array(src_pos))
    return inputs, targets, input_sizes, input_size_list, target_sizes

'''
class torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, 
                                        sampler=None, batch_sampler=None, num_workers=0, 
                                        collate_fn=<function default_collate>, 
                                        pin_memory=False, drop_last=False)
subclass of DataLoader and rewrite the collate_fn to form batch
'''

class myDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(myDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = create_RNN_input

class myCNNDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(myCNNDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = create_CNN_input

def create_CNN_input(batch):
    def func(p):
        return p[0].size(0)
    
    #sort batch according to the frame nums
    batch = sorted(batch, reverse=True, key=func)
    longest_sample = batch[0][0]
    feat_size = longest_sample.size(1)
    max_length = longest_sample.size(0)
    batch_size = len(batch)
    inputs = torch.zeros(batch_size, 1, max_length, feat_size)
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

if __name__ == '__main__':
    dev_dataset = myDataset('../data_prepare/data', data_set='train', feature_type="spectrum", out_type='phone', n_feats=201, mel=True)
    #dev_dataloader = myDataLoader(dev_dataset, batch_size=2, shuffle=True)
    
    import visdom
    viz = visdom.Visdom(env='fan')
    for i in range(1):
        show = dev_dataset[i][0].transpose(0, 1)
        text = dev_dataset[i][1]
        for num in range(len(text)):
            text[num] = dev_dataset.int2phone[text[num]]
        text = ' '.join(text)
        opts = dict(title=text, xlabel='frame', ylabel='spectrum')
        viz.heatmap(show, opts = opts)
