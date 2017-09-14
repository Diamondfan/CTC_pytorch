#!/usr/bin/python
#encoding=utf-8

import os
import h5py
import numpy as np
import torch
import sys
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

data_dir = '/home/fan/pytorch/CTC_pytorch/data_prepare/timit'

#Override the class of Dataset
#Define my own dataset over timit used the feature extracted by kaldi
class myDataset(Dataset):
    def __init__(self, data_set='train', feature_type='mfcc', n_feats=39):
        self.n_feats = n_feats
        self.data_set = data_set
        self.feature_type = feature_type
        h5_file = os.path.join(data_dir, feature_type+'_tmp', data_set+'.h5py')
        mfcc_file = os.path.join(data_dir, "feature_"+feature_type, data_set+'.txt')
        label_file = os.path.join(data_dir,"label_char", data_set+'_label.txt')
        char_file = os.path.join(data_dir, 'char_list.txt')
        if not os.path.exists(h5_file):
            print("Process %s data in kaldi format..." % data_set)
            self.process_txt(mfcc_file, label_file, char_file, h5_file)
        else:
            print("Loading %s data from h5py file..." % data_set)
            self.load_h5py(h5_file)
    
    def process_txt(self, mfcc_file, label_file, char_file, h5_file):
        #read map file
        self.char_map = dict()
        f = open(char_file, 'r')
        for line in f.readlines():
            char, num = line.split(' ')
            self.char_map[char] = int(num.strip())
        f.close()

        #read the label file
        label_dict = dict()
        f = open(label_file, 'r')
        for label in f.readlines():
            label = label.strip()
            utt = label.split('\t', 1)[0]
            label = label.split('\t', 1)[1]
            label_list = []
            for i in range(len(label)):
                if label[i].lower() in self.char_map:
                    label_list.append(self.char_map[label[i].lower()])
                if label[i] == ' ':
                    label_list.append(28)
            label_dict[utt] = np.array(label_list)
        f.close()
        
        #read the mfcc file
        mfcc_dict = dict()
        f = open(mfcc_file, 'r')
        for line in f.readlines():
            mfcc_frame = list()
            line = line.strip().split()
            if len(line) == 2:
                utt = line[0]
                mfcc_dict[utt] = list()
                continue
            if len(line) > 2:
                for i in range(self.n_feats):
                    mfcc_frame.append(float(line[i]))
            mfcc_dict[utt].append(mfcc_frame)
        f.close()
        
        if len(mfcc_dict) != len(label_dict):
            print("%s data: The num of wav and text are not the same!" % self.data_set)
            sys.exit(1)

        self.features_label = []
        #save the data as h5 file
        f = h5py.File(h5_file, 'w')
        for utt in mfcc_dict:
            grp = f.create_group(utt)
            self.features_label.append((torch.FloatTensor(np.array(mfcc_dict[utt])), label_dict[utt].tolist()))
            grp.create_dataset('data', data=np.array(mfcc_dict[utt]))
            grp.create_dataset('label', data=label_dict[utt])
        print("Saved the %s data to h5py file" % self.data_set)
        #print(self.__getitem__(1))
            
            
    def load_h5py(self, h5_file):
        self.features_label = []
        f = h5py.File(h5_file, 'r')
        for grp in f:
            self.features_label.append((torch.FloatTensor(np.asarray(f[grp]['data'])), np.asarray(f[grp]['label']).tolist()))
        print("Load %d sentences from %s dataset" % (self.__len__(), self.data_set))

    def __getitem__(self, idx):
        return self.features_label[idx]

    def __len__(self):
        return len(self.features_label) 

def create_input(batch):
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

#class torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, 
#                           sampler=None, batch_sampler=None, num_workers=0, 
#                           collate_fn=<function default_collate>, 
#                           pin_memory=False, drop_last=False)
#subclass of DataLoader and rewrite the collate_fn to form batch

class myDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(myDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = create_input

if __name__ == '__main__':
    dev_dataset = myDataset(data_set='dev', feature_type="fbank", n_feats=40)
    dev_loader = myDataLoader(dev_dataset, batch_size=8, shuffle=True, 
                     num_workers=4, pin_memory=False)
    i = 0
    for data in dev_loader:
        if i == 0:
            print(data)
        i += 1

