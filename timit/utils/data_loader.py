#!/usr/bin/python
#encoding=utf-8

import torch
import kaldiio
import numpy as np
from torch.utils.data import Dataset, DataLoader

from utils.tools import load_wave, F_Mel, make_context, skip_feat

audio_conf = {"sample_rate":16000, 'window_size':0.025, 'window_stride':0.01, 'window': 'hamming'}

class Vocab(object):
    def __init__(self, vocab_file):
        self.vocab_file = vocab_file
        self.word2index = {"blank": 0, "UNK": 1}
        self.index2word = {0: "blank", 1: "UNK"}
        self.word2count = {}
        self.n_words = 2
        self.read_lang()

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def read_lang(self):
        print("Reading vocabulary from {}".format(self.vocab_file))
        with open(self.vocab_file, 'r') as rf:
            line = rf.readline()
            while line:
                line = line.strip().split(' ')
                if len(line) > 1:
                    sen = ' '.join(line[1:])
                else:
                    sen = line[0]
                self.add_sentence(sen)
                line = rf.readline()
        print("Vocabulary size is {}".format(self.n_words))


class SpeechDataset(Dataset):
    def __init__(self, vocab, scp_path, lab_path, opts):
        self.vocab = vocab
        self.scp_path = scp_path
        self.lab_path = lab_path
        self.left_ctx = opts.left_ctx
        self.right_ctx = opts.right_ctx
        self.n_skip_frame = opts.n_skip_frame
        self.n_downsample = opts.n_downsample
        self.feature_type = opts.feature_type
        self.mel = opts.mel
        
        if opts.feature_type == "waveform":
            self.label_dict = process_label_file(label_file, self.out_type, self.class2int)
            self.item = []
            with open(wav_path, 'r') as f:
                for line in f.readlines():
                    utt, path = line.strip().split('\t')
                    self.item.append((path, self.label_dict[utt]))
        else:
            self.process_feature_label()
    
    def process_feature_label(self):
        path_dict = []
        #read the ark path
        with open(self.scp_path, 'r') as rf:
            line = rf.readline()
            while line:
                utt, path = line.strip().split(' ')
                path_dict.append((utt, path))
                line = rf.readline()
        
       	#read the label
        label_dict = dict()
        with open(self.lab_path, 'r') as rf:
            line = rf.readline()
            while line:
                utt, label = line.strip().split(' ', 1)
                label_dict[utt] = [self.vocab.word2index[c] if c in self.vocab.word2index else self.vocab.word2index['UNK'] for c in label.split()]
                line = rf.readline() 
        
        assert len(path_dict) == len(label_dict)
        print("Reading %d lines from %s" % (len(label_dict), self.lab_path))
        
        self.item = []
        for i in range(len(path_dict)):
            utt, path = path_dict[i]
            self.item.append((path, label_dict[utt], utt))

    def __getitem__(self, idx):
        if self.feature_type == "waveform":
            path, label = self.item[idx]
            return (load_wave(path), label)
        else:
            path, label, utt = self.item[idx]
            feat = kaldiio.load_mat(path)
            feat= skip_feat(make_context(feat, self.left_ctx, self.right_ctx), self.n_skip_frame)
            seq_len, dim = feat.shape
            if seq_len % self.n_downsample != 0:
                pad_len = self.n_downsample - seq_len % self.n_downsample
                feat = np.vstack([feat, np.zeros((pad_len, dim))])
            if self.mel:
                return (F_Mel(torch.from_numpy(feat), audio_conf), label)
            else:
                return (torch.from_numpy(feat), torch.LongTensor(label), utt)

    def __len__(self):
        return len(self.item) 

def create_input(batch):
    inputs_max_length = max(x[0].size(0) for x in batch)
    feat_size = batch[0][0].size(1)
    targets_max_length = max(x[1].size(0) for x in batch)
    batch_size = len(batch)
    batch_data = torch.zeros(batch_size, inputs_max_length, feat_size) 
    batch_label = torch.zeros(batch_size, targets_max_length)
    input_sizes = torch.zeros(batch_size)
    target_sizes = torch.zeros(batch_size)
    utt_list = []

    for x in range(batch_size):
        feature, label, utt = batch[x]
        feature_length = feature.size(0)
        label_length = label.size(0)

        batch_data[x].narrow(0, 0, feature_length).copy_(feature)
        batch_label[x].narrow(0, 0, label_length).copy_(label)
        input_sizes[x] = feature_length / inputs_max_length
        target_sizes[x] = label_length
        utt_list.append(utt)
    return batch_data.float(), input_sizes.float(), batch_label.long(), target_sizes.long(), utt_list 
    
'''
class torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, 
                                                    collate_fn=<function default_collate>, pin_memory=False, drop_last=False)
subclass of DataLoader and rewrite the collate_fn to form batch
'''

class SpeechDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(SpeechDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = create_input

if __name__ == '__main__':
    dev_dataset = SpeechDataset()
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
