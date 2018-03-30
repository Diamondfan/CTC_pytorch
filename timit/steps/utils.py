#!/usr/bin/python
#encoding=utf-8

__author__ = 'Richardfan'

import math
import torch
import librosa
import torchaudio

def load_audio(path):
    '''
    Input:
        path     : string 载入音频的路径
    Output:
        sound    : numpy.ndarray 单声道音频数据，如果是多声道进行平均
    '''
    sound, _ = torchaudio.load(path)
    sound = sound.numpy()
    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)
    return sound

def parse_audio(path, audio_conf, windows, normalize=False):
    '''
    Input:
        path       : string 导入音频的路径
        audio_conf : dict 求频谱的音频参数
        windows    : dict 加窗类型
    Output:
        spect      : FloatTensor  每帧的频谱
    '''
    y = load_audio(path)
    n_fft = int(audio_conf['sample_rate']*audio_conf["window_size"])
    win_length = n_fft
    hop_length = int(audio_conf['sample_rate']*audio_conf['window_stride'])
    window = windows[audio_conf['window']]
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                        win_length=win_length, window=window)
    spect, phase = librosa.magphase(D) 
    
    spect = torch.FloatTensor(spect)
    spect = spect.log1p()
    
    if normalize:
        mean = spect.mean()
        std = spect.std()
        spect.add_(-mean)
        spect.div_(std)
    
    return spect.transpose(0,1)

def F_Mel(fre_f, audio_conf):
    '''
    Input:
        fre_f       : FloatTensor log spectrum
        audio_conf  : 主要需要用到采样率
    Output:
        mel_f       : FloatTensor  换成mel频谱
    '''
    n_mels = fre_f.size(1)
    mel_bin = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=audio_conf["sample_rate"]/2) * audio_conf["window_size"]
    count = 0
    fre_f = fre_f.numpy().tolist()
    mel_f = []
    for frame in fre_f:
        mel_f_frame = []
        for i in range(n_mels):
            left = int(math.floor(mel_bin[i]))
            right = left + 1
            tmp = (frame[right] - frame[left]) * (mel_bin[i] - left) + frame[left]      #线性插值
            mel_f_frame.append(tmp)
        mel_f.append(mel_f_frame)
    return torch.FloatTensor(mel_f)

def process_kaldi_feat(feat_file, feat_size):
    '''
    Input:
        feat_file  :  string 特征文件路径
        feat_size  :  int    特征大小 即特征维数
    Output:
        feat_dict  :  dict   特征文件中的特征，每个utt的特征是list类型
    '''
    feat_dict = dict()
    f = open(feat_file, 'r')
    for line in f.readlines():
        feat_frame = list()
        line = line.strip().split()
        if len(line) == 2:
            utt = line[0]
            feat_dict[utt] = list()
            continue
        if len(line) > 2:
            for i in range(feat_size):
                feat_frame.append(float(line[i]))
        feat_dict[utt].append(feat_frame)
    f.close()
    return feat_dict

def process_label_file(label_file, label_type, class2int):
    '''
    Input:
        label_file  : string  标签文件路径
        label_type  : string  标签类型(目前只支持字符和音素)
        class2int   : dict    标签和数字的对应关系
    Output:
        label_dict  : dict    所有句子的标签，每个句子是numpy类型
    '''
    label_dict = dict()
    f = open(label_file, 'r')
    for label in f.readlines():
        label = label.strip()
        label_list = []
        if label_type == 'char':
            utt = label.split('\t', 1)[0]
            label = label.split('\t', 1)[1]
            for i in range(len(label)):
                if label[i].lower() in class2int:
                    label_list.append(class2int[label[i].lower()])
                if label[i] == ' ':
                    label_list.append(class2int['SPACE'])
        else:
            label = label.split()
            utt = label[0]
            for i in range(1,len(label)):
                label_list.append(class2int[label[i]])
        label_dict[utt] = label_list
    f.close()
    return label_dict

def process_map_file(map_file):
    '''
    Input:
        map_file  : string label和数字的对应关系文件
    Output:
        class2int  : dict  label到数字的对应关系
        int2class  : dict  数字到label的对应关系
    '''
    class2int = dict()
    int2class = dict()
    f = open(map_file, 'r')
    for line in f.readlines():
        label, num = line.strip().split(' ')
        class2int[label] = int(num)
        int2class[int(num)] = label
    f.close()
    int2class[0] = '#'
    return class2int, int2class

if __name__ == '__main__':
    import scipy.signal
    windows = {'hamming':scipy.signal.hamming, 'hann':scipy.signal.hann, 'blackman':scipy.signal.blackman,
            'bartlett':scipy.signal.bartlett}
    audio_conf = {"sample_rate":16000, 'window_size':0.025, 'window_stride':0.01, 'window': 'hamming'} 
    path = '/home/fan/Audio_data/TIMIT/train/dr1/fcjf0/sa1.wav'
    spect = parse_audio(path, audio_conf, windows, normalize=False)
    mel_f = F_Mel(spect, audio_conf)
    
    import visdom
    viz = visdom.Visdom(env='fan')
    viz.heatmap(spect.transpose(0, 1), opts=dict(title="Log Spectrum", xlabel="She had your dark suit in greasy wash water all year.", ylabel="Frequency"))
    viz.heatmap(mel_f.transpose(0, 1), opts=dict(title="Log Mel Spectrum", xlabel="She had your dark suit in greasy wash water all year.", ylabel="Frequency"))
    
