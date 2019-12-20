#!/usr/bin/python
#encoding=utf-8

__author__ = 'Ruchao Fan'

import math
import torch
import numpy as np
#import librosa
#import torchaudio

def load_audio(path):
    """
    Args:
        path     : string 载入音频的路径
    Returns:
        sound    : numpy.ndarray 单声道音频数据，如果是多声道进行平均
    """
    sound, _ = torchaudio.load(path)
    sound = sound.numpy()
    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)
    return sound

def load_wave(path, normalize=True):
    """
    Args:
        path     : string 载入音频的路径
    Returns:
    """
    sound = load_audio(path)
    wave = torch.FloatTensor(sound)
    if normalize:
        mean = wave.mean()
        std = wave.std()
        wave.add_(-mean)
        wave.div_(std)
    return wave

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

def make_context(feature, left, right):
    if left==0 and right == 0:
        return feature
    feature = [feature]
    for i in range(left):
        feature.append(np.vstack((feature[-1][0], feature[-1][:-1])))
    feature.reverse()
    for i in range(right):
        feature.append(np.vstack((feature[-1][1:], feature[-1][-1])))
    return np.hstack(feature)

def skip_feat(feature, skip):
    '''
    '''
    if skip == 1 or skip == 0:
        return feature
    skip_feature=[]
    for i in range(feature.shape[0]):
        if i % skip == 0:
            skip_feature.append(feature[i])
    return np.vstack(skip_feature)

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

'''
if __name__ == '__main__':
    import scipy.signal
    windows = {'hamming':scipy.signal.hamming, 'hann':scipy.signal.hann, 'blackman':scipy.signal.blackman,
            'bartlett':scipy.signal.bartlett}
    audio_conf = {"sample_rate":16000, 'window_size':0.025, 'window_stride':0.01, 'window': 'hamming'} 
    path = '/home/fan/Audio_data/TIMIT/test/dr7/fdhc0/si1559.wav'
    spect = parse_audio(path, audio_conf, windows, normalize=True)
    mel_f = F_Mel(spect, audio_conf)
    wave = load_wav(path)
    print(wave)

    import visdom
    viz = visdom.Visdom(env='fan')
    viz.heatmap(spect.transpose(0, 1), opts=dict(title="Log Spectrum", xlabel="She had your dark suit in greasy wash water all year.", ylabel="Frequency"))
    viz.heatmap(mel_f.transpose(0, 1), opts=dict(title="Log Mel Spectrum", xlabel="She had your dark suit in greasy wash water all year.", ylabel="Frequency"))
    viz.line(wave.numpy())
'''

