#encoding=utf-8

#The code make the Full-mell Spectrogram feature and save it as ark and scp
#Author:  Richardfan
#Date:    2018.4.24

import torchaudio
import librosa
import numpy as np
import struct
import sys
import scipy.signal

class KaldiWriteOut(object):
    def __init__(self, ark_path, scp_path):
        self.ark_path = ark_path
        self.scp_path = scp_path
	self.ark_file_write = open(ark_path, 'wb')
        self.scp_file_write = open(scp_path, 'w')
        self.pos = 0

    def write_kaldi_mat(self, utt_id, utt_mat):
        utt_mat = np.asarray(utt_mat, dtype=np.float32)
        rows, cols = utt_mat.shape
        self.ark_file_write.write(struct.pack('<%ds'%(len(utt_id)), utt_id))
        self.ark_file_write.write(struct.pack('<cxcccc', ' ', 'B', 'F', 'M', ' '))
        self.ark_file_write.write(struct.pack('<bi', 4, rows))
        self.ark_file_write.write(struct.pack('<bi', 4, cols))
        self.ark_file_write.write(utt_mat)
        self.pos += len(utt_id) + 1
        self.scp_file_write.write(utt_id + ' ' + self.ark_path + ':' + str(self.pos) + '\n')
        self.pos += 3 * 5 + (rows * cols * 4)

    def close(self):
        self.ark_file_write.close()
        self.scp_file_write.close()

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
            sound - sound.mean(axis=1)
    return sound

def parse_audio(path, audio_conf, windows, normalize=True):
    '''
    Input:
        path       : string 导入音频的路径
        audio_conf : dcit 求频谱的音频参数
        windows    : dict 加窗类型
    Output:
        spect      : ndarray  每帧的频谱
    '''
    y = load_audio(path)
    n_fft = int(audio_conf['sample_rate']*audio_conf["window_size"])
    win_length = n_fft
    hop_length = int(audio_conf['sample_rate']*audio_conf['window_stride'])
    window = windows[audio_conf['window']]
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                        win_length=win_length, window=window)
    spect, phase = librosa.magphase(D)

    spect = np.log1p(spect)
    
    if normalize:
        mean = spect.mean()
        std = spect.std()
        spect = np.add(spect, -mean)
        spect = np.divide(spect, std)
    
    return spect.transpose()

def make_spectrum(wave_path, ark_file, scp_file):
    windows = {'hamming':scipy.signal.hamming, 'hann':scipy.signal.hann, 'blackman':scipy.signal.blackman,
                'bartlett':scipy.signal.bartlett}
    audio_conf = {"sample_rate":16000, 'window_size':0.025, 'window_stride':0.01, 'window': 'hamming'}
    arkwriter = KaldiWriteOut(ark_file, scp_file)
    with open(wave_path, 'r') as rf:
        i = 0
        for lines in rf.readlines():
            utt_id, path = lines.strip().split()
            utt_mat = parse_audio(path, audio_conf, windows, normalize=True)
            arkwriter.write_kaldi_mat(utt_id, utt_mat)
            i += 1
            if i %10 == 0:
                print("Processed %d sentences..." % i)
        arkwriter.close()
        print("Done. Processed %d sentences..." % i)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python "+sys.argv[0] + ' [wav_path] [ark file to write] [scp file to write]')
        sys.exit(1)
    wave_path = sys.argv[1]
    ark_file = sys.argv[2]
    scp_file = sys.argv[3]
    make_spectrum(wave_path, ark_file, scp_file)   
     


