#/usr/bin/python
#encoding=utf-8

import re
import sys

stage = sys.argv[1]
print(stage)
if stage == '0':
    import nltk
    read_file = './lm_train.text'
    write_file = 'LM.txt'
    label_file = './timit/phone_list.txt'

    char2int = dict()
    char2int['#']=0
    f = open(label_file, 'r')
    for line in f.readlines():
        line = line.strip().split()
        char2int[line[0]] = int(line[1])
    f.close
    
    sen = []
    f = open(read_file, 'r')
    wf = open(write_file, 'w')
    dict_map = '#123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLM'
    
    for line in f.readlines():
        line = line.strip().lower().split()
        for phone in line:
            sen.append(dict_map[char2int[phone]])
        wf.writelines(' '.join(sen)+'\n')
        sen = []
    f.close()
    
if stage == '1':
    #import pytorch_ctc
    #labels = "#'acbedgfihkjmlonqpsrutwvyxz_"
    #pytorch_ctc.generate_lm_trie('./dict.txt', './bigram.ken','./trie', labels, 0, 28)
    dicte = '123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLM'
    f = open('./dict.txt', 'w')
    for i in range(len(dicte)):
        f.writelines(dicte[i]+'\n')
    f.writelines('<s>\n')
    f.writelines('</s>\n')
    f.writelines(' \n')
    f.close()
