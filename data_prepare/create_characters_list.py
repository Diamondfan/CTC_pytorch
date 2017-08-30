#/usr/bin/python
#encoding=utf-8

from collections import Counter
import re

label_file = './timit/train_label.txt'
char_file = './timit/char_list.txt'

char_list = []
f = open(label_file, 'r')
for label in f.readlines():
    label = label.strip()
    utt, label = label.split('\t', 1)
    label = label.lower()
    char_list += [ label[i] for i in range(len(label))]
f.close()
char_list = list(set(char_list))
f = open(char_file, 'w')
f.write("SPACE 0\n")
count = 1
for x in char_list:
    if re.search('[a-z\']', x) != None:
        f.write(x+' '+str(count)+'\n')
        count += 1
f.close()
