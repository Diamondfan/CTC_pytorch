#!/usr/bin/python
#encoding=utf-8

import pickle

map_file = open('./phones.60-48-39.map', 'r')

map_dict = {}
for line in map_file.readlines():
    line = line.strip().split('\t')
    if len(line) == 3:
        map_dict[line[1]] = line[2]
map_file.close()

#map_dict['q'] = ''
print(map_dict)
f = open('./map_dict.pkl', 'wb')
pickle.dump(map_dict, f)
f.close
