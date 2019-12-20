#encoding=utf-8

import os
import sys
import argparse

parser = argparse.ArgumentParser(description="Normalize the phoneme on TIMIT")
parser.add_argument("--map", default="./decode_map_48-39/phones.60-48-39.map", help="The map file")
parser.add_argument("--to", default=48, help="Determine how many phonemes to map")
parser.add_argument("--src", default='./data_prepare/train/phn_text', help="The source file to mapping")
parser.add_argument("--tgt", default='./data_prepare/train/48_text' ,help="The target file after mapping")

def main():
    args = parser.parse_args()
    if not os.path.exists(args.map) or not os.path.exists(args.src):
        print("Map file or source file not exist !")
        sys.exit(1)
    
    map_dict = {}
    with open(args.map) as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            if args.to == "60-48":
                if len(line) == 1:
                    map_dict[line[0]] = ""
                else:
                    map_dict[line[0]] = line[1]
            elif args.to == "60-39": 
                if len(line) == 1:
                    map_dict[line[0]] = ""
                else:
                    map_dict[line[0]] = line[2]
            elif args.to == "48-39":
                if len(line) == 3:
                    map_dict[line[1]] = line[2]
            else:
                print("%s phonemes are not supported" % args.to)
                sys.exit(1)
    
    with open(args.src, 'r') as rf, open(args.tgt, 'w') as wf:
        for line in rf.readlines():
            line = line.strip().split(' ')
            uttid, utt = line[0], line[1:]
            map_utt = [ map_dict[phone] for phone in utt if map_dict[phone] != "" ]
            wf.writelines(uttid + ' ' + ' '.join(map_utt) + '\n')

if __name__ == "__main__":
    main()
