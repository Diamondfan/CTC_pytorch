
import sys

if len(sys.argv) != 2:
    print("We need training text to generate the modelling units.")
    sys.exit(1)

train_text = sys.argv[1]
units_file = 'data/units'

units = {}
with open(train_text, 'r') as fin:   
    line = fin.readline()
    while line:
        line = line.strip().split(' ')
        for char in line[1:]:
            try:
                if units[char] == True:
                    continue
            except:
                units[char] = True
        line = fin.readline()

fwriter = open(units_file, 'w')
for char in units:
    print(char, file=fwriter)


