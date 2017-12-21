#!/usrbin/python
#encoding=utf-8
import re

class LanguageModel:
    "simple language model: word level bigrams for beam search"
    def __init__(self, text_file):
	"read text from file to generate language model"
	#self.classes = classes
        self.initWordList(text_file)
	self.initCharBigrams(text_file)

    def initWordList(self, fn):
	"internal init of word list"
        f = open(fn, 'r')
        self.word_list = []
        for line in f.readlines():
            self.word_list.extend(line.strip().split())
        f.close()
	self.words=list(set(self.word_list))

    def initCharBigrams(self, fn):
	"internal init of word bigrams"
	self.bigram={}
	self.numSamples={}
        self.num_all = 0
        
        # init bigrams with 0 values
	for w1 in self.words:
	    self.bigram[w1]={}
	    self.numSamples[w1]=0
	    for w2 in self.words:
		self.bigram[w1][w2]=0
		
	# go through text and create each char bigrams
        f = open(fn, 'r')
        for lines in f.readlines():
            line = lines.strip().split()
            self.num_all += len(line)
            for i in range(len(line)-1):
                first=line[i]
	        second=line[i+1]
			
	        # ignore unknown chars
	        if first not in self.bigram or second not in self.bigram[first]:
		    continue
			
	        self.bigram[first][second]+=1
	        self.numSamples[first]+=1
            self.numSamples[line[-1]] += 1
        f.close()
        

    def getCharBigram(self, first, second):
	"probability of seeing character 'first' next to 'second'"
        first=first if len(first) else ' ' # map start to word beginning
	second=second if len(second) else ' ' # map end to word end
		
        if first == ' ':
            return float(self.numSamples[second]) / self.num_all 
        if self.numSamples[first]==0:
	    return 0
	return float(self.bigram[first][second]) / self.numSamples[first]


if __name__ == "__main__":
    lm = LanguageModel('../data_prepare/lm_train.text')
    print(lm.bigram['sil'])
    print(lm.num_all)
    print(lm.numSamples)
    print(lm.getCharBigram('sil', 'k'))
