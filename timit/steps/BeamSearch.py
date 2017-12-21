#!/usr/bin/python
#encoding=utf-8

import numpy as np
import torch

class BeamEntry:
    "information about one single beam at specific time-step"
    def __init__(self):
        self.prTotal=0 # blank and non-blank
	self.prNonBlank=0 # non-blank
	self.prBlank=0 # blank
	self.y=() # labelling at current time-step


class BeamState:
    "information about beams at specific time-step"
    def __init__(self):
	self.entries={}

    def norm(self):
	"length-normalise probabilities to avoid penalising long labellings"
	for (k,v) in self.entries.items():
	    labellingLen=len(self.entries[k].y)
	    self.entries[k].prTotal=self.entries[k].prTotal*(1.0/(labellingLen if labellingLen else 1))

    def sort(self):
	"return beams sorted by probability"
	u=[v for (k,v) in self.entries.items()]
	s=sorted(u, reverse=True, key=lambda x:x.prTotal)
	return [x.y for x in s]

class ctcBeamSearch(object):
    def __init__(self, classes, beam_width, lm, lm_alpha=0.01, blank_index=0):	
        self.classes = classes
        self.beamWidth = beam_width
        self.lm_alpha = lm_alpha
        self.lm = lm
        self.blank_index = blank_index
        
    def calcExtPr(self, k, y, t, mat, beamState):
        "probability for extending labelling y to y+k"
	
        # language model (char bigrams)
        bigramProb=1
        if self.lm:
	    c1=self.classes[y[-1]] if len(y) else ' '
	    c2=self.classes[k]
            bigramProb=self.lm.getCharBigram(c1,c2)**self.lm_alpha

        # optical model (RNN)
        if len(y) and y[-1]==k:
	    return mat[t, k]*bigramProb*beamState.entries[y].prBlank
        else:
	    return mat[t, k]*bigramProb*beamState.entries[y].prTotal
    
    def addLabelling(self, beamState, y):
        "adds labelling if it does not exist yet"
        if y not in beamState.entries:
	    beamState.entries[y]=BeamEntry()
    
    def decode(self, inputs, inputs_list):
        '''
        mat : FloatTesnor batch * timesteps * class
        '''
        batches, maxT, maxC = inputs.size()
	res = []
        
        for batch in range(batches):
            mat = inputs[batch].numpy()
            # Initialise beam state
            last=BeamState()
            y=()
            last.entries[y]=BeamEntry()
            last.entries[y].prBlank=1
            last.entries[y].prTotal=1
            
            # go over all time-steps
            for t in range(inputs_list[batch]):
                curr=BeamState()
                if (1 - mat[t, self.blank_index]) < 0.1:         #跳过概率很接近1的blank帧，增加解码速度
                    continue
                # get best labellings
                BHat=last.sort()[0:self.beamWidth]                 #取前beam个最好的结果
                #print(BHat)
                # go over best labellings
                for y in BHat:
                    prNonBlank=0
                    # if nonempty labelling
                    if len(y)>0:
                        # seq prob so far and prob of seeing last label again
                        prNonBlank=last.entries[y].prNonBlank*mat[t, y[-1]]                    #相同的y两种可能，加入重复或者加入空白,如果之前没有字符，在NonBlank概率为0
                            
                    # calc probabilities
                    prBlank=(last.entries[y].prTotal)*mat[t, self.blank_index]
                    # save result
                    self.addLabelling(curr, y)
                    curr.entries[y].y=y
                    curr.entries[y].prNonBlank+=prNonBlank
                    curr.entries[y].prBlank+=prBlank
                    curr.entries[y].prTotal+=prBlank+prNonBlank
                            
                    # extend current labelling
                    for k in range(maxC):                                         #t时刻加入其它的label,此时Blank的概率为0，如果加入的label与最后一个相同，因为不能重复，所以上一个字符一定是blank
                        if k != self.blank_index:
                            newY=y+(k,)
                            prNonBlank=self.calcExtPr(k, y, t, mat, last)
                                    
                            # save result
                            self.addLabelling(curr, newY)
                            curr.entries[newY].y=newY
                            curr.entries[newY].prNonBlank+=prNonBlank
                            curr.entries[newY].prTotal+=prNonBlank
                    
                    # set new beam state
                last=curr
                    
            # normalise probabilities according to labelling length
            last.norm() 
            
            # sort by probability
            bestLabelling=last.sort()[0] # get most probable labelling
            
            # map labels to chars
            res_b =' '.join([self.classes[l] for l in bestLabelling])
            res.append(res_b)
        return res


if __name__=='__main__':
    classes=["a","b"]
    mat=np.array([[[0.4, 0, 0.6], [0.4, 0, 0.6], [0, 1, 0], [0, 0, 0]], [[0.4, 0, 0.6],[0.4, 0, 0.6], [0.4, 0.1, 0.5], [0.2, 0.5, 0.3]]])
    mat = torch.FloatTensor(mat)
    input_list = [2, 2]
    print('Test beam search')
    expected='a'
    decoder = ctcBeamSearch(classes, 10, None, blank_index=len(classes))
    actual=decoder.decode(mat, input_list)
    print('Expected: "'+expected+'"')
    print(actual)
