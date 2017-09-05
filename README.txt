Start time : 2017.8.24
Author     : Richardfan

The END-To-END CTC for speech recognition system. This part used pytorch

Data:
English Corpus: Timit
	Training set: 3696 sentences
	Dev set: 400 sentences
	Test set: 192 sentences

Chinese Corpus:863 Corpus
	Training set:
	M50		F50		A1-A521
				  	AW1-AW129		650 sentences
	M54		F54		B522-B1040
					BW130-BW259		649 sentences
	M60		F60		C1041-C1560
					CW260-CW388  	649 sentences
	M64		F64		D1-D625         625 sentences
	Total:5146 sentences

	Test set:
	M51		F51		A1-A100			100 sentences
	M55		F55		B522-B521		100 sentences
	M61		F61		C1041-C1140		100 sentences
	M63		F63		D1-D100         100 sentences
	Total:800 sentences

warpctc_pytorch install:
	Notice: Reinstall the pytorch with source code instead of pip.
	Baidu warp-ctc and binding with pytorch.
	https://github.com/SeanNaren/warp-ctc/

ToDo:
data preparation for 863 corpus, little different from English one
model testing
training part
test part
combine with LM


