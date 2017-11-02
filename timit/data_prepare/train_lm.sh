#!/bin/bash

#cut -d' ' -f2- ./timit/label_phone/train.text > ./lm_train.text
#| sed -e 's:^:<s> :' -e 's:$: </s>:' \
#	> ./lm_train.text

/home/fan/KenLM/kenlm/build/bin/lmplz -o 2 --discount_fallback < ./LM.txt > bigram.arpa
/home/fan/KenLM/kenlm/build/bin/build_binary bigram.arpa bigram.binary
