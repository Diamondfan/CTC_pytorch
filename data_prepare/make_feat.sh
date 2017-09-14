#!/bin/bash

#The script is to make feat for NN input

. ./cmd.sh
[ -f path.sh ] && . ./path.sh
set -e

stage=1
feats_nj=8
feat_type='mfcc'

if [ "$feat_type" == "Fbank" ]; then
	echo ============================================================================
	echo "                Fbank Feature Extration and CMVN                 "
	echo ============================================================================

	fbankdir=fbank
	
	if [ $stage -gt 0 ]; then
		cat ./data/train/feats.scp > $fbankdir/feats.scp
		cat ./data/test/feats.scp >> $fbankdir/feats.scp
		cat ./data/dev/feats.scp >> $fbankdir/feats.scp
		compute-cmvn-stats --binary=false scp:$fbankdir/feats.scp $fbankdir/global_cmvn.txt
	fi

	for x in train dev test; do
  		if [ $stage -eq 0 ]; then
			steps/make_fbank.sh --cmd "$train_cmd" --nj $feats_nj data/$x exp/make_fbank/$x $fbankdir
		else
			apply-cmvn --norm-vars=true $fbankdir/global_cmvn.txt scp:data/$x/feats.scp ark,t:$fbankdir/$x.txt
		fi
	done
fi

if [ "$feat_type" == 'mfcc' ]; then
	echo ============================================================================
	echo "         MFCC Feature Extration & CMVN for Training and Test set          "
	echo ============================================================================

	# Now make MFCC features.
	mfccdir=mfcc
	
	if [ $stage -gt 0 ]; then
		cat ./data/train/feats.scp > $mfccdir/feats.scp
		cat ./data/test/feats.scp >> $mfccdir/feats.scp
		cat ./data/dev/feats.scp >> $mfccdir/feats.scp
		compute-cmvn-stats --binary=false scp:$mfccdir/feats.scp $mfccdir/global_cmvn.txt
	fi

	for x in train dev test; do
  		if [ $stage -eq 0 ]; then
			steps/make_mfcc.sh --cmd "$train_cmd" --nj $feats_nj data/$x exp/make_mfcc/$x $mfccdir
		else
			apply-cmvn --norm-vars=true $mfccdir/global_cmvn.txt scp:data/$x/feats.scp ark:- | add-deltas ark:- ark,t:$mfccdir/$x.txt
		fi
	done
fi	

echo ============================================================================
echo "Finished successfully on" `date`
echo ============================================================================

exit 0
