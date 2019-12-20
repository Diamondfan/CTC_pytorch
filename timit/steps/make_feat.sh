#!/bin/bash

#The script is to make fbank, mfcc and spectrogram from kaldi

feat_type=$1
data_dir=$2
conf_dir=conf
compress=false

if [ "$feat_type" != "fbank" || "$feat_type" != "mfcc" || "$feat_type" != "spectrogram" ]; then
    echo "Feature type $feat_type does not support!"
    exit 1;
else
    echo ============================================================================
    echo "                $feat_type Feature Extration and CMVN                          "
    echo ============================================================================

    feat_config=$conf_dir/$feat_type.conf
    if [ ! -f $feat_config ]; then
        echo "missing file $feat_config!"
        exit 1;
    fi

    x=train
    compute-$feat_type-feats --config=$feat_config scp,p:$data_dir/$x/wav_sph.scp \
                        ark,scp:$data_dir/$x/raw_$feat_type.ark,$data_dir/$x/raw_$feat_type.scp 
    #compute mean and variance with all training samples
    compute-cmvn-stats --binary=false scp:$data_dir/$x/raw_$feat_type.scp $data_dir/global_${feat_type}_cmvn.txt
    #apply cmvn for training set
    apply-cmvn --norm-vars=true $data_dir/global_${feat_type}_cmvn.txt scp:$data_dir/$x/raw_$feat_type.scp ark:- |\
        copy-feats --compress=$compress ark:- ark,scp:$data_dir/$x/$feat_type.ark,$data_dir/$x/$feat_type.scp
    rm -f $data_dir/$x/raw_$feat_type.ark $data_dir/$x/raw_$feat_type.scp

	for x in dev test; do
        compute-$feat_type-feats --config=$feat_config scp,p:$data_dir/$x/wav_sph.scp ark:- | \
            apply-cmvn --norm-vars=true $data_dir/global_${feat_type}_cmvn.txt ark:- ark:- |\
                copy-feats --compress=$compress ark:- ark,scp:$data_dir/$x/$feat_type.ark,$data_dir/$x/$feat_type.scp
    done
fi

echo "Finished successfully on" `date`
exit 0
