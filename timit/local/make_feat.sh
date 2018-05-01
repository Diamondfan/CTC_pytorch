#!/bin/bash

#The script is to make Fbank and MFCC from kaldi
#The spectrum is extracted by librosa in data_loader.py

feat_type=$1
compress=true
data_dir=./data_prepare

if [ "$feat_type" == "Fbank" ]; then
    echo ============================================================================
    echo "                Fbank Feature Extration and CMVN                          "
    echo ============================================================================

    fbank_config=./conf/fbank.conf

	for x in train dev test; do
        compute-fbank-feats --config=$fbank_config scp,p:$data_dir/$x/wav_sph.scp ark:- | \
            copy-feats --compress=$compress ark:- ark,scp:$data_dir/$x/raw_fbank.ark,$data_dir/$x/raw_fbank.scp
        if [ $x == train ]; then
            compute-cmvn-stats --binary=false scp:$data_dir/$x/raw_fbank.scp $data_dir/global_fbank_cmvn.txt
        fi
        apply-cmvn --norm-vars=true $data_dir/global_fbank_cmvn.txt scp:$data_dir/$x/raw_fbank.scp \
            ark,scp:$data_dir/$x/fbank.ark,$data_dir/$x/fbank.scp
        rm -f $data_dir/$x/raw_fbank.ark $data_dir/$x/raw_fbank.scp
    done
fi

if [ "$feat_type" == 'MFCC' ]; then
	echo ============================================================================
	echo "                    MFCC Feature Extraction & CMVN                        "
	echo ============================================================================

	mfcc_config=./conf/mfcc.conf

	for x in train dev test; do
        compute-mfcc-feats --config=$mfcc_config scp,p:$data_dir/$x/wav_sph.scp ark:- | \
            copy-feats --compress=$compress ark:- ark,scp:$data_dir/$x/raw_mfcc.ark,$data_dir/$x/raw_mfcc.scp
        if [ $x == train ]; then
            compute-cmvn-stats --binary=false scp:$data_dir/$x/raw_mfcc.scp $data_dir/global_mfcc_cmvn.txt
        fi
        apply-cmvn --norm-vars=true $data_dir/global_mfcc_cmvn.txt scp:$data_dir/$x/raw_mfcc.scp ark:- | add-deltas ark:- \
            ark,scp:$data_dir/$x/mfcc.ark,$data_dir/$x/mfcc.scp
        rm -f $data_dir/$x/raw_mfcc.ark $data_dir/$x/raw_mfcc.scp
    done
fi	

if [ "$feat_type" == "Spect" ]; then
    echo ============================================================================
    echo "                     Spectrum Feature Extraction                          "
    echo ============================================================================
    
    for x in train dev test; do
        echo "$x dataset:"
        python ./local/make_spectrum.py $data_dir/$x/wav.scp $data_dir/$x/spectrum.ark $data_dir/$x/spectrum.scp
    done
fi

echo "Finished successfully on" `date`
exit 0
