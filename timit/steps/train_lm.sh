#training LM with irlstm in kaldi/tools

. path.sh
export IRSTLM=$KALDI_ROOT/tools/irstlm/
export PATH=${PATH}:$IRSTLM/bin

srcdir=$1

if ! command -v prune-lm >/dev/null 2>&1 ; then
  echo "$0: Error: the IRSTLM is not available or compiled" >&2
  echo "$0: Error: We used to install it by default, but." >&2
  echo "$0: Error: this is no longer the case." >&2
  echo "$0: Error: To install it, go to $KALDI_ROOT/tools" >&2
  echo "$0: Error: and run extras/install_irstlm.sh" >&2
  exit 1
fi

cut -d' ' -f2- $srcdir/train/phn_text | sed -e 's:^:<s> :' -e 's:$: </s>:' \
  > $srcdir/lm_train.text

build-lm.sh -i $srcdir/lm_train.text -n 2 -o lm_phone_bg.ilm.gz

compile-lm lm_phone_bg.ilm.gz -t=yes /dev/stdout > $srcdir/lm_phone_bg.arpa

rm -f lm_phone_bg.ilm.gz



