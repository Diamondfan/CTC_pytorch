#/usr/bin/python
#encoding=utf-8

#greedy decoder and beamsearch decoder for ctc

import torch
import numpy as np

class Decoder(object):
    def __init__(self, int2char, space_idx = 1, blank_index = 0):
        self.int_to_char = int2char
        self.space_idx = space_idx
        self.blank_index = blank_index
        self.num_word = 0
        self.num_char = 0

    def greedy_decoder(self, prob_tensor, frame_seq_len):
        prob_tensor = prob_tensor.transpose(0,1)         #batch_size*seq_len*output_size
        _, decoded = torch.max(prob_tensor, 2)
        decoded = decoded.view(decoded.size(0), decoded.size(1))
        strings = self._convert_to_strings(decoded, frame_seq_len)
        return self._process_strings(strings, remove_rep=True)

    def decode(self):
        raise NotImplementedError;

    def cer_wer(self, prob_tensor, frame_seq_len, targets, target_sizes):
        strings = self.decode(prob_tensor, frame_seq_len)
        targets = self._unflatten_targets(targets, target_sizes)
        target_strings = self._process_strings(self._convert_to_strings(targets))
        cer = 0
        wer = 0
        for x in range(len(target_strings)):
            cer += self.cer(strings[x], target_strings[x]) / float(len(target_strings[x]))
            wer += self.wer(strings[x], target_strings[x]) / float(len(target_strings[x].split()))
        return cer, wer

    def phone_word_error(self, prob_tensor, frame_seq_len, targets, target_sizes):
        strings = self.decode(prob_tensor, frame_seq_len)
        targets = self._unflatten_targets(targets, target_sizes)
        target_strings = self._process_strings(self._convert_to_strings(targets))
        cer = 0
        wer = 0
        for x in range(len(target_strings)):
            cer += self.cer(strings[x], target_strings[x])
            wer += self.wer(strings[x], target_strings[x])
            self.num_word += len(target_strings[x].split())
            self.num_char += len(target_strings[x])
        return cer, wer

    def _unflatten_targets(self, targets, target_sizes):
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset : offset + size])
            offset += size
        return split_targets

    def _process_strings(self, seqs, remove_rep = False):
        processed_strings = []
        for seq in seqs:
            string = self._process_string(seq, remove_rep)
            processed_strings.append(string)
        return processed_strings
   
    def _process_string(self, seq, remove_rep = False):
        string = ''
        for i, char in enumerate(seq):
            if char != self.int_to_char[self.blank_index]:
                if remove_rep and i != 0 and char == seq[i - 1]: #remove dumplicates
                    pass
                elif self.space_idx == -1:
                    string = string + ' '+ char
                elif char == self.int_to_char[self.space_idx]:
                    string += ' '
                else:
                    string = string + char
        return string

    def _convert_to_strings(self, seq, sizes=None):
        strings = []
        for x in range(len(seq)):
            seq_len = sizes[x] if sizes is not None else len(seq[x])
            string = self._convert_to_string(seq[x], seq_len)
            strings.append(string)
        return strings

    def _convert_to_string(self, seq, sizes):
        result = []
        for i in range(sizes):
            result.append(self.int_to_char[seq[i]])
        if self.space_idx == -1:
            return result
        else:
            return ''.join(result)
 
    def wer(self, s1, s2):
        b = set(s1.split() + s2.split())
        word2int = dict(zip(b, range(len(b))))

        w1 = [word2int[w] for w in s1.split()]
        w2 = [word2int[w] for w in s2.split()]
        return self._edit_distance(w1, w2)

    def cer(self, s1, s2):
        return self._edit_distance(s1, s2)
    
    def _edit_distance(self, src_seq, tgt_seq):      # compute edit distance between two iterable objects
        L1, L2 = len(src_seq), len(tgt_seq)
        if L1 == 0: return L2
        if L2 == 0: return L1
        # construct matrix of size (L1 + 1, L2 + 1)
        dist = [[0] * (L2 + 1) for i in range(L1 + 1)]
        for i in range(1, L2 + 1):
            dist[0][i] = dist[0][i-1] + 1
        for i in range(1, L1 + 1):
            dist[i][0] = dist[i-1][0] + 1
        for i in range(1, L1 + 1):
            for j in range(1, L2 + 1):
                if src_seq[i - 1] == tgt_seq[j - 1]:
                    cost = 0
                else:
                    cost = 1
                dist[i][j] = min(dist[i][j-1] + 1, dist[i-1][j] + 1, dist[i-1][j-1] + cost)
        return dist[L1][L2]


class GreedyDecoder(Decoder):
    def decode(self, prob_tensor, frame_seq_len):
        prob_tensor = prob_tensor.transpose(0,1)         # (n, t, c)
        _, decoded = torch.max(prob_tensor, 2)
        decoded = decoded.view(decoded.size(0), decoded.size(1))
        decoded = self._convert_to_strings(decoded, frame_seq_len)     # convert digit idx to chars
        return self._process_strings(decoded, remove_rep=True)


class BeamDecoder(Decoder):
    def __init__(self, int2char, top_paths = 1, beam_width = 200, blank_index = 0, space_idx = -1,
                    lm_path=None, trie_path=None, dict_path=None, lm_alpha=None, lm_beta1=None, lm_beta2=None):
        self.beam_width = beam_width
        self.top_n = top_paths
        self.labels = ['#']
        int2phone = dict()
        for digit in int2char:
            if digit != 0:
                label = bytes.decode(int2char[digit].tostring())
                self.labels.append(label)
                int2phone[digit] = label
        int2phone[0] = '#'
        super(BeamDecoder, self).__init__(int2phone, space_idx=space_idx, blank_index=blank_index)
        self.label2 = '#123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLM'

        try:
            from pytorch_ctc import CTCBeamDecoder, Scorer, KenLMScorer
            import pytorch_ctc
        except ImportError:
            raise ImportError("BeamCTCDecoder requires pytorch_ctc package.")

        if lm_path is not None:
            pytorch_ctc.generate_lm_trie(dict_path, lm_path, trie_path, self.label2, 0, -1)
            scorer = KenLMScorer(self.label2, lm_path, trie_path)
            scorer.set_lm_weight(lm_alpha)
            scorer.set_word_weight(lm_beta1)
            scorer.set_valid_word_weight(lm_beta2)
        else:
            scorer = Scorer()
        self._decoder = CTCBeamDecoder(scorer = scorer, labels = self.labels, top_paths = top_paths, beam_width = beam_width, blank_index = blank_index, space_index = space_idx, merge_repeated=False)

    def decode(self, prob_tensor, frame_seq_len):
        frame_seq_len = torch.IntTensor(frame_seq_len).cpu()
        decoded, _, out_seq_len = self._decoder.decode(prob_tensor, seq_len = frame_seq_len)
        decoded = decoded[0]
        out_seq_len = out_seq_len[0]
        decoded = self._convert_to_strings(decoded, out_seq_len)
        return self._process_strings(decoded)

if __name__ == '__main__':
    decoder = Decoder('abcde', 1, 2)
    print(decoder._convert_to_strings([[1,2,1,0,3],[1,2,1,1,1]]))

