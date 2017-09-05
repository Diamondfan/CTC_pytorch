#/usr/bin/python
#encoding=utf-8

#greedy decoder and beamsearch decoder for ctc

import torch

class Decoder(object):
    def __init__(self, labels, space_idx = 0, blank_index = 28):
        self.labels = labels
        self.int_to_char = dict([(i, c) for (i, c) in enumerate(labels)])
        self.space_idx = space_idx
        self.blank_index = blank_index

    def greedy_decoder(self, prob_tensor, frame_seq_len):
        prob_tensor = prob_tensor.transpose(0,1)         #batch_size*seq_len*output_size
        _, decoded = torch.max(prob_tensor, 2)
        decoded = decoded.view(decoded.size(0), decoded.size(1))
        strings = self._convert_to_strings(decoded, frame_seq_len)
        return self._process_strings(strings, remove_rep=True)

    def cer_wer(self, prob_tensor, frame_seq_len, targets, target_sizes):
        strings = self.greedy_decoder(prob_tensor, frame_seq_len)
        targets = self._unflatten_targets(targets, target_sizes)
        target_strings = self._process_strings(self._convert_to_strings(targets))
        cer = 0
        wer = 0
        for x in range(len(target_strings)):
            cer += self.cer(strings[x], target_strings[x]) / float(len(target_strings[x]))
            wer += self.wer(strings[x], target_strings[x]) / float(len(target_strings[x].split()))
        return cer, wer

    def phone_word_error(self, prob_tensor, frame_seq_len, targets, target_sizes):
        strings = self.greedy_decoder(prob_tensor, frame_seq_len)
        targets = self._unflatten_targets(targets, target_sizes)
        target_strings = self._process_strings(self._convert_to_strings(targets))
        cer = 0
        wer = 0
        self.num_word = 0
        for x in range(len(target_strings)):
            cer += self.cer(strings[x], target_strings[x])
            wer += self.wer(strings[x], target_strings[x])
            self.num_word += len(target_strings[x].split())
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
                elif char == self.labels[self.space_idx]:
                    string += ' '
                else:
                    string = string + char          
        return string

    def _convert_to_strings(self, seq, sizes=None):
        strings = []
        for x in range(len(seq)):
            seq_len = sizes[x] if sizes is not None else len(seq[x])
            strings.append(self._convert_to_string(seq[x], seq_len))
        return strings

    def _convert_to_string(self, seq, sizes):
        result = []
        for i in range(sizes):
            result.append(self.int_to_char[seq[i]])
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


class GreedyDecoder_test(Decoder):
    def __init__(self, labels, output = 'char', space_idx = -1):
        super(GreedyDecoder_test, self).__init__(space_idx)
        self.labels = labels
        self.output = output


    def decode(self, prob_tensor, frame_seq_len):
        prob_tensor = prob_tensor.transpose(0,1)         # (n, t, c)
        _, decoded = torch.max(prob_tensor, 2)
        decoded = decoded.view(decoded.size(0), decoded.size(1))
        decoded = self._process_seqs(decoded, frame_seq_len, remove_rep = True)
        decoded = self._convert_to_chars(decoded, self.labels)     # convert digit idx to chars
        if self.output == 'word':
             decoded = self._convert_to_words(decoded)
        return decoded


class BeamDecoder_test(Decoder):
    def __init__(self, labels, scorer, top_paths = 1, beam_width = 20, output = 'char', space_idx = -1):
        super(BeamDecoder_test, self).__init__(space_idx)
        self.labels = labels
        self.output = output
        assert top_paths == 1, "Only supports top 1 path in the current version"

        try:
            import pytorch_ctc
        except ImportError:
            raise ImportError("BeamCTCDecoder requires pytorch_ctc package.")

        self._decoder = pytorch_ctc.CTCBeamDecoder(scorer = scorer, labels = self.labels, top_paths = top_paths, beam_width = beam_width, blank_index = 0, space_index = self.space_idx, merge_repeated=False)

    def decode(self, prob_tensor, frame_seq_len):
        frame_seq_len = torch.IntTensor(frame_seq_len).cpu()
        decoded, _, out_seq_len = self._decoder.decode(prob_tensor, seq_len = frame_seq_len)
        decoded = decoded[0]
        out_seq_len = out_seq_len[0]
        decoded = self._process_seqs(decoded, out_seq_len, remove_rep = False)
        decoded = self._convert_to_chars(decoded, self.labels)     # convert digit idx to chars
        if self.output == 'word':
            decoded = self._convert_to_words(decoded)
        return decoded

if __name__ == '__main__':
    decoder = Decoder('abcde', 1, 2)
    print(decoder.wer("I love you",'I Love me'))

