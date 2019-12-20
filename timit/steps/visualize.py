#!/usr/bin/python
#encoding=utf-8

#old version, need updated
from data_loader import myDataset
from data_loader import myDataLoader, myCNNDataLoader
from model import *
from ctcDecoder import GreedyDecoder, BeamDecoder
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import visdom


def test():
    model_path = '../log/exp_cnn_lstm_ctc_spectrum201/exp_cnn3*41_3*21_4lstm_ctc_Melspectrum_stride_1_2/exp2_82.1483/best_model_cv80.8660423723.pkl'
    package = torch.load(model_path)
    data_dir = '../data_prepare/data'
    rnn_param = package["rnn_param"]
    add_cnn = package["add_cnn"]
    cnn_param = package["cnn_param"]
    num_class = package["num_class"]
    feature_type = package['epoch']['feature_type']
    n_feats = package['epoch']['n_feats']
    out_type = package['epoch']['out_type']
    drop_out = package['_drop_out']
    try:
        mel = package['epoch']['mel']
    except:
        mel = False
    #weight_decay = package['epoch']['weight_decay']
    #print(weight_decay)

    decoder_type = 'Greedy'

    test_dataset = myDataset(data_dir, data_set='train', feature_type=feature_type, out_type=out_type, n_feats=n_feats, mel=mel)
    
    model = CTC_Model(rnn_param=rnn_param, add_cnn=add_cnn, cnn_param=cnn_param, num_class=num_class, drop_out=drop_out)
        
    if add_cnn:
        test_loader = myCNNDataLoader(test_dataset, batch_size=1, shuffle=False,
                                        num_workers=4, pin_memory=False)
    else:
        test_loader = myDataLoader(test_dataset, batch_size=1, shuffle=False,
                                        num_workers=4, pin_memory=False)
    
    model.load_state_dict(package['state_dict'])
    model.eval()
    
    if USE_CUDA:
        model = model.cuda()

    if decoder_type == 'Greedy':
        decoder  = GreedyDecoder(test_dataset.int2phone, space_idx=-1, blank_index=0)
    else:
        decoder = BeamDecoder(test_dataset.int2phone)    
    
    import pickle
    f = open('../decode_map_48-39/map_dict.pkl', 'rb')
    map_dict = pickle.load(f)
    f.close()
    print(map_dict)

    vis = visdom.Visdom(env='fan')
    legend = []
    for i in range(49):
        legend.append(test_dataset.int2phone[i])
    
    for data in test_loader:
        inputs, target, input_sizes, input_size_list, target_sizes = data 
        if not add_cnn:
            inputs = inputs.transpose(0,1)

        inputs = Variable(inputs, volatile=True, requires_grad=False)
        if USE_CUDA:
            inputs = inputs.cuda()
        
        if not add_cnn:
            inputs = nn.utils.rnn.pack_padded_sequence(inputs, input_size_list)
    
        probs, visual = model(inputs, visualize=True)
        probs = probs.data.cpu()
        
        if add_cnn:
            max_length = probs.size(0)
            input_size_list = [int(x*max_length) for x in input_size_list]

        decoded = decoder.decode(probs, input_size_list)
        targets = decoder._unflatten_targets(target, target_sizes)
        labels = decoder._process_strings(decoder._convert_to_strings(targets))
        
        for x in range(len(labels)):
            label = labels[x].strip().split(' ')
            for i in range(len(label)):
                label[i] = map_dict[label[i]]
            labels[x] = ' '.join(label)
            decode = decoded[x].strip().split(' ')
            for i in range(len(decode)):
                decode[i] = map_dict[decode[i]]
            decoded[x] = ' '.join(decode)

        for x in range(len(labels)):
            print("origin: "+ labels[x])
            print("decoded: "+ decoded[x])
        
        if add_cnn:
            spectrum_inputs = visual[0][0][0].transpose(0, 1).data.cpu()
            opts = dict(title=labels[0], xlabel="frame", ylabel='spectrum')
            vis.heatmap(spectrum_inputs, opts = opts)
        
            opts = dict(title=labels[0], xlabel="frame", ylabel='feature_after_cnn')
            after_cnn = visual[1][0][0].transpose(0, 1).data.cpu()
            vis.heatmap(after_cnn, opts = opts)
        
            opts = dict(title=labels[0], xlabel="frame", ylabel='feature_before_rnn')
            before_rnn = visual[2].transpose(0, 1)[0].transpose(0, 1).data.cpu()
            vis.heatmap(before_rnn, opts=opts)
        
            show_prob = visual[3].transpose(0, 1)[0].data.cpu()
            line_opts = dict(title=decoded[0], xlabel="frame", ylabel="probability", legend=legend)
            x = show_prob.size()[0]
            vis.line(show_prob.numpy(), X=np.array(range(x)), opts=line_opts)
        else:
            spectrum_inputs = visual[0][0][0].transpose(0, 1).data.cpu()
            opts = dict(title=labels[0], xlabel="frame", ylabel='spectrum')
            vis.heatmap(spectrum_inputs, opts = opts)

            show_prob = visual[1].transpose(0, 1)[0].data.cpu()
            line_opts = dict(title=decoded[0], xlabel="frame", ylabel="probability", legend=legend)
            x = show_prob.size()[0]
            vis.line(show_prob.numpy(), X=np.array(range(x)), opts=line_opts)
        break

if __name__ == "__main__":
    test()



