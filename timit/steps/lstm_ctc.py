#!/usr/bin/python
#encoding=utf-8

#train process for the model

from data_loader import myDataset, myDataLoader
from model import *
from ctcDecoder import GreedyDecoder
from warpctc_pytorch import CTCLoss
import torch
import torch.nn as nn
from torch.autograd import Variable
import time 
import numpy as np
import argparse
import ConfigParser
import os
import copy


def train(model, train_loader, loss_fn, optimizer, logger, print_every=20):
    model.train()
    
    total_loss = 0
    print_loss = 0
    i = 0
    for data in train_loader:
        inputs, targets, input_sizes, input_sizes_list, target_sizes = data
        batch_size = inputs.size(0)
        inputs = inputs.transpose(0,1)
        
        inputs = Variable(inputs, requires_grad=False)
        targets = Variable(targets, requires_grad=False)
        input_sizes = Variable(input_sizes, requires_grad=False)
        target_sizes = Variable(target_sizes, requires_grad=False)
        
        if USE_CUDA:
            inputs = inputs.cuda()
        
        #pack padded input sequence
        inputs = nn.utils.rnn.pack_padded_sequence(inputs, input_sizes_list)
        out = model(inputs)

        loss = loss_fn(out, targets, input_sizes, target_sizes)
        loss /= batch_size
        print_loss += loss.data[0]

        if (i + 1) % print_every == 0:
            print('batch = %d, loss = %.4f' % (i+1, print_loss / print_every))
            logger.debug('batch = %d, loss = %.4f' % (i+1, print_loss / print_every))
            print_loss = 0
        
        total_loss += loss.data[0]
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), 400)                #防止梯度爆炸或者梯度消失，限制参数范围       
        optimizer.step()
        i += 1
    average_loss = total_loss / i
    print("Epoch done, average loss: %.4f" % average_loss)
    logger.info("Epoch done, average loss: %.4f" % average_loss)
    return average_loss

def dev(model, dev_loader, decoder, logger):
    model.eval()
    total_cer = 0
    total_tokens = 0

    for data in dev_loader:
        inputs, targets, input_sizes, input_sizes_list, target_sizes =data
        batch_size = inputs.size(1)
        inputs = inputs.transpose(0, 1)
        
        inputs = Variable(inputs, volatile=True, requires_grad=False)

        if USE_CUDA:
            inputs = inputs.cuda()

        inputs = nn.utils.rnn.pack_padded_sequence(inputs, input_sizes_list)
        probs = model(inputs)
        
        probs = probs.data.cpu()
        if decoder.space_idx == -1:
            total_cer += decoder.phone_word_error(probs, input_sizes_list, targets, target_sizes)[1]
        else:
            total_cer += decoder.phone_word_error(probs, input_sizes_list, targets, target_sizes)[0]
        total_tokens += sum(target_sizes)
    acc = 1 - float(total_cer) / total_tokens
    return acc*100

def init_logger(log_file):
    import logging
    from logging.handlers import RotatingFileHandler

    logger = logging.getLogger()
    hdl = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=10)
    formatter=logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
    hdl.setFormatter(formatter)
    logger.addHandler(hdl)
    logger.setLevel(logging.DEBUG)
    return logger

RNN = {'nn.LSTM':nn.LSTM, 'nn.GRU': nn.GRU, 'nn.RNN':nn.RNN}
parser = argparse.ArgumentParser(description='lstm_ctc')
parser.add_argument('--conf', default='../conf/lstm_ctc_setting.conf' , help='conf file with Argument of LSTM and training')
parser.add_argument('--log-dir', dest='log_dir', default='../log/', help='log dir for training')

def main():
    args = parser.parse_args()
    cf = ConfigParser.ConfigParser()
    try:
        cf.read(args.conf)
    except:
        print("conf file not exists")
    
    logger = init_logger(os.path.join(args.log_dir, 'train_lstm_ctc.log'))
    dataset = cf.get('Data', 'dataset')
    data_dir = cf.get('Data', 'data_dir')
    feature_type = cf.get('Data', 'feature_type')
    out_type = cf.get('Data', 'out_type')
    n_feats = cf.getint('Data', 'n_feats')
    batch_size = cf.getint("Training", 'batch_size')
    
    #Data Loader
    train_dataset = myDataset(data_dir, data_set='train', feature_type=feature_type, out_type=out_type, n_feats=n_feats)
    train_loader = myDataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                        num_workers=4, pin_memory=False)
    dev_dataset = myDataset(data_dir, data_set="dev", feature_type=feature_type, out_type=out_type, n_feats=n_feats)
    dev_loader = myDataLoader(dev_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=False)
    
    #decoder for dev set
    decoder = GreedyDecoder(dev_dataset.int2phone, space_idx=-1, blank_index=0)
    
    #Define Model
    rnn_input_size = cf.getint('Model', 'rnn_input_size')
    rnn_hidden_size = cf.getint('Model', 'rnn_hidden_size')
    rnn_layers = cf.getint('Model', 'rnn_layers')
    rnn_type = RNN[cf.get('Model', 'rnn_type')]
    bidirectional = cf.getboolean('Model', 'bidirectional')
    batch_norm = cf.getboolean('Model', 'batch_norm')
    num_class = cf.getint('Model', 'num_class')
    drop_out = cf.getfloat('Model', 'num_class')
    model = CTC_RNN(rnn_input_size=rnn_input_size, rnn_hidden_size=rnn_hidden_size, rnn_layers=rnn_layers, 
                        rnn_type=rnn_type, bidirectional=bidirectional, batch_norm=batch_norm, 
                        num_class=num_class, drop_out=drop_out)
    #model.apply(xavier_uniform_init)
    print(model.name)
    if USE_CUDA:
        model = model.cuda()
    
    #Training
    init_lr = cf.getfloat('Training', 'init_lr')
    num_epoches = cf.getint('Training', 'num_epoches')
    end_adjust_acc = cf.getfloat('Training', 'end_adjust_acc')
    decay = cf.getfloat("Training", 'lr_decay')
    weight_decay = cf.getfloat("Training", 'weight_decay')
    params = { 'num_epoches':num_epoches, 'end_adjust_acc':end_adjust_acc,
            'decay':decay, 'learning_rate':init_lr, 'weight_decay':weight_decay, 'batch_size':batch_size,
            'feature_type':feature_type, 'n_feats': n_feats, 'out_type': out_type }
    
    loss_fn = CTCLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay)
    
    #visualization for training
    from visdom import Visdom
    viz = Visdom()
    title = dataset+' '+feature_type+str(n_feats)+' LSTM_CTC'
    opts = [dict(title=title+" Loss", ylabel = 'Loss', xlabel = 'Epoch'),
            dict(title=title+" CER on Train", ylabel = 'CER', xlabel = 'Epoch'),
            dict(title=title+' CER on DEV', ylabel = 'DEV CER', xlabel = 'Epoch')]
    viz_window = [None, None, None]
    
    count = 0
    learning_rate = init_lr
    acc_best = -100
    adjust_rate_flag = False
    stop_train = False
    adjust_time = 0
    start_time = time.time()
    loss_results = []
    training_cer_results = []
    dev_cer_results = []
    
    while not stop_train:
        if count >= num_epoches:
            break
        count += 1
        
        if adjust_rate_flag:
            learning_rate *= decay
            adjust_rate_flag = False
            for param in optimizer.param_groups:
                param['lr'] *= decay
        
        print("Start training epoch: %d, learning_rate: %.5f" % (count, learning_rate))
        logger.info("Start training epoch: %d, learning_rate: %.5f" % (count, learning_rate))
        
        loss = train(model, train_loader, loss_fn, optimizer, logger)
        loss_results.append(loss)
        cer = dev(model, train_loader, decoder, logger)
        print("cer on training set is %.4f" % cer)
        logger.info("cer on training set is %.4f" % cer)
        training_cer_results.append(cer)
        acc = dev(model, dev_loader, decoder, logger)
        dev_cer_results.append(acc)
        
        #model_path_accept = './log/epoch'+str(count)+'_lr'+str(learning_rate)+'_cv'+str(acc)+'.pkl'
        #model_path_reject = './log/epoch'+str(count)+'_lr'+str(learning_rate)+'_cv'+str(acc)+'_rejected.pkl'
        
        if acc > (acc_best + end_adjust_acc):
            acc_best = acc
            adjust_rate_count = 0
            model_state = copy.deepcopy(model.state_dict())
            op_state = copy.deepcopy(optimizer.state_dict())
        elif (acc > acc_best - end_adjust_acc):
            adjust_rate_count += 1
            if acc > acc_best and adjust_rate_count == 10:
                acc_best = acc
                model_state = copy.deepcopy(model.state_dict())
                op_state = copy.deepcopy(optimizer.state_dict())
        else:
            adjust_rate_count = 0
        
        #torch.save(model.state_dict(), model_path_reject)
        print("adjust_rate_count:"+str(adjust_rate_count))
        print('adjust_time:'+str(adjust_time))
        logger.info("adjust_rate_count:"+str(adjust_rate_count))
        logger.info('adjust_time:'+str(adjust_time))

        if adjust_rate_count == 10:
            adjust_rate_flag = True
            adjust_time += 1
            adjust_rate_count = 0

        if adjust_time == 8:
            model.load_state_dict(model_state)
            optimizer.load_state_dict(op_state)
            stop_train = True
        
        time_used = (time.time() - start_time) / 60
        print("epoch %d done, cv acc is: %.4f, time_used: %.4f minutes" % (count, acc, time_used))
        logger.info("epoch %d done, cv acc is: %.4f, time_used: %.4f minutes" % (count, acc, time_used))
        x_axis = range(count)
        y_axis = [loss_results[0:count], training_cer_results[0:count], dev_cer_results[0:count]]
        for x in range(len(viz_window)):
            if viz_window[x] is None:
                viz_window[x] = viz.line(X = np.array(x_axis), Y = np.array(y_axis[x]), opts = opts[x],)
            else:
                viz.line(X = np.array(x_axis), Y = np.array(y_axis[x]), win = viz_window[x], update = 'replace',)

    print("End training, best cv acc is: %.4f" % acc_best)
    logger.info("End training, best cv acc is: %.4f" % acc_best)
    best_path = os.path.join(args.log_dir, 'best_model'+'_cv'+str(acc_best)+'.pkl')
    cf.set('Model', 'model_file', best_path)
    cf.write(open(args.conf, 'w'))
    params['epoch']=count
    torch.save(CTC_RNN.save_package(model, optimizer=optimizer, epoch=params, loss_results=loss_results, training_cer_results=training_cer_results, dev_cer_results=dev_cer_results), best_path)
    

if __name__ == '__main__':
    main()
