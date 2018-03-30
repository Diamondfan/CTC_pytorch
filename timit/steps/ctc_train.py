#!/usr/bin/python
#encoding=utf-8

import os
import sys
import copy
import time 
import torch
import argparse
import numpy as np
import ConfigParser
import torch.nn as nn
from torch.autograd import Variable

from ctc_model import *
from warpctc_pytorch import CTCLoss
from ctcDecoder import GreedyDecoder
from data_loader import SpeechDataset, SpeechCNNDataLoader, SpeechDataLoader

supported_rnn = {'nn.LSTM':nn.LSTM, 'nn.GRU': nn.GRU, 'nn.RNN':nn.RNN}
supported_activate = {'relu':nn.ReLU, 'tanh':nn.Tanh, 'sigmoid':nn.Sigmoid}

parser = argparse.ArgumentParser(description='cnn_lstm_ctc')
parser.add_argument('--conf', default='../conf/cnn_lstm_ctc_setting.conf' , help='conf file with Argument of LSTM and training')
parser.add_argument('--log-dir', dest='log_dir', default='../log', help='log file for training')


def train(model, train_loader, loss_fn, optimizer, logger, add_cnn=True, print_every=20, USE_CUDA=True):
    '''训练一个epoch，即将整个训练集跑一遍
    Args:
        model         :  定义的网络模型
        train_loader  :  加载训练集的类对象
        loss_fn       :  损失函数，此处为CTCLoss
        optimizer     :  优化器类对象
        logger        :  日志类对象
        print_every   :  每20个batch打印一次loss
        USE_CUDA      :  是否使用GPU
    Returns:
        average_loss  :  一个epoch的平均loss
    '''
    model.train()
    
    total_loss = 0
    print_loss = 0
    i = 0
    for data in train_loader:
        inputs, targets, input_sizes, input_sizes_list, target_sizes = data
        batch_size = inputs.size(0)
        if not add_cnn:
            inputs = inputs.transpose(0, 1)
        
        inputs = Variable(inputs, requires_grad=False)
        targets = Variable(targets, requires_grad=False)
        target_sizes = Variable(target_sizes, requires_grad=False)

        if USE_CUDA:
            inputs = inputs.cuda()
        
        if not add_cnn:
            inputs = nn.utils.rnn.pack_padded_sequence(inputs, input_sizes_list)
        
        out = model(inputs)
        if add_cnn:
            max_length = out.size(0)
            input_sizes = Variable(input_sizes.mul_(int(max_length)).int(), requires_grad=False)
            input_sizes_list = [int(x*max_length) for x in input_sizes_list]
        else:
            input_sizes = Variable(input_sizes, requires_grad=False)
        
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
        nn.utils.clip_grad_norm(model.parameters(), 400)
        optimizer.step()
        i += 1
    average_loss = total_loss / i
    print("Epoch done, average loss: %.4f" % average_loss)
    logger.info("Epoch done, average loss: %.4f" % average_loss)
    return average_loss

def dev(model, dev_loader, loss_fn, decoder, logger, add_cnn=True, USE_CUDA=True):
    '''验证集的计算过程，与train()不同的是不需要反向传播过程，并且需要计算正确率
    Args:
        model       :   模型
        dev_loader  :   加载验证集的类对象
        loss_fn     :   损失函数
        decoder     :   解码类对象，即将网络的输出解码成文本
        logger      :   日志类对象
        USE_CUDA    :   是否使用GPU
    Returns:
        acc * 100    :   字符正确率，如果space不是一个标签的话，则为词正确率
        average_loss :   验证集的平均loss
    '''
    model.eval()
    total_cer = 0
    total_tokens = 0
    total_loss = 0
    i = 0

    for data in dev_loader:
        inputs, targets, input_sizes, input_sizes_list, target_sizes = data
        batch_size = inputs.size(0)
        if not add_cnn:
            inputs = inputs.transpose(0, 1)

        inputs = Variable(inputs, requires_grad=False)
        targets = Variable(targets, requires_grad=False)
        target_sizes = Variable(target_sizes, requires_grad=False)

        if USE_CUDA:
            inputs = inputs.cuda()
        
        if not add_cnn:
            inputs = nn.utils.rnn.pack_padded_sequence(inputs, input_sizes_list)

        out, probs = model(inputs, dev=True)
        
        if add_cnn:
            max_length = out.size(0)
            input_sizes = Variable(input_sizes.mul_(int(max_length)).int(), requires_grad=False)
            input_sizes_list = [int(x*max_length) for x in input_sizes_list]
        else:
            input_sizes = Variable(input_sizes, requires_grad=False)

        loss = loss_fn(out, targets, input_sizes, target_sizes)
        loss /= batch_size
        total_loss += loss.data[0]
        
        probs = probs.data.cpu()
        targets = targets.data
        target_sizes = target_sizes.data

        if decoder.space_idx == -1:
            total_cer += decoder.phone_word_error(probs, input_sizes_list, targets, target_sizes)[1]
        else:
            total_cer += decoder.phone_word_error(probs, input_sizes_list, targets, target_sizes)[0]
        total_tokens += sum(target_sizes)
        i += 1
    acc = 1 - float(total_cer) / total_tokens
    average_loss = total_loss / i
    return acc*100, average_loss

def init_logger(log_file):
    '''得到一个日志的类对象
    Args:
        log_file   :  日志文件名
    Returns:
        logger     :  日志类对象
    '''
    import logging
    from logging.handlers import RotatingFileHandler

    logger = logging.getLogger()
    hdl = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=10)
    formatter=logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
    hdl.setFormatter(formatter)
    logger.addHandler(hdl)
    logger.setLevel(logging.DEBUG)
    return logger

def add_weights_noise(m):
    for param in m.parameters():
        weight_noise = param.data.new(param.size()).normal_(0, 0.075)
        if USE_CUDA:
            weight_noise = weight_noise.cuda()
        param = torch.nn.parameter.Parameter(param.data + weight_noise)

def main():
    args = parser.parse_args()
    cf = ConfigParser.ConfigParser()
    try:
        cf.read(args.conf)
    except:
        print("conf file not exists")
        sys.exit(1)
    try:
        seed = cf.get('Training', 'seed')
        seed = long(seed)
    except:
        seed = torch.cuda.initial_seed()
        cf.set('Training', 'seed', seed)
        cf.write(open(args.conf, 'w'))
    
    USE_CUDA = cf.getboolean("Training", "use_cuda")
    torch.manual_seed(seed)
    if USE_CUDA:
        torch.cuda.manual_seed(seed)
    
    logger = init_logger(os.path.join(args.log_dir, 'train_ctc_model.log'))
    
    #Define Model
    rnn_input_size = cf.getint('Model', 'rnn_input_size')
    rnn_hidden_size = cf.getint('Model', 'rnn_hidden_size')
    rnn_layers = cf.getint('Model', 'rnn_layers')
    rnn_type = supported_rnn[cf.get('Model', 'rnn_type')]
    bidirectional = cf.getboolean('Model', 'bidirectional')
    batch_norm = cf.getboolean('Model', 'batch_norm')
    rnn_param = {"rnn_input_size":rnn_input_size, "rnn_hidden_size":rnn_hidden_size, "rnn_layers":rnn_layers, 
                    "rnn_type":rnn_type, "bidirectional":bidirectional, "batch_norm":batch_norm}
    
    num_class = cf.getint('Model', 'num_class')
    drop_out = cf.getfloat('Model', 'drop_out')
    add_cnn = cf.getboolean('Model', 'add_cnn')
    
    cnn_param = {}
    layers = cf.getint('CNN', 'layers')
    channel = eval(cf.get('CNN', 'channel'))
    kernel_size = eval(cf.get('CNN', 'kernel_size'))
    stride = eval(cf.get('CNN', 'stride'))
    padding = eval(cf.get('CNN', 'padding'))
    pooling = eval(cf.get('CNN', 'pooling'))
    batch_norm = cf.getboolean('CNN', 'batch_norm')
    activation_function = supported_activate[cf.get('CNN', 'activation_function')]
    
    cnn_param['batch_norm'] = batch_norm
    cnn_param['activate_function'] = activation_function
    cnn_param["layer"] = []
    for layer in range(layers):
        layer_param = [channel[layer], kernel_size[layer], stride[layer], padding[layer]]
        if pooling is not None:
            layer_param.append(pooling[layer])
        else:
            layer_param.append(None)
        cnn_param["layer"].append(layer_param)

    model = CTC_Model(rnn_param=rnn_param, add_cnn=add_cnn, cnn_param=cnn_param, num_class=num_class, drop_out=drop_out)
    
    for idx, m in enumerate(model.children()):
        print(idx, m)
        logger.info(str(idx) + "->" + str(m))
    
    dataset = cf.get('Data', 'dataset')
    data_dir = cf.get('Data', 'data_dir')
    feature_type = cf.get('Data', 'feature_type')
    out_type = cf.get('Data', 'out_type')
    n_feats = cf.getint('Data', 'n_feats')
    mel = cf.getboolean('Data', 'mel')
    batch_size = cf.getint("Training", 'batch_size')
    
    #Data Loader
    train_dataset = SpeechDataset(data_dir, data_set='train', feature_type=feature_type, out_type=out_type, n_feats=n_feats, mel=mel)
    dev_dataset = SpeechDataset(data_dir, data_set="dev", feature_type=feature_type, out_type=out_type, n_feats=n_feats, mel=mel)
    if add_cnn:
        train_loader = SpeechCNNDataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                            num_workers=4, pin_memory=False)
        dev_loader = SpeechCNNDataLoader(dev_dataset, batch_size=batch_size, shuffle=False,
                                            num_workers=4, pin_memory=False)
    else:
        train_loader = SpeechDataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                            num_workers=4, pin_memory=False)
        dev_loader = SpeechDataLoader(dev_dataset, batch_size=batch_size, shuffle=False,
                                            num_workers=4, pin_memory=False)
    #decoder for dev set
    decoder = GreedyDecoder(dev_dataset.int2class, space_idx=-1, blank_index=0)
        
    #Training
    init_lr = cf.getfloat('Training', 'init_lr')
    num_epoches = cf.getint('Training', 'num_epoches')
    end_adjust_acc = cf.getfloat('Training', 'end_adjust_acc')
    decay = cf.getfloat("Training", 'lr_decay')
    weight_decay = cf.getfloat("Training", 'weight_decay')
    
    params = { 'num_epoches':num_epoches, 'end_adjust_acc':end_adjust_acc, 'mel': mel, 'seed':seed,
                'decay':decay, 'learning_rate':init_lr, 'weight_decay':weight_decay, 'batch_size':batch_size,
                'feature_type':feature_type, 'n_feats': n_feats, 'out_type': out_type }
    print(params)
    
    if USE_CUDA:
        model = model.cuda()
    
    loss_fn = CTCLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay)

    #visualization for training
    from visdom import Visdom
    viz = Visdom()
    if add_cnn:
        title = dataset+' '+feature_type+str(n_feats)+' CNN_LSTM_CTC'
    else:
        title = dataset+' '+feature_type+str(n_feats)+' LSTM_CTC'

    opts = [dict(title=title+" Loss", ylabel = 'Loss', xlabel = 'Epoch'),
            dict(title=title+" Loss on Dev", ylabel = 'DEV Loss', xlabel = 'Epoch'),
            dict(title=title+' CER on DEV', ylabel = 'DEV CER', xlabel = 'Epoch')]
    viz_window = [None, None, None]
    
    count = 0
    learning_rate = init_lr
    loss_best = 1000
    loss_best_true = 1000
    adjust_rate_flag = False
    stop_train = False
    adjust_time = 0
    acc_best = 0
    start_time = time.time()
    loss_results = []
    dev_loss_results = []
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
        
        loss = train(model, train_loader, loss_fn, optimizer, logger, add_cnn=add_cnn, print_every=20)
        loss_results.append(loss)
        acc, dev_loss = dev(model, dev_loader, loss_fn, decoder, logger, add_cnn=add_cnn)
        print("loss on dev set is %.4f" % dev_loss)
        logger.info("loss on dev set is %.4f" % dev_loss)
        dev_loss_results.append(dev_loss)
        dev_cer_results.append(acc)
        
        #adjust learning rate by dev_loss
        if dev_loss < (loss_best - end_adjust_acc):
            loss_best = dev_loss
            loss_best_true = dev_loss
            #acc_best = acc
            adjust_rate_count = 0
            model_state = copy.deepcopy(model.state_dict())
            op_state = copy.deepcopy(optimizer.state_dict())
        elif (dev_loss < loss_best + end_adjust_acc):
            adjust_rate_count += 1
            if dev_loss < loss_best and dev_loss < loss_best_true:
                loss_best_true = dev_loss
                #acc_best = acc
                model_state = copy.deepcopy(model.state_dict())
                op_state = copy.deepcopy(optimizer.state_dict())
        else:
            adjust_rate_count = 10
        
        if acc > acc_best:
            acc_best = acc
            best_model_state = copy.deepcopy(model.state_dict())
            best_op_state = copy.deepcopy(optimizer.state_dict())

        print("adjust_rate_count:"+str(adjust_rate_count))
        print('adjust_time:'+str(adjust_time))
        logger.info("adjust_rate_count:"+str(adjust_rate_count))
        logger.info('adjust_time:'+str(adjust_time))

        if adjust_rate_count == 10:
            adjust_rate_flag = True
            adjust_time += 1
            adjust_rate_count = 0
            if loss_best > loss_best_true:
                loss_best = loss_best_true
            model.load_state_dict(model_state)
            optimizer.load_state_dict(op_state)

        if adjust_time == 8:
            stop_train = True
        
        time_used = (time.time() - start_time) / 60
        print("epoch %d done, cv acc is: %.4f, time_used: %.4f minutes" % (count, acc, time_used))
        logger.info("epoch %d done, cv acc is: %.4f, time_used: %.4f minutes" % (count, acc, time_used))
        
        x_axis = range(count)
        y_axis = [loss_results[0:count], dev_loss_results[0:count], dev_cer_results[0:count]]
        for x in range(len(viz_window)):
            if viz_window[x] is None:
                viz_window[x] = viz.line(X = np.array(x_axis), Y = np.array(y_axis[x]), opts = opts[x],)
            else:
                viz.line(X = np.array(x_axis), Y = np.array(y_axis[x]), win = viz_window[x], update = 'replace',)

    print("End training, best dev loss is: %.4f, acc is: %.4f" % (loss_best, acc_best))
    logger.info("End training, best dev loss acc is: %.4f, acc is: %.4f" % (loss_best, acc_best)) 
    model.load_state_dict(best_model_state)
    optimizer.load_state_dict(best_op_state)
    best_path = os.path.join(args.log_dir, 'best_model'+'_dev'+str(acc_best)+'.pkl')
    cf.set('Model', 'model_file', best_path)
    cf.write(open(args.conf, 'w'))
    params['epoch']=count

    torch.save(CTC_Model.save_package(model, optimizer=optimizer, epoch=params, loss_results=loss_results, dev_loss_results=dev_loss_results, dev_cer_results=dev_cer_results), best_path)

if __name__ == '__main__':
    main()
