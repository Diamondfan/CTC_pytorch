#!/usr/bin/python
#encoding=utf-8

#train process for the model

from data_loader import myDataset, myCNNDataLoader, myDataLoader
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

RNN = {'nn.LSTM':nn.LSTM, 'nn.GRU': nn.GRU, 'nn.RNN':nn.RNN}
activate_f = {'relu':nn.ReLU, 'tanh':nn.Tanh, 'sigmoid':nn.Sigmoid}

parser = argparse.ArgumentParser(description='cnn_lstm_ctc')
parser.add_argument('--conf', default='../conf/cnn_lstm_ctc_setting.conf' , help='conf file with Argument of LSTM and training')
parser.add_argument('--log-dir', dest='log_dir', default='../log', help='log file for training')

def train(model, train_loader, loss_fn, optimizer, logger, add_cnn=True, print_every=20):
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
        nn.utils.clip_grad_norm(model.parameters(), 400)                #防止梯度爆炸或者梯度消失，限制参数范围       
        optimizer.step()
        
        if USE_CUDA:
            torch.cuda.synchronize()
        
        i += 1
    average_loss = total_loss / i
    print("Epoch done, average loss: %.4f" % average_loss)
    logger.info("Epoch done, average loss: %.4f" % average_loss)
    return average_loss

def dev(model, dev_loader, loss_fn, decoder, logger, add_cnn=True):
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
    #print(total_cer, total_tokens)
    acc = 1 - float(total_cer) / total_tokens
    average_loss = total_loss / i
    return acc*100, average_loss

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

def main():
    args = parser.parse_args()
    cf = ConfigParser.ConfigParser()
    try:
        cf.read(args.conf)
    except:
        print("conf file not exists")
    
    try:
        seed = cf.get('Training', 'seed')
        seed = long(seed)
    except:
        seed = torch.cuda.initial_seed()
    
    torch.manual_seed(seed)
    if USE_CUDA:
        torch.cuda.manual_seed_all(seed)
    
    logger = init_logger(os.path.join(args.log_dir, 'train_ctc_model.log'))
    
    #Define Model
    rnn_input_size = cf.getint('Model', 'rnn_input_size')
    rnn_hidden_size = cf.getint('Model', 'rnn_hidden_size')
    rnn_layers = cf.getint('Model', 'rnn_layers')
    rnn_type = RNN[cf.get('Model', 'rnn_type')]
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
    activation_function = activate_f[cf.get('CNN', 'activation_function')]
    
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

    model = CTC_Model(rnn_param=rnn_param, add_cnn=add_cnn, cnn_param=cnn_param,
                         num_class=num_class, drop_out=drop_out)
    #model.apply(xavier_uniform_init)
    for idx, m in enumerate(model.modules()):
        print(idx, m)
        break
    
    dataset = cf.get('Data', 'dataset')
    data_dir = cf.get('Data', 'data_dir')
    feature_type = cf.get('Data', 'feature_type')
    out_type = cf.get('Data', 'out_type')
    n_feats = cf.getint('Data', 'n_feats')
    mel = cf.getboolean('Data', 'mel')
    batch_size = cf.getint("Training", 'batch_size')
    
    #Data Loader
    train_dataset = myDataset(data_dir, data_set='train', feature_type=feature_type, out_type=out_type, n_feats=n_feats, mel=mel)
    dev_dataset = myDataset(data_dir, data_set="dev", feature_type=feature_type, out_type=out_type, n_feats=n_feats, mel=mel)
    if add_cnn:
        train_loader = myCNNDataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                            num_workers=4, pin_memory=False)
        dev_loader = myCNNDataLoader(dev_dataset, batch_size=batch_size, shuffle=False,
                                            num_workers=4, pin_memory=False)
    else:
        train_loader = myDataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                            num_workers=4, pin_memory=False)
        dev_loader = myDataLoader(dev_dataset, batch_size=batch_size, shuffle=False,
                                            num_workers=4, pin_memory=False)
    #decoder for dev set
    decoder = GreedyDecoder(dev_dataset.int2phone, space_idx=-1, blank_index=0)
    
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
    acc_best_true = 0
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
            adjust_rate_count = 0
            model_state = copy.deepcopy(model.state_dict())
            op_state = copy.deepcopy(optimizer.state_dict())
        elif (dev_loss < loss_best + end_adjust_acc):
            adjust_rate_count += 1
            if dev_loss < loss_best and dev_loss < loss_best_true:
                loss_best_true = dev_loss
                model_state = copy.deepcopy(model.state_dict())
                op_state = copy.deepcopy(optimizer.state_dict())
        else:
            adjust_rate_count = 10
        
        if acc > acc_best:
            acc_best = acc
            best_model_state = copy.deepcopy(model.state_dict())
            best_op_state = copy.deepcopy(optimizer.state_dict())
        
        '''
        #adjust learning rate by dev_acc
        if acc > (acc_best + end_adjust_acc):
            acc_best = acc
            adjust_rate_count = 0
            loss_best = dev_loss
            model_state = copy.deepcopy(model.state_dict())
            op_state = copy.deepcopy(optimizer.state_dict())
        elif (acc > acc_best - end_adjust_acc):
            adjust_rate_count += 1
            if acc > acc_best and acc > acc_best_true:
                acc_best_true = acc
                loss_best = dev_loss
                model_state = copy.deepcopy(model.state_dict())
                op_state = copy.deepcopy(optimizer.state_dict())
        else:
            adjust_rate_count = 0
        #torch.save(model.state_dict(), model_path_reject)
        '''

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
            #if acc_best < acc_best_true:
            #    acc_best = acc_best_true
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

    print("End training, best cv loss is: %.4f, acc is: %.4f" % (loss_best, acc_best))
    logger.info("End training, best loss acc is: %.4f, acc is: %.4f" % (loss_best, acc_best)) 
    model.load_state_dict(best_model_state)
    optimizer.load_state_dict(best_op_state)
    best_path = os.path.join(args.log_dir, 'best_model'+'_cv'+str(acc_best)+'.pkl')
    cf.set('Model', 'model_file', best_path)
    cf.write(open(args.conf, 'w'))
    params['epoch']=count

    torch.save(CTC_Model.save_package(model, optimizer=optimizer, epoch=params, loss_results=loss_results, dev_loss_results=dev_loss_results, dev_cer_results=dev_cer_results), best_path)

if __name__ == '__main__':
    main()
