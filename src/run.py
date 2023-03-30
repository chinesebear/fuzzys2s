from torch import nn
import torch
import datetime
from loguru import logger
from model import FNN
from loaddata import read_data,fcm
from setting import options, setting_info

def model_info(model):
    logger.info("model setting info:%s" %(setting_info()))
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("model total parameters: %d, trainable  parameters: %d " %(total_params,total_trainable_params))


def valid(model, valid_data):
    count = 0
    acc = 0
    total_loss = 0
    softmax = nn.Softmax(dim=-1)
    criterion = nn.CrossEntropyLoss()
    for features, target in valid_data:
        x = torch.tensor(features).to(options.device)
        target = torch.tensor(target).to(options.device)
        output = model(x)
        loss = criterion(output, target)
        output = softmax(output)
        predict = torch.argmax(output, dim = -1)
        count = count +1
        total_loss = total_loss +loss.item()
        if predict == target:
            acc = acc +1
    return total_loss/count, acc/count

def train():
    logger.add(options.base_path+'output/fnn-'+str(datetime.date.today()) +'.log')
    head,train_data, train_len,valid_data, valid_len, test_data, test_len= read_data()
    center,sigma = fcm(train_data, cluster_num= options.cluster_num, h= options.h)
    # center,sigma = "",""
    fnn = FNN(options.feature_num, options.rule_num, center,sigma ).to(options.device)
    optimizer = torch.optim.Adam(fnn.parameters(), lr=options.learning_rate, weight_decay=0)
    criterion = nn.CrossEntropyLoss()
    model_info(fnn)
    epoch = options.epoch
    for i in range(epoch):
        count = 0
        total_acc= 0
        total_bleu=0
        total_loss = 0
        for features,target in train_data:
            optimizer.zero_grad()
            x = torch.tensor(features).to(options.device)
            target = torch.tensor(target).to(options.device)
            output = fnn(x)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            count = count+ 1
            total_loss =total_loss + loss.item()
            if count %500 ==0:
                test_loss, acc = valid(fnn, valid_data)
                logger.info("epoch: %d, count: %d, train loss: %.4f, test loss: %.4f,acc: %.4f" %(i, count, total_loss/count, test_loss, acc))
                # print(dict(fnn.named_parameters()))
                # print(optimizer.state_dict())
                # for name, parms in fnn.named_parameters():
                #     print('-->name:', name)
                #     # print('-->para:', parms)
                #     print('-->grad_requirs:',parms.requires_grad)
                #     print('-->grad_value:',parms.grad)
train()