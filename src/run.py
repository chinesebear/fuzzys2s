from torch import nn
import torch
import torch.nn.functional as F
import datetime
import numpy as np
import random
import math
from loguru import logger
from model import FuzzyS2S
from loaddata import read_data,fcm, rescaling, gen_features_map
from setting import options, setting_info

def model_info(model):
    logger.info("[model info]")
    logger.info("%s" %(setting_info()))
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("total parameters: %d, trainable  parameters: %d " %(total_params,total_trainable_params))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def padding1d(input, limit_size):
    size = limit_size - len(input)
    output = F.pad(input, (0,size), mode="constant", value=options.PAD_token).long()
    return output

def truncat1d(input, limit_size):
    return input[0:limit_size]

def align1d(input, limit_size):
    if len(input) < limit_size:
        return padding1d(input, limit_size)
    elif len(input) > limit_size:
        return truncat1d(input, limit_size)
    else:
        return input

def getlen1d(input):
    """get sentence length without padding"""
    count=0
    for i in range(len(input)):
        if input[i] == options.EOS_token or (input[i] == options.PAD_token and input[i+1] == options.PAD_token and input[i+2] == options.PAD_token):
            return count
        else:
            count += 1
    return count

def calc_bp(candidate, reference):
    r = getlen1d(candidate)
    c = getlen1d(reference)
    bp = 0
    if c > r :
        bp = 1
    else:
        bp = math.exp(1 - r/c)
    return bp

def count_ngram(ngram, sentence, n):
    limit = len(sentence) - n + 1
    count = 0
    for i in range(limit):
        item = str(sentence[i:i+n])
        if ngram == item:
            count+=1
    return count

def calc_ngram_score(candidate, reference, n):
    count = 0
    candidate_len = getlen1d(candidate)
    reference_len = getlen1d(reference)
    candidate_ngram_dict = {}
    limit = candidate_len -n + 1
    for i in range(limit):
        ngram = str(candidate[i:i+n])
        if ngram in candidate_ngram_dict.keys():
            candidate_ngram_dict[ngram] += 1
        else:
            candidate_ngram_dict[ngram] = 1
    hc = sum(candidate_ngram_dict.values())
    limit = reference_len - n + 1
    min_hc_hs = 0
    for ngram in candidate_ngram_dict.keys():
        count = count_ngram(ngram, reference, n)
        min_hc_hs += min(count, candidate_ngram_dict[ngram])
    if min_hc_hs == 0 or hc == 0:
        return 0.0
    Pn = min_hc_hs / hc
    return Pn

def calcBLEU(candidate, reference):
    N = 4
    log_Pn = [0.0,0.0,0.0,0.0]
    for i in range(N):
        Pn = calc_ngram_score(candidate, reference, i+1)
        if Pn == 0:
            return 0.0
        else:
            log_Pn[i] = math.log(Pn)
    bp = calc_bp(candidate, reference)
    bleu = bp * math.exp(sum(log_Pn) / N)
    return bleu

def countequal1d(a, b, limit):
    count = 0
    for i in range(limit):
        if a[i] == b[i]:
            count += 1
    return count

def calcACC(predict, target, length):
    correct = countequal1d(predict, target, length)
    total = length
    acc = correct / total
    return acc

def calcPPL(loss):
    ppl = math.exp(min(loss, 100.0))
    return ppl

def evaluate(output, target, target_len):
    # print("output:",output.shape,",target:",target.shape)
    predict  = torch.argmax(output,dim=-1)
    predict = align1d(predict, target_len)
    acc = calcACC(predict, target, target_len)
    reference = target.tolist()
    candidate = predict.tolist()
    bleu = calcBLEU(candidate, reference)
    return acc,bleu

def valid(model, valid_data):
    eos = torch.ones(50, 1).long() #options.EOS_token = 1
    bos = torch.zeros(50,1).long() #options.SOS_token= 0
    src_with_eos = torch.cat((valid_data, eos), 1)
    tgt_with_bos = torch.cat((bos,valid_data), 1)
    tgt_with_eos = torch.cat((valid_data, eos), 1).to(options.device)
    src_with_eos_map = rescaling(gen_features_map(src_with_eos),0,1)
    tgt_with_bos_map = rescaling(gen_features_map(tgt_with_bos),0,1)
    tgt_with_eos_map = rescaling(gen_features_map(tgt_with_eos),0,1)
    count = 0
    total_acc = 0
    total_bleu = 0
    total_loss = 0
    softmax = nn.Softmax(dim=-1)
    criterion = nn.CrossEntropyLoss()
    for i in range(len(valid_data)):
        src = src_with_eos_map[i].to(options.device)
        tgt = tgt_with_bos_map[i].to(options.device)
        target = tgt_with_bos[i].to(options.device)
        output = model(src, tgt)
        loss = criterion(output, target)
        count = count+ 1
        total_loss =total_loss + loss.item()
        output = softmax(output)
        acc,bleu = evaluate(output, target, len(target))
        total_acc = total_acc + acc
        total_bleu = total_bleu + bleu
    return total_loss/count, total_acc/count,total_bleu/count


def copy_task():
    logger.add(options.base_path+'output/fnn-'+str(datetime.date.today()) +'.log')
    setup_seed(options.seed_id)
    eos = torch.ones(1100, 1).long() #options.EOS_token = 1
    bos = torch.zeros(1100,1).long() #options.SOS_token= 0
    data = torch.randint(4,14, (1100, 10))
    data_len = len(data)
    src_with_eos = torch.cat((data, eos), 1)
    tgt_with_bos = torch.cat((bos,data), 1)
    tgt_with_eos = torch.cat((data, eos), 1)
    src_with_eos_map = rescaling(gen_features_map(src_with_eos),0,1)
    tgt_with_bos_map = rescaling(gen_features_map(tgt_with_bos),0,1)
    tgt_with_eos_map = rescaling(gen_features_map(tgt_with_eos),0,1)
    data_set = torch.cat((src_with_eos_map, tgt_with_bos_map), 0)
    feature_in = 4
    feature_out = 15
    rule_num = options.rule_num
    center,sigma = fcm(data_set, cluster_num= rule_num, h= options.h)
    model = FuzzyS2S(feature_in, feature_out, rule_num, center, sigma, center, sigma).to(options.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=options.learning_rate, weight_decay=0)
    criterion = nn.CrossEntropyLoss()
    model_info(model)
    for epoch in range(options.epoch):
        count = 0
        total_loss = 0
        for i in range(data_len-100):
            optimizer.zero_grad()
            src = src_with_eos_map[i].to(options.device)
            tgt = tgt_with_bos_map[i].to(options.device)
            target = tgt_with_bos[i].to(options.device)
            output = model(src, tgt)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            count = count+ 1
            total_loss =total_loss + loss.item()
            if count %100 ==0:
                valid_loss, acc, bleu = valid(model, data[1000:1050])
                logger.info("epoch: %d, count: %d, train loss: %.4f, valid loss: %.4f, acc: %.4f, bleu:%.4f" %(epoch, count, total_loss/count, valid_loss,acc, bleu))

                # for name, parms in model.named_parameters():
                    # print('-->name:', name)
                    # # print('-->para:', parms)
                    # print('-->grad_requirs:',parms.requires_grad)
                    # print('-->grad_value:',parms.grad)


copy_task()


