from torch import nn
import torch
import torch.nn.functional as F
import datetime
import numpy as np
import random
import math
from loguru import logger
from model import FuzzyS2S,TransformerModel
from loaddata import read_data,fcm, gen_sen_feature_map_with_rescaling,combine_token_feature_map,del_eos,del_sos,insert_sos,attach_eos
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
    # count=0
    # for i in range(len(input)):
    #     if input[i] == options.EOS or (input[i] == options.PAD and input[i+1] == options.PAD and input[i+2] == options.PAD):
    #         return count
    #     else:
    #         count += 1
    # return count
    return len(input)

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

def valid(model, valid_data, vocab_src, vocab_tgt):
    count = 0
    total_acc = 0
    total_bleu = 0
    total_loss = 0
    softmax = nn.Softmax(dim=-1)
    criterion = nn.CrossEntropyLoss()
    for src, tgt in valid_data:
        src_with_eos = del_sos(src)
        tgt_with_sos = del_eos(tgt)
        tgt_with_eos = del_sos(tgt)
        src_with_eos = torch.tensor(gen_sen_feature_map_with_rescaling(vocab_src, src_with_eos)).to(options.device)
        tgt_with_sos = torch.tensor(gen_sen_feature_map_with_rescaling(vocab_tgt, tgt_with_sos)).to(options.device)
        tgt_with_eos = torch.tensor(tgt_with_eos).to(options.device)
        output = model(src_with_eos, tgt_with_sos)
        loss = criterion(output, tgt_with_eos)
        output = softmax(output)
        acc , bleu = evaluate(output , tgt_with_eos, len(tgt_with_eos))
        count = count+ 1
        total_loss =total_loss + loss.item()
        total_acc = total_acc +acc
        total_bleu = total_bleu + bleu
    return total_loss/count, total_acc/count,total_bleu/count

def s2s_task():
    logger.add(options.base_path+'output/fuzzys2s-'+str(datetime.date.today()) +'.log')
    setup_seed(options.seed_id)
    train_data, valid_data, test_data, vocab_src, vocab_tgt = read_data("copy")
    feature_in = options.feature_num
    feature_out = vocab_tgt.n_words
    rule_num = options.rule_num
    token_feature_map_src, token_feature_map_tgt = combine_token_feature_map(train_data, valid_data, test_data, vocab_src, vocab_tgt)
    logger.info("src token clustering")
    center_src,sigma_src = fcm(token_feature_map_src, cluster_num= rule_num, h= options.h)
    logger.info("tgt token clustering")
    center_tgt,sigma_tgt = fcm(token_feature_map_tgt, cluster_num= rule_num, h= options.h)
    model = FuzzyS2S(feature_in, feature_out, rule_num, center_src, sigma_src, center_tgt, sigma_tgt).to(options.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=options.learning_rate, weight_decay=0)
    criterion = nn.CrossEntropyLoss()
    model_info(model)
    for epoch in range(options.epoch):
        count = 0
        total_loss = 0
        for src, tgt in train_data:
            src_with_eos = del_sos(src)
            tgt_with_sos = del_eos(tgt)
            tgt_with_eos = del_sos(tgt)
            src_with_eos = torch.tensor(gen_sen_feature_map_with_rescaling(vocab_src, src_with_eos)).to(options.device)
            tgt_with_sos = torch.tensor(gen_sen_feature_map_with_rescaling(vocab_tgt, tgt_with_sos)).to(options.device)
            tgt_with_eos = torch.tensor(tgt_with_eos).to(options.device)
            optimizer.zero_grad()
            output = model(src_with_eos, tgt_with_sos)
            loss = criterion(output, tgt_with_eos)
            loss.backward()
            optimizer.step()
            count = count+ 1
            total_loss =total_loss + loss.item()
            if count % 100 ==0:
                valid_loss, acc, bleu = valid(model, valid_data,vocab_src, vocab_tgt)
                logger.info("epoch: %d, count: %d, train loss: %.4f, valid loss: %.4f, acc: %.4f, bleu:%.4f" %(epoch, count, total_loss/count, valid_loss,acc, bleu))

                # for name, parms in model.named_parameters():
                #     # encoder.rfs_block.center
                #     # encoder.rfs_block.sigma
                #     # encoder.rfs_block.recurrent_weight
                #     #
                #     # decoder.rfs_block.center
                #     # decoder.rfs_block.sigma
                #     # decoder.rfs_block.recurrent_weight
                #     #
                #     # decoder.fc.weight
                #     # decoder.fc.bias
                #     if name == "decoder.mlp.fc.weight":
                #         print('-->name:', name)
                #         # print('-->para:', parms)
                #         print('-->grad_requirs:',parms.requires_grad)
                #         print('-->grad_value:',parms.grad)

def valid_trans(model, valid_data, vocab_src, vocab_tgt):
    count = 0
    total_acc = 0
    total_bleu = 0
    total_loss = 0
    softmax = nn.Softmax(dim=-1)
    criterion = nn.CrossEntropyLoss()
    for src, tgt in valid_data:
        src_with_eos = attach_eos(src)
        tgt_with_sos = insert_sos(tgt)
        tgt_with_eos = attach_eos(tgt)
        src_mask = model._generate_square_subsequent_mask(len(src_with_eos))
        tgt_mask = model._generate_square_subsequent_mask(len(tgt_with_sos))
        output = model(src_with_eos, tgt_with_sos,src_mask, tgt_mask).squeeze()
        loss = criterion(output, tgt_with_eos)
        output = softmax(output)
        acc , bleu = evaluate(output , tgt_with_eos, len(tgt_with_eos))
        count = count+ 1
        total_loss =total_loss + loss.item()
        total_acc = total_acc +acc
        total_bleu = total_bleu + bleu
    return total_loss/count, total_acc/count,total_bleu/count

def trans_task():
    logger.add(options.base_path+'output/transformer-'+str(datetime.date.today()) +'.log')
    setup_seed(options.seed_id)
    train_data, valid_data, test_data, vocab_src, vocab_tgt = read_data("copy")
    model = TransformerModel(vocab_tgt.n_words, 128,16,128,3,0.1).to(options.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=options.learning_rate, weight_decay=0)
    criterion = nn.CrossEntropyLoss()
    model_info(model)
    for epoch in range(options.epoch):
        count = 0
        total_loss = 0
        for src, tgt in train_data:
            src_with_eos = attach_eos(src)
            tgt_with_sos = insert_sos(tgt)
            tgt_with_eos = attach_eos(tgt)
            src_mask = model._generate_square_subsequent_mask(len(src_with_eos))
            tgt_mask = model._generate_square_subsequent_mask(len(tgt_with_sos))
            optimizer.zero_grad()
            output = model(src_with_eos, tgt_with_sos,src_mask, tgt_mask).squeeze()
            loss = criterion(output, tgt_with_eos)
            loss.backward()
            optimizer.step()
            count = count+ 1
            total_loss =total_loss + loss.item()
            if count % 100 == 0:
                valid_loss, acc, bleu = valid_trans(model, valid_data,vocab_src, vocab_tgt)
                logger.info("epoch: %d, count: %d, train loss: %.4f, valid loss: %.4f, acc: %.4f, bleu:%.4f" %(epoch, count, total_loss/count, valid_loss,acc, bleu))

                # for name, parms in model.named_parameters():
                #     # encoder.rfs_block.center
                #     # encoder.rfs_block.sigma
                #     # encoder.rfs_block.recurrent_weight
                #     #
                #     # decoder.rfs_block.center
                #     # decoder.rfs_block.sigma
                #     # decoder.rfs_block.recurrent_weight
                #     #
                #     # decoder.fc.weight
                #     # decoder.fc.bias
                #     if name == "decoder.fc.weight" or name == "decoder.fc.bias":
                #         print('-->name:', name)
                #         # print('-->para:', parms)
                #         print('-->grad_requirs:',parms.requires_grad)
                #         print('-->grad_value:',parms.grad)


s2s_task()


