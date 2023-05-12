from torch import nn
import torch
import torch.nn.functional as F
from torchtext.data import get_tokenizer
import datetime
import numpy as np
import random
import math
from loguru import logger
from model import FuzzyS2S,TransformerModel
from loaddata import read_data,fcm, gen_sen_feature_map,combine_sen_feature_map,insert_sos,attach_eos,get_base_tokenizer
from setting import options, setting_info
import os
from evaluator import Evaluator
from tqdm import tqdm

def model_info(model):
    logger.info("[model %s]" %(model.name))
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

def savemodel(model,file):
    torch.save(model.state_dict(), options.model_parameter_path+file+".pth")
    logger.info("save %s model parameters done." %(file))

def loadmodel(model, file):
    if os.path.exists(options.model_parameter_path+file+".pth"):
        model.load_state_dict(torch.load(options.model_parameter_path+file+".pth"))
        model.eval()
        logger.info("load %s model parameters done." %(file))

def tensor2string(input_lang, source):
   output =  [input_lang.index2word[idx.item()] for idx in source]
   outstr = ''.join(x + ' ' for x in output)
   return outstr

def idx2word(source, vocab):
   word_list =  [vocab.index2word[idx] for idx in source]
   return word_list

def idx2string(source, vocab):
   word_list =  [vocab.index2word[idx] for idx in source]
   string_out = " ".join(word for word in word_list)
   return string_out


def predict(model, test_data, vocab_src, vocab_tgt, evaluator):
    count = 0
    total_acc = 0
    total_bleu = 0
    total_loss = 0
    total_ppl = 0
    total_precision = 0
    total_recall =0
    total_f1 = 0
    total_mae = 0
    total_mape = 0
    total_smape = 0
    total_rouge = np.array([0,0,0,0])
    total_meteor = 0
    softmax = nn.Softmax(dim=-1)
    criterion = nn.CrossEntropyLoss()
    for src, tgt in test_data:
        if len(src) > options.sen_len_max or len(tgt) > options.sen_len_max:
                continue
        if len(src) == 7 or len(src) == 7:
                continue
        # src_with_eos = attach_eos(src)
        # tgt_with_sos = torch.tensor([options.SOS]).to(options.device)
        predict = []
        tgt_with_eos = attach_eos(tgt)
        # src_feature_map = torch.tensor(gen_sen_feature_map(vocab_src, src)).to(options.device)
        # tgt_feature_map = torch.zeros(options.feature_num).to(options.device)
        tgt_len = len(tgt_with_eos)
        output = 0
        for i in range(tgt_len):
            output = model(src, predict).squeeze()
            _, indices = torch.max(softmax(output), dim=-1)
            indices = indices.view(-1,1)
            # predict = torch.cat((predict,indices[i]), dim= -1)
            predict.append(indices[i].item())
            # tgt_feature_map = torch.tensor(gen_sen_feature_map(vocab_tgt, predict.tolist())).to(options.device)
        loss = criterion(output, tgt_with_eos)
        output = softmax(output)
        predict_idx_list = torch.argmax(output,dim=-1).tolist()
        target_idx_list = tgt_with_eos.tolist()
        predict_word_list = idx2word(predict_idx_list,vocab_tgt)
        target_word_list = idx2word(predict_idx_list,vocab_tgt)
        predict_string = idx2string(predict_idx_list,vocab_tgt)
        target_string = idx2string(predict_idx_list,vocab_tgt)
        acc , bleu = evaluator(predict_idx_list , target_idx_list)
        ppl = evaluator.calc_ppl(loss.item())
        precision = evaluator.calc_precision(predict_idx_list , target_idx_list)
        recall = evaluator.calc_recall(predict_idx_list , target_idx_list)
        f1 = evaluator.calc_f1(predict_idx_list , target_idx_list)
        mae = evaluator.calc_mae(predict_idx_list , target_idx_list)
        mape = evaluator.calc_mape(predict_idx_list , target_idx_list)
        smape = evaluator.calc_smape(predict_idx_list , target_idx_list)
        rouge = evaluator.calc_rouge(prediction=predict_string, reference=target_string)
        meteor= evaluator.calc_meteor(prediction=predict_word_list, reference=target_word_list)

        count = count+ 1
        total_loss =total_loss + loss.item()
        total_acc = total_acc +acc
        total_bleu = total_bleu + bleu
        total_ppl = total_ppl + ppl
        total_precision = total_precision + precision
        total_recall = total_recall + recall
        total_f1 = total_f1 + f1
        total_mae = total_mae + mae
        total_mape = total_mape + mape
        total_smape = total_smape + smape
        total_rouge = total_rouge + np.array(rouge)
        total_meteor = total_meteor + meteor
        if count % int(len(test_data)/10) == 0 :
            logger.info("-----------------------------%s----------------------------------------"%(model.name))
            logger.info("[source ] %s" %(idx2string(src, vocab_src)))
            logger.info("[target ] %s" %(idx2string(tgt, vocab_tgt)))
            logger.info("[predict] %s" %(idx2string(predict, vocab_tgt)))
    loss = total_loss/count
    acc = total_acc/count
    bleu = total_bleu/count
    ppl = total_ppl/count
    precision = total_precision/count
    recall = total_recall/count
    f1 = total_f1/count
    mae = total_mae/count
    mape = total_mape/count
    smape = total_smape/count
    rouge= total_rouge/count
    meteor = total_meteor/count
    return loss, acc, bleu, ppl, precision, recall, f1, mae, mape, smape,rouge.tolist(), meteor


def valid(model, valid_data,evaluator):
    count = 0
    total_acc = 0
    total_bleu = 0
    total_loss = 0
    softmax = nn.Softmax(dim=-1)
    criterion = nn.CrossEntropyLoss()
    for src, tgt in valid_data:
        if len(src) > options.sen_len_max or len(tgt) > options.sen_len_max:
                continue
        if len(src) == 7 or len(src) == 7:
                continue
        # src_with_eos = attach_eos(src)
        # tgt_with_sos = insert_sos(tgt)
        tgt_with_eos = attach_eos(tgt)
        # src_feature_map = torch.tensor(gen_sen_feature_map(vocab_src, src)).to(options.device)
        # tgt_feature_map = torch.tensor(gen_sen_feature_map(vocab_tgt, tgt)).to(options.device)
        output = model(src, tgt)
        output = output.squeeze()
        loss = criterion(output, tgt_with_eos)
        output = softmax(output)
        predict_idx_list = torch.argmax(output,dim=-1).tolist()
        target_idx_list = tgt_with_eos.tolist()
        acc , bleu = evaluator(predict_idx_list , target_idx_list)
        count = count+ 1
        total_loss =total_loss + loss.item()
        total_acc = total_acc +acc
        total_bleu = total_bleu + bleu
    return total_loss/count, total_acc/count,total_bleu/count

def train(model, model_name, dataset_name, train_data, valid_data, test_data, vocab_src, vocab_tgt, pretrain_used=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=options.learning_rate, weight_decay=0)
    criterion = nn.CrossEntropyLoss()
    evaluator = Evaluator()
    model_info(model)
    if pretrain_used == True:
        loadmodel(model, model.name + '-' +dataset_name)
    else:
        for epoch in range(options.epoch):
            count = 0
            total_loss = 0
            for src, tgt in tqdm(train_data):
                if len(src) > options.sen_len_max or len(tgt) > options.sen_len_max:
                    continue
                if len(src) == 7 or len(src) == 7:
                    continue
                # src_with_eos = attach_eos(src)
                # tgt_with_sos = insert_sos(tgt)
                tgt_with_eos = attach_eos(tgt)
                # src_feature_map = torch.tensor(gen_sen_feature_map(vocab_src, src)).to(options.device)
                # tgt_feature_map = torch.tensor(gen_sen_feature_map(vocab_tgt, tgt)).to(options.device)
                optimizer.zero_grad()
                output = model(src, tgt)
                output = output.squeeze()
                loss = criterion(output, tgt_with_eos)
                loss.backward()
                optimizer.step()
                count = count+ 1
                total_loss =total_loss + loss.item()
                if count % int(len(train_data)/10) ==0:
                    valid_loss, acc, bleu = valid(model, valid_data, evaluator)
                    logger.info("[%s-%s]epoch: %d, count: %d, train loss: %.4f, valid loss: %.4f, acc: %.2f, bleu:%.2f" %(model_name, dataset_name, epoch, count, total_loss/count, valid_loss,acc*100, bleu*100))

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
        savemodel(model, model.name + '-' +dataset_name)
    loss, acc, bleu, ppl, precision, recall, f1, mae, mape, smape,rouge, meteor = \
        predict(model, test_data, vocab_src, vocab_tgt, evaluator)
    logger.info("[%s-%s]acc: %.2f, bleu: %.2f" %(model.name,dataset_name ,acc*100, bleu*100))
    return model_name, dataset_name, acc, bleu, ppl, precision, recall, f1, mae, mape, smape,rouge, meteor

def s2s_task(dataset_name, tokenizer, pretrain_used=False):
    model_name = 'fuzzys2s'
    log_file = logger.add(options.base_path+'output/'+model_name+'-'+dataset_name+'-'+str(datetime.date.today()) +'.log')
    logger.info('model %s on dataset %s start...' %(model_name, dataset_name))
    setup_seed(options.seed_id)
    train_data, valid_data, test_data, vocab_src, vocab_tgt = read_data(dataset_name, tokenizer)
    src_sen_feature_map, tgt_sen_feature_map = combine_sen_feature_map(train_data,vocab_src, vocab_tgt)
    logger.info("src token clustering")
    center_src,sigma_src = fcm(src_sen_feature_map, cluster_num= options.rule_num, h= options.h)
    logger.info("tgt token clustering")
    center_tgt,sigma_tgt = fcm(tgt_sen_feature_map, cluster_num= options.rule_num, h= options.h)
    model = FuzzyS2S(vocab_src, vocab_tgt, options.feature_num, options.rule_num, center_src, sigma_src, center_tgt, sigma_tgt).to(options.device)
    model_name, dataset_name, acc, bleu, ppl, precision, recall, f1, mae, mape, smape,rouge, meteor = \
        train(model, model_name, dataset_name, train_data, valid_data, test_data, vocab_src, vocab_tgt, pretrain_used)
    logger.remove(log_file)
    return model_name, dataset_name, acc, bleu, ppl, precision, recall, f1, mae, mape, smape,rouge, meteor


def trans_task(dataset_name, tokenizer, pretrain_used=False):
    model_name = 'transformer'
    log_file = logger.add(options.base_path+'output/'+model_name+'-'+dataset_name+'-'+str(datetime.date.today()) +'.log')
    logger.info('model %s on dataset %s start...' %(model_name, dataset_name))
    setup_seed(options.seed_id)
    train_data, valid_data, test_data, vocab_src, vocab_tgt = read_data(dataset_name, tokenizer)
    model = TransformerModel(vocab_src.n_words,
                             vocab_tgt.n_words,
                             options.trans.embedding_dim,
                             options.trans.nhead,
                             options.trans.hidden_size,
                             options.trans.nlayer,
                             options.trans.drop_out).to(options.device)
    model_name, dataset_name, acc, bleu, ppl, precision, recall, f1, mae, mape, smape,rouge, meteor = \
        train(model, model_name, dataset_name, train_data, valid_data, test_data, vocab_src, vocab_tgt, pretrain_used)
    logger.remove(log_file)
    return model_name, dataset_name, acc, bleu, ppl, precision, recall, f1, mae, mape, smape,rouge, meteor

def run():
    # datasets = ['opus_euconst','hearthstone','magic', 'spider','samsum', 'gem', 'xlsum','django','conala', 'geo', 'atis']
    datasets =['geo']
    results = []
    tokenizer = get_tokenizer("basic_english")
    # tokenizer = get_base_tokenizer('bert-base-uncased')
    for dataset in datasets:
        model_name, dataset_name, acc, bleu, ppl, precision, recall, f1, mae, mape, smape,rouge, meteor = \
            trans_task(dataset, tokenizer,pretrain_used=False)
        results.append([model_name, dataset_name, acc, bleu, ppl, precision, recall, f1, mae, mape, smape,rouge, meteor])
        model_name, dataset_name, acc, bleu, ppl, precision, recall, f1, mae, mape, smape,rouge, meteor = \
            s2s_task(dataset, tokenizer,pretrain_used=False)
        results.append([model_name, dataset_name, acc, bleu, ppl, precision, recall, f1, mae, mape, smape,rouge, meteor])
    log_file = logger.add(options.base_path+'output/result-'+str(datetime.date.today()) +'.log')
    for model_name, dataset_name, acc, bleu, ppl, precision, recall, f1, mae, mape, smape,rouge, meteor in results:
        logger.info("------------------------------------result------------------------------------------" )
        logger.info("model: %s, datset: %s, acc   : %.2f, bleu: %.2f" %(model_name, dataset_name, acc*100, bleu*100))
        logger.info("model: %s, datset: %s, ppl   : %.2f, precision: %.2f" %(model_name, dataset_name, ppl, precision*100))
        logger.info("model: %s, datset: %s, recall: %.2f, f1: %.2f" %(model_name, dataset_name, recall*100, f1*100))
        logger.info("model: %s, datset: %s, mae   : %.2f, mape: %.2f" %(model_name, dataset_name, mae, mape*100))
        logger.info("model: %s, datset: %s, rouge1: %.2f, rouge2: %.2f" %(model_name, dataset_name, rouge[0]*100, rouge[1]*100))
        logger.info("model: %s, datset: %s, rougeL: %.2f, rougeLsum: %.2f" %(model_name, dataset_name, rouge[2]*100, rouge[3]*100))
        logger.info("model: %s, datset: %s, smape : %.2f, meteor: %.2f" %(model_name, dataset_name, smape*100, meteor*100))
    logger.remove(log_file)
    return 0
run()
