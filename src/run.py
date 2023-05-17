from torch import nn
import torch
import torch.nn.functional as F
from torchtext.data import get_tokenizer
import datetime
import numpy as np
import random
import math
from loguru import logger
from model import FuzzyS2S,TransformerModel,RnnModel
from loaddata import read_data,fcm, gen_sen_feature_map,combine_sen_feature_map,insert_sos,attach_eos,get_base_tokenizer
from setting import options, setting_info
import os
from evaluator import Evaluator,MetricsValue
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration,MT5ForConditionalGeneration, AutoTokenizer,BartForConditionalGeneration, BartTokenizer,AutoModel,AutoModelForSeq2SeqLM

def model_info(model):
    logger.info("[model %s]" %(model.name))
    logger.info("%s" %(setting_info()))
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("total parameters: %d, trainable  parameters: %d " %(total_params,total_trainable_params))
    for name,parameters in model.named_parameters():
        logger.info('%s : %s' %(name,str(parameters.size())))

def model_param(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("total parameters: %d, trainable  parameters: %d " %(total_params,total_trainable_params))
    for name,parameters in model.named_parameters():
        logger.info('%s : %s' %(name,str(parameters.size())))

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
    total_metrics = MetricsValue()
    softmax = nn.Softmax(dim=-1)
    criterion = nn.CrossEntropyLoss()
    for src, tgt in tqdm(test_data,'test data'):
        if len(src) > options.sen_len_max or len(tgt) > options.sen_len_max:
            continue
        if len(src) == 7 or len(src) == 7:
            continue
        if len(src) == 0 or len(src) == 0:
            continue
        # src_with_eos = attach_eos(src)
        # tgt_with_sos = torch.tensor([options.SOS]).to(options.device)
        predict = []  # model add sos for input
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
            if indices[i].item() == options.EOS:
                break
            # tgt_feature_map = torch.tensor(gen_sen_feature_map(vocab_tgt, predict.tolist())).to(options.device)
        # loss = criterion(output, tgt_with_eos)
        output = softmax(output)
        predict_idx_list = torch.argmax(output,dim=-1).tolist()
        target_idx_list = tgt_with_eos.tolist()
        predict_string = idx2string(predict_idx_list,vocab_tgt)
        target_string = idx2string(target_idx_list,vocab_tgt)
        acc , bleu = evaluator(predict_idx_list , target_idx_list)
        # evaluator.calc_ppl(loss.item())
        evaluator.calc_accuracy(predict_idx_list , target_idx_list)
        evaluator.calc_precision(predict_idx_list , target_idx_list)
        evaluator.calc_recall(predict_idx_list , target_idx_list)
        evaluator.calc_f1(predict_idx_list , target_idx_list)
        evaluator.calc_mae(predict_idx_list , target_idx_list)
        evaluator.calc_mape(predict_idx_list , target_idx_list)
        evaluator.calc_smape(predict_idx_list , target_idx_list)
        evaluator.calc_sen_bleu(candidate=predict_string, reference=target_string)
        evaluator.calc_sacre_bleu(candidate=predict_string, reference=target_string)
        evaluator.calc_chrf2(candidate=predict_string, reference=target_string)
        evaluator.calc_ter(candidate=predict_string, reference=target_string)
        evaluator.calc_rouge(prediction=predict_string, reference=target_string)
        evaluator.calc_meteor(prediction=predict_string, reference=target_string)
        result = evaluator.metrics()
        count = count+ 1
        total_metrics.add(result)
        if count % int(len(test_data)/10) == 0 :
            logger.info("-----------------------------%s----------------------------------------"%(model.name))
            logger.info("[source ] %s" %(idx2string(src, vocab_src)))
            logger.info("[target ] %s" %(idx2string(tgt, vocab_tgt)))
            logger.info("[predict] %s" %(idx2string(predict, vocab_tgt)))
    result = total_metrics.average(count)
    return result

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
        tgt_with_eos = attach_eos(tgt)
        # if is_recurrent:
        #     src_with_eos = attach_eos(src)
        #     hidden = model.encoder(src_with_eos)
        #     target = [options.SOS]
        #     loss = 0
        #     for i in range(len(tgt)):
        #         output = model.decoder(torch.tensor(target).to(options.device),hidden)
        #         output = output.view(len(target), -1)
        #         loss = loss + criterion(output,torch.tensor(target).to(options.device))
        #         target.append(tgt[i])
        # else:
        # src_with_eos = attach_eos(src)
        # tgt_with_sos = insert_sos(tgt)
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
    if pretrain_used == True :
        loadmodel(model, model_name + '-' +dataset_name)
    else:
        for epoch in range(options.epoch):
            count = 0
            total_loss = 0
            for src, tgt in tqdm(train_data,'train data'):
                if len(src) > options.sen_len_max or len(tgt) > options.sen_len_max:
                    continue
                if len(src) == 7 or len(src) == 7:
                    continue
                if len(src) == 0 or len(src) == 0:
                    continue
                # src_with_eos = attach_eos(src)
                # tgt_with_sos = insert_sos(tgt)
                # src_feature_map = torch.tensor(gen_sen_feature_map(vocab_src, src)).to(options.device)
                # tgt_feature_map = torch.tensor(gen_sen_feature_map(vocab_tgt, tgt)).to(options.device)
                optimizer.zero_grad()
                tgt_with_eos = attach_eos(tgt)
                output = model(src, tgt)
                output = output.squeeze()
                loss = criterion(output, tgt_with_eos)
                loss.backward()
                optimizer.step()
                count = count+ 1
                total_loss =total_loss + loss.item()
                if count % int(len(train_data)/10) ==0:
                    valid_loss, acc, bleu = valid(model, valid_data, evaluator)
                    logger.info("[%s-%s]epoch: %d, count: %d, train loss: %.4f, valid loss: %.4f, acc: %.2f" %(model_name, dataset_name, epoch, count, total_loss/count, valid_loss,acc))

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
        savemodel(model,model_name + '-' +dataset_name)
    result = predict(model, test_data, vocab_src, vocab_tgt, evaluator)
    result['model_name'] = model_name
    result['dataset_name'] = dataset_name
    logger.info("[%s-%s]acc: %.2f, sen_bleu: %.2f" %(model_name,dataset_name ,result['acc'], result['sen_bleu']))
    return result

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
    result = train(model, model_name, dataset_name, train_data, valid_data, test_data, vocab_src, vocab_tgt, pretrain_used)
    logger.remove(log_file)
    return result

def rnn_task(dataset_name, tokenizer, pretrain_used=False):
    model_name = 'rnn'
    log_file = logger.add(options.base_path+'output/'+model_name+'-'+dataset_name+'-'+str(datetime.date.today()) +'.log')
    logger.info('model %s on dataset %s start...' %(model_name, dataset_name))
    setup_seed(options.seed_id)
    train_data, valid_data, test_data, vocab_src, vocab_tgt = read_data(dataset_name, tokenizer)
    model = RnnModel(vocab_src.n_words,
                    vocab_tgt.n_words,
                    options.rnn.hidden_size,
                    options.rnn.nlayer,
                    options.rnn.drop_out).to(options.device)
    result = train(model, model_name, dataset_name, train_data, valid_data, test_data, vocab_src, vocab_tgt, pretrain_used)
    logger.remove(log_file)
    return result

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
    result = train(model, model_name, dataset_name, train_data, valid_data, test_data, vocab_src, vocab_tgt, pretrain_used)
    logger.remove(log_file)
    return result

def t5_task(model_name, dataset_name, pretrain_used=True):
    log_file = logger.add(options.base_path+'output/'+model_name+'-'+dataset_name+'-'+str(datetime.date.today()) +'.log')
    logger.info('model %s on dataset %s start...' %(model_name, dataset_name))
    setup_seed(options.seed_id)
    train_sen_pairs, valid_sen_pairs, test_sen_pairs = read_data(dataset_name, sen_out=True)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(options.device)
    model_param(model)
    max_source_length = 512
    max_target_length = 128
    count = 0
    total_metrics = MetricsValue()
    evaluator = Evaluator()
    task_prefix = "translate English to French: "
    for src,tgt in tqdm(test_sen_pairs,"test data"):
        if len(src) > max_source_length or len(tgt) >max_target_length:
            continue
        if len(src) == 0 or len(tgt) == 0:
            continue
        tgt_ids = tokenizer(tgt, return_tensors="pt").input_ids.to(options.device)
        src_ids= tokenizer(task_prefix + src, return_tensors="pt").input_ids.to(options.device)
        outputs = model.generate(src_ids, max_length= max_target_length)
        predict_idx = outputs[0]
        predict_str = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predict_idx_list = predict_idx.tolist()
        target_idx_list = tgt_ids[0].tolist()
        predict_string = predict_str
        target_string = tgt
        acc , bleu = evaluator(predict_idx_list , target_idx_list)
        evaluator.calc_accuracy(predict_idx_list , target_idx_list)
        evaluator.calc_precision(predict_idx_list , target_idx_list)
        evaluator.calc_recall(predict_idx_list , target_idx_list)
        evaluator.calc_f1(predict_idx_list , target_idx_list)
        evaluator.calc_mae(predict_idx_list , target_idx_list)
        evaluator.calc_mape(predict_idx_list , target_idx_list)
        evaluator.calc_smape(predict_idx_list , target_idx_list)
        evaluator.calc_sen_bleu(candidate=predict_string, reference=target_string)
        evaluator.calc_sacre_bleu(candidate=predict_string, reference=target_string)
        evaluator.calc_google_bleu(candidate=predict_string, reference=target_string)
        evaluator.calc_chrf2(candidate=predict_string, reference=target_string)
        evaluator.calc_ter(candidate=predict_string, reference=target_string)
        evaluator.calc_rouge(prediction=predict_string, reference=target_string)
        evaluator.calc_meteor(prediction=predict_string, reference=target_string)
        result = evaluator.metrics()
        count = count+ 1
        total_metrics.add(result)
        if count % int(len(test_sen_pairs)/10) == 0 :
            logger.info("-----------------------------%s----------------------------------------"%(model_name))
            logger.info("[source ] %s" %(src))
            logger.info("[target ] %s" %(tgt))
            logger.info("[predict] %s" %(predict_str))
    result = total_metrics.average(count)
    result['model_name']=model_name
    result['dataset_name'] = dataset_name
    logger.info("[%s-%s]acc: %.2f, sen_bleu: %.2f" %(result['model_name'],result['dataset_name'] ,result['acc'], result['sen_bleu']))
    logger.remove(log_file)
    return result

def mt5_task(model_name, dataset_name, pretrain_used=True):
    log_file = logger.add(options.base_path+'output/'+model_name+'-'+dataset_name+'-'+str(datetime.date.today()) +'.log')
    logger.info('model %s on dataset %s start...' %(model_name, dataset_name))
    setup_seed(options.seed_id)
    train_sen_pairs, valid_sen_pairs, test_sen_pairs = read_data(dataset_name, sen_out=True)
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small").to(options.device)
    model_param(model)
    max_source_length = 512
    max_target_length = 128
    count = 0
    total_metrics = MetricsValue()
    evaluator = Evaluator()
    task_prefix = "translate English to French: "
    for src,tgt in tqdm(test_sen_pairs,"test data"):
        if len(src) > max_source_length or len(tgt) >max_target_length:
            continue
        if len(src) == 0 or len(tgt) == 0:
            continue
        tgt_ids = tokenizer(tgt, return_tensors="pt").input_ids.to(options.device)
        input_ids= tokenizer(task_prefix + src, return_tensors="pt").input_ids.to(options.device)
        outputs = model.generate(input_ids, max_length= max_target_length)
        predict_idx = outputs[0]
        predict_str = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predict_idx_list = predict_idx.tolist()
        target_idx_list = tgt_ids[0].tolist()
        predict_string = predict_str
        target_string = tgt
        # predict_token_list = tokenizer.convert_ids_to_tokens(predict_idx_list)
        # target_token_list = tokenizer.convert_ids_to_tokens(target_idx_list)
        # predict_word_list = predict_str.split(' ')
        # target_word_list = tgt.split(' ')
        acc , bleu = evaluator(predict_idx_list , target_idx_list)
        evaluator.calc_accuracy(predict_idx_list , target_idx_list)
        evaluator.calc_precision(predict_idx_list , target_idx_list)
        evaluator.calc_recall(predict_idx_list , target_idx_list)
        evaluator.calc_f1(predict_idx_list , target_idx_list)
        evaluator.calc_mae(predict_idx_list , target_idx_list)
        evaluator.calc_mape(predict_idx_list , target_idx_list)
        evaluator.calc_smape(predict_idx_list , target_idx_list)
        evaluator.calc_sen_bleu(candidate=predict_string, reference=target_string)
        evaluator.calc_sacre_bleu(candidate=predict_string, reference=target_string)
        evaluator.calc_google_bleu(candidate=predict_string, reference=target_string)
        evaluator.calc_chrf2(candidate=predict_string, reference=target_string)
        evaluator.calc_ter(candidate=predict_string, reference=target_string)
        evaluator.calc_rouge(prediction=predict_string, reference=target_string)
        evaluator.calc_meteor(prediction=predict_string, reference=target_string)
        result = evaluator.result()
        count = count+ 1
        total_metrics.add(result)
        if count % int(len(test_sen_pairs)/10) == 0 :
            logger.info("-----------------------------%s----------------------------------------"%(model_name))
            logger.info("[source ] %s" %(src))
            logger.info("[target ] %s" %(tgt))
            logger.info("[predict] %s" %(predict_str))
    result = total_metrics.average(count)
    result['model_name']=model_name
    result['dataset_name'] = dataset_name
    logger.info("[%s-%s]acc: %.2f, sen_bleu: %.2f" %(result['model_name'],result['dataset_name'] ,result['acc'], result['sen_bleu']))
    logger.remove(log_file)
    return result

def opus_mt_task(model_name, dataset_name, pretrain_used=True):
    log_file = logger.add(options.base_path+'output/'+model_name+'-'+dataset_name+'-'+str(datetime.date.today()) +'.log')
    logger.info('model %s on dataset %s start...' %(model_name, dataset_name))
    setup_seed(options.seed_id)
    train_sen_pairs, valid_sen_pairs, test_sen_pairs = read_data(dataset_name, sen_out=True)
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr").to(options.device)
    model_param(model)
    max_source_length = 512
    max_target_length = 128
    count = 0
    total_metrics = MetricsValue()
    evaluator = Evaluator()
    # task_prefix = "translate English to French: "
    for src,tgt in tqdm(test_sen_pairs,"test data"):
        if len(src) > max_source_length or len(tgt) >max_target_length:
            continue
        if len(src) == 0 or len(tgt) == 0:
            continue
        tgt_ids = tokenizer(tgt, return_tensors="pt").input_ids.to(options.device)
        src_ids= tokenizer(src, return_tensors="pt").input_ids.to(options.device)
        outputs = model.generate(src_ids, max_length= max_target_length)
        predict_idx = outputs[0]
        predict_str = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predict_idx_list = predict_idx.tolist()
        target_idx_list = tgt_ids[0].tolist()
        predict_string = predict_str
        target_string = tgt
        acc , bleu = evaluator(predict_idx_list , target_idx_list)
        evaluator.calc_accuracy(predict_idx_list , target_idx_list)
        evaluator.calc_precision(predict_idx_list , target_idx_list)
        evaluator.calc_recall(predict_idx_list , target_idx_list)
        evaluator.calc_f1(predict_idx_list , target_idx_list)
        evaluator.calc_mae(predict_idx_list , target_idx_list)
        evaluator.calc_mape(predict_idx_list , target_idx_list)
        evaluator.calc_smape(predict_idx_list , target_idx_list)
        evaluator.calc_sen_bleu(candidate=predict_string, reference=target_string)
        evaluator.calc_sacre_bleu(candidate=predict_string, reference=target_string)
        evaluator.calc_google_bleu(candidate=predict_string, reference=target_string)
        evaluator.calc_chrf2(candidate=predict_string, reference=target_string)
        evaluator.calc_ter(candidate=predict_string, reference=target_string)
        evaluator.calc_rouge(prediction=predict_string, reference=target_string)
        evaluator.calc_meteor(prediction=predict_string, reference=target_string)
        result = evaluator.metrics()
        count = count+ 1
        total_metrics.add(result)
        if count % int(len(test_sen_pairs)/10) == 0 :
            logger.info("-----------------------------%s----------------------------------------"%(model_name))
            logger.info("[source ] %s" %(src))
            logger.info("[target ] %s" %(tgt))
            logger.info("[predict] %s" %(predict_str))
    result = total_metrics.average(count)
    result['model_name']=model_name
    result['dataset_name'] = dataset_name
    logger.info("[%s-%s]acc: %.2f, sen_bleu: %.2f" %(result['model_name'],result['dataset_name'] ,result['acc'], result['sen_bleu']))
    logger.remove(log_file)
    return result

def run():
    #datasets = ['opus_euconst','hearthstone','magic', 'spider','samsum', 'gem', 'xlsum','django','conala', 'geo', 'atis']
    # datasets =['opus_euconst', 'tatoeba','opus100','wmt14']
    # datasets =['hearthstone', 'magic',  'django', 'conala']
    datasets= ['opus_euconst']
    results = []
    tokenizer = get_tokenizer("basic_english")
    # tokenizer = get_base_tokenizer('bert-base-uncased')
    for dataset in datasets:
        # result = t5_task('t5-small',dataset)
        # results.append(result)
        # result = t5_task('t5-base',dataset)
        # results.append(result)
        # result = t5_task('t5-large',dataset)
        # results.append(result)
        # result = mt5_task('google/mt5-small',dataset)
        # results.append(result)
        # result = opus_mt_task('Helsinki-NLP/opus-mt-en-fr', dataset)
        # results.append(result)
        # result = trans_task(dataset, tokenizer,pretrain_used=True)
        # results.append(result)
        # result = s2s_task(dataset, tokenizer,pretrain_used=True)
        # results.append(result)
        result = rnn_task(dataset, tokenizer,pretrain_used=False)
        results.append(result)
    log_file = logger.add(options.base_path+'output/result-'+str(datetime.date.today()) +'.log')
    for result in results:
        logger.info("------------------------------------result------------------------------------------" )
        logger.info("model: %s, datset: %s, acc   : %.2f, sen_bleu: %.2f" %(result['model_name'], result['dataset_name'], result['acc'], result['sen_bleu']))
        logger.info("model: %s, datset: %s, sacre_bleu: %.2f, google_bleu: %.2f" %(result['model_name'], result['dataset_name'], result['sacre_bleu'], result['google_bleu']))
        logger.info("model: %s, datset: %s, chrf2 : %.2f, ter: %.2f" %(result['model_name'], result['dataset_name'], result['chrf2'], result['ter']))
        logger.info("model: %s, datset: %s, ppl   : %.2f, precision: %.2f" %(result['model_name'], result['dataset_name'], result['ppl'],result['precision']))
        logger.info("model: %s, datset: %s, recall: %.2f, f1: %.2f" %(result['model_name'], result['dataset_name'], result['recall'], result['f1']))
        logger.info("model: %s, datset: %s, mae   : %.2f, mape: %.2f" %(result['model_name'], result['dataset_name'], result['mae'], result['mape']))
        logger.info("model: %s, datset: %s, rouge1: %.2f, rouge2: %.2f" %(result['model_name'], result['dataset_name'], result['rouge1'], result['rouge2']))
        logger.info("model: %s, datset: %s, rougeL: %.2f, rougeLsum: %.2f" %(result['model_name'], result['dataset_name'], result['rougeL'], result['rougeLsum']))
        logger.info("model: %s, datset: %s, smape : %.2f, meteor: %.2f" %(result['model_name'], result['dataset_name'], result['smape'], result['meteor']))
    logger.remove(log_file)
    return 0
run()
