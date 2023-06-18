from torch import nn
import torch
import torch.nn.functional as F
from torchtext.data import get_tokenizer
import datetime
import numpy as np
import random
import math
from loguru import logger
from model import FuzzyS2S,TransformerModel,RnnModel, FuzzyS2S_B
from loaddata import read_data,fcm, gen_sen_feature_map,combine_sen_feature_map,insert_sos,attach_eos,get_base_tokenizer
from setting import options, setting_info
import os
from evaluator import Evaluator,MetricsValue
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration,MT5ForConditionalGeneration, AutoTokenizer,\
    BartForConditionalGeneration, BartTokenizer,AutoModel,AutoModelForSeq2SeqLM,TrainingArguments,Trainer, \
    Seq2SeqTrainingArguments, Seq2SeqTrainer,AutoModelForCausalLM,CodeGenModel,RobertaForCausalLM,\
    PegasusForConditionalGeneration, PegasusTokenizer, PegasusXForConditionalGeneration
from datasets import load_dataset,load_from_disk,Dataset
from fuzzy_tokenizer import get_fuzzy_tokenizer
import csv

def setup_seed(seed):
    # https://zhuanlan.zhihu.com/p/462570775
    torch.use_deterministic_algorithms(True) # 检查pytorch中有哪些不确定性
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # 大于CUDA 10.2 需要设置
    logger.info("seed: %d, random:%.4f, torch random:%.4f, np random:%.4f" %(seed, random.random(), torch.rand(1), np.random.rand(1)))

def model_info(model):
    logger.info("[model %s]" %(model.name))
    logger.info("%s" %(setting_info()))
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("total parameters: %d, trainable  parameters: %d " %(total_params,total_trainable_params))
    # for name,parameters in model.named_parameters():
    #     logger.info('%s : %s' %(name,str(parameters.size())))

def model_param(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("total parameters: %d, trainable  parameters: %d " %(total_params,total_trainable_params))
    for name,parameters in model.named_parameters():
        logger.info('%s : %s' %(name,str(parameters.size())))

def model_finetune_pairs_ids(pairs, tokenizer, task_prefix, max_src_len, max_tgt_len):
    input_ids = torch.zeros(len(pairs), max_src_len).to(options.device)
    labels = torch.zeros(len(pairs), max_tgt_len).to(options.device)
    for i in tqdm(range(len(pairs))):
        pair = pairs[i]
        src = task_prefix + pair[0]
        tgt = pair[1]
        src = tokenizer(src, return_tensors="pt").input_ids
        tgt =  tokenizer(tgt, return_tensors="pt").input_ids
        tmp_ids = model_finetune_padding(src, max_src_len)
        tmp_labels = model_finetune_padding(tgt, max_tgt_len)
        input_ids[i] = tmp_ids[0]
        labels[i] = tmp_labels[0]
    raw_dict = {'input_ids':input_ids.long(), 'labels':labels.long()}
    dataset = Dataset.from_dict(raw_dict)
    return dataset


def model_finetune(model,model_path, tokenizer,task_prefix, train_sen_pairs, valid_sen_pairs, epoch_num=1, max_src_length=512, max_tgt_length=512):
    logger.info('model fine tune start...')
    model.train()
    train_dataset = model_finetune_pairs_ids(train_sen_pairs, tokenizer, task_prefix, max_src_length, max_tgt_length)
    valid_dataset = model_finetune_pairs_ids(valid_sen_pairs, tokenizer, task_prefix, max_src_length, max_tgt_length)
    default_args = {
        "output_dir": model_path,
        "evaluation_strategy": "steps",
        "num_train_epochs": epoch_num,
        'per_device_train_batch_size': 8,
        'per_device_eval_batch_size': 8,
        "log_level": "error",
        "report_to": "none",
    }
    training_args = Seq2SeqTrainingArguments(**default_args)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer
    )
    trainer.train()
    logger.info('model fine tune done')
    model.eval()

def model_finetune_padding(input, max_len):
    output = torch.empty(1,max_len).to(options.device)
    for i in range(max_len):
        if i < len(input[0]):
            output[0][i] = input[0][i]
        else :
            output[0][i] = 4  ## padding token id
    return output

def model_finetune_align(src, tgt):
    if len(src[0]) == len(tgt[0]):
        return src, tgt
    elif len(src[0]) > len(tgt[0]):
        max_len = len(src[0])
        tgt = model_finetune_padding(tgt, max_len)
    else:
        max_len = len(tgt[0])
        src = model_finetune_padding(src, max_len)
    return src.long(), tgt.long()

def model_finetune2(model,model_path, tokenizer,task_prefix, train_sen_pairs, valid_sen_pairs, epoch_num=2, max_src_length=512, max_tgt_length=512):
    logger.info('model fine tune start...')
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=options.learning_rate, weight_decay=0)
    for i in range(epoch_num):
        for src, tgt in tqdm(train_sen_pairs, 'finetune'):
            optimizer.zero_grad()
            input_ids = tokenizer(task_prefix+ src, return_tensors="pt").input_ids.to(options.device)
            labels = tokenizer(tgt, return_tensors="pt").input_ids.to(options.device)
            input_ids, labels = model_finetune_align( input_ids, labels)
            if len(input_ids[0]) > max_src_length:
                temp_ids = input_ids[0][0:max_src_length]
                input_ids= temp_ids.view(1,-1)
            if len(labels[0]) > max_tgt_length:
                temp_labels = labels[0][0:max_tgt_length]
                labels = temp_labels.view(1,-1)
            raw_dict = {'input_ids':input_ids, 'labels':labels}
            loss = model(**raw_dict).loss
            loss.backward()
            optimizer.step()
    logger.info('model fine tune done')
    model.eval()

def savemodel(model,file):
    if model.name == "fuzzys2s":
        if options.tokenizer.fuzzy:
            path = options.model_parameter_path+file+"_"+str(options.tokenizer.fuzzy_rule_num)+"_rule_"+str(options.rule_num)+"_rule_fuzzy.pth"
        else:
            path = options.model_parameter_path+file+"_"+str(options.rule_num)+"_rule_basic.pth"
    elif model.name == "transformer":
        if options.tokenizer.fuzzy:
            path = options.model_parameter_path+file+"_"+str(options.tokenizer.fuzzy_rule_num)+"_rule_fuzzy.pth"
        else:
            path = options.model_parameter_path+file+"_basic.pth"
    else:
         path = options.model_parameter_path+file+".pth"
    torch.save(model.state_dict(), path)
    logger.info("save %s model parameters done." %(file))

def loadmodel(model, file):
    if model.name == "fuzzys2s":
        if options.tokenizer.fuzzy:
            path = options.model_parameter_path+file+"_"+str(options.tokenizer.fuzzy_rule_num)+"_rule_"+str(options.rule_num)+"_rule_fuzzy.pth"
        else:
            path = options.model_parameter_path+file+"_"+str(options.rule_num)+"_rule_basic.pth"
    elif model.name == "transformer":
        if options.tokenizer.fuzzy:
            path = options.model_parameter_path+file+"_"+str(options.tokenizer.fuzzy_rule_num)+"_rule_fuzzy.pth"
        else:
            path = options.model_parameter_path+file+"_basic.pth"
    else:
         path = options.model_parameter_path+file+".pth"
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        model.eval()
        logger.info("load %s model parameters done." %(file))

def save_train_info(name, data, attach=False):
    logger.info("save %s train info..." %(name))
    info_path = options.base_path+'output/csv/train_'+name+'.csv'
    headers = ['loss', 'acc']
    rows = np.empty((len(data), 2)).tolist()
    for i in range(len(rows)):
        rows[i][0] = data[i][0]
        rows[i][1] = data[i][1]
    if attach:
        with open(info_path, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
    else :
        with open(info_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)

def load_train_info(name):
    logger.info("load %s centers..." %(name))
    info_path = options.base_path+'output/csv/train_'+name+'.csv'
    data = []
    with open(info_path, encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        print(headers)
        for row in reader:
            data.append([row[0], row[1]])
    return data

def check_fcm_info(dataset):
    if options.tokenizer.fuzzy:
        info_path = options.base_path+'output/fuzzys2s/'+ dataset+'_'+str(options.tokenizer.fuzzy_rule_num)+'_rule_'+str(options.rule_num)+'_rule_cluster_info.csv'
    else :
        info_path = options.base_path+'output/fuzzys2s/'+ dataset+'_0_rule_'+str(options.rule_num)+'_rule_cluster_info.csv'
    if os.path.exists(info_path):
        logger.info('find %s fcm info' %(dataset))
        return True
    else:
        return False

def save_fcm_info(dataset, centers, sigma):
    if options.tokenizer.fuzzy:
        info_path = options.base_path+'output/fuzzys2s/'+ dataset+'_'+str(options.tokenizer.fuzzy_rule_num)+'_rule_'+str(options.rule_num)+'_rule_cluster_info.csv'
    else :
        info_path = options.base_path+'output/fuzzys2s/'+ dataset+'_0_rule_'+str(options.rule_num)+'_rule_cluster_info.csv'
    headers = ['center_x', 'center_y', 'sigma_x', 'sigma_y']
    rows = np.empty((len(centers), 4)).tolist()
    for i in range(len(rows)):
        rows[i][0] = centers[i][0]
        rows[i][1] = centers[i][1]
        rows[i][2] = sigma[i][0]
        rows[i][3] = sigma[i][1]
    with open(info_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    logger.info('save %s fcm info' %(dataset))

def load_fcm_info(dataset):
    if options.tokenizer.fuzzy:
        info_path = options.base_path+'output/fuzzys2s/'+ dataset+'_'+str(options.tokenizer.fuzzy_rule_num)+'_rule_'+str(options.rule_num)+'_rule_cluster_info.csv'
    else :
        info_path = options.base_path+'output/fuzzys2s/'+ dataset+'_0_rule_'+str(options.rule_num)+'_rule_cluster_info.csv'
    centers = []
    sigma = []
    with open(info_path, encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        # print(headers)
        for row in reader:
            centers.append([float(row[0]), float(row[1])])
            sigma.append([float(row[2]), float(row[3])])
    centers = torch.Tensor(centers).to(options.device)
    sigma = torch.Tensor(sigma).to(options.device)
    rule_num = options.rule_num
    src_center = centers[:rule_num]
    src_sigma = sigma[:rule_num]
    tgt_center = centers[rule_num:]
    tgt_sigma = sigma[rule_num:]
    logger.info('load %s fcm info' %(dataset))
    return src_center, src_sigma, tgt_center, tgt_sigma

def tensor2string(input_lang, source):
   output =  [input_lang.index2word[idx.item()] for idx in source]
   outstr = ''.join(x + ' ' for x in output)
   return outstr

def idx2word(source, vocab):
   word_list =  [vocab.index2word[idx] for idx in source]
   return word_list

def idx2string(source, vocab):
    if type(source)!= type([1,2,4]):
        return '<UNK>'
    word_list =  [vocab.index2word[idx] for idx in source]
    string_out = " ".join(word for word in word_list)
    return string_out

def handle_train_result(results):
    if len(results) < 2:
        return True
    pre = results[-2]
    cur = results[-1]
    count = 0
    for i in range(len(pre)):
        if cur[i] > pre[i]:
            count = count +1
    if count > 0:
        return True
    else:
         return False

def predict(model, test_data, vocab_src, vocab_tgt, evaluator):
    count = 0
    total_metrics = MetricsValue()
    softmax = nn.Softmax(dim=-1)
    criterion = nn.CrossEntropyLoss()
    model.eval()
    for src, tgt in tqdm(test_data,'test data'):
        if len(src) > options.sen_len_max:
            src = src[:options.sen_len_max]
        if len(tgt) > options.sen_len_max:
            tgt = tgt[:options.sen_len_max]
        if len(src) == 7 or len(tgt) == 7:
            continue
        if len(src) == 0 or len(tgt) == 0:
            continue
        if len(src) == 1 or len(tgt) ==1:
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
            if indices[i].item() == options.EOS:
                break
            predict.append(indices[i].item())
            # tgt_feature_map = torch.tensor(gen_sen_feature_map(vocab_tgt, predict.tolist())).to(options.device)
        # loss = criterion(output, tgt_with_eos)
        output = softmax(output)
        predict_idx_list = torch.argmax(output,dim=-1).tolist()
        target_idx_list = tgt_with_eos.tolist()
        if type(predict_idx_list) != type([1,2,3]) or type(target_idx_list) != type([1,2,3]):
            continue # predict is null
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
        evaluator.calc_google_bleu(candidate=predict_string, reference=target_string)
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
        if len(src) > options.sen_len_max:
            src = src[:options.sen_len_max]
        if len(tgt) > options.sen_len_max:
            tgt = tgt[:options.sen_len_max]
        if len(src) == 7 or len(tgt) == 7:
            continue
        if len(src) == 0 or len(tgt) == 0:
            continue
        if len(src) == 1 or len(tgt) ==1:
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

def train(model, model_name, dataset_name, train_data, valid_data, test_data, vocab_src, vocab_tgt, pretrain_used=False, continual_learning=False,epoch_num = options.epoch):
    optimizer = torch.optim.Adam(model.parameters(), lr=options.learning_rate, weight_decay=0)
    criterion = nn.CrossEntropyLoss()
    evaluator = Evaluator()
    model_info(model)

    if pretrain_used == True :
        loadmodel(model, model_name + '-' +dataset_name)
    else:
        if continual_learning:
            loadmodel(model, model_name + '-' +dataset_name)
        model.train()
        train_results = []
        for epoch in range(epoch_num):
            count = 0
            total_loss = 0
            # if epoch > 2 and handle_train_result(train_results):
            #     logger.info("model: %s, dataset: %s, epoch %d result is best" %(model_name, dataset_name, epoch))
            #     logger.info("acc: %s" %(str(train_results[-1])))
            #     break
            for src, tgt in tqdm(train_data,'train data'):
                if len(src) > options.sen_len_max:
                    src = src[:options.sen_len_max]
                if len(tgt) > options.sen_len_max:
                    tgt = tgt[:options.sen_len_max]
                if len(src) == 7 or len(tgt) == 7:
                    continue
                if len(src) == 0 or len(tgt) == 0:
                    continue
                if len(src) == 1 or len(tgt) ==1:
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
                    train_results.append([total_loss/count, acc])
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
        if continual_learning:
            attach = True
        else:
            attach = False
        save_train_info(model_name+'_'+dataset_name, train_results, attach=attach)
    result = predict(model, test_data, vocab_src, vocab_tgt, evaluator)
    result['model_name'] = model_name
    result['dataset_name'] = dataset_name
    result['epoch'] = epoch_num
    result['tokenizer_rule'] = options.tokenizer.fuzzy_rule_num
    result['s2s_rule'] = options.rule_num
    logger.info("[%s-%s]acc: %.2f, sen_bleu: %.2f, meteor: %.2f" %(result['model_name'],result['dataset_name'] ,result['acc'], result['sen_bleu'], result['meteor']))
    logger.info("[ablation]fuzzy tokenizer: %s, fuzzy vae: %s" %(str(options.ablation.fuzzy_tokenizer), str(options.ablation.fuzzy_vae)))
    return result

def s2s_task(dataset_name, tokenizer, pretrain_used=False, continual_learning=False):
    model_name = 'fuzzys2s'
    log_file = logger.add(options.base_path+'output/log/'+model_name+'-'+dataset_name+'-'+str(datetime.date.today()) +'.log')
    logger.info('model %s on dataset %s start...' %(model_name, dataset_name))
    setup_seed(options.seed_id)
    train_data, valid_data, test_data, vocab_src, vocab_tgt = read_data(dataset_name, tokenizer)
    if check_fcm_info(dataset_name):
        center_src,sigma_src,center_tgt,sigma_tgt = load_fcm_info(dataset_name)
    else:
        src_sen_feature_map, tgt_sen_feature_map = combine_sen_feature_map(train_data,vocab_src, vocab_tgt)
        logger.info("src token clustering")
        center_src,sigma_src = fcm(src_sen_feature_map, cluster_num= options.rule_num, h= options.h)
        logger.info("tgt token clustering")
        center_tgt,sigma_tgt = fcm(tgt_sen_feature_map, cluster_num= options.rule_num, h= options.h)
        centers = torch.cat((center_src, center_tgt), dim=0).tolist()
        sigma = torch.cat((sigma_src, sigma_tgt), dim=0).tolist()
        save_fcm_info(dataset_name, centers, sigma)
    trans_model = TransformerModel(vocab_src.n_words,
                             vocab_tgt.n_words,
                             options.trans.embedding_dim,
                             options.trans.nhead,
                             options.trans.hidden_size,
                             options.trans.nlayer,
                             options.trans.drop_out).to(options.device)
    loadmodel(trans_model, 'transformer-'+dataset_name)
    model = FuzzyS2S(vocab_src, vocab_tgt, options.feature_num, options.rule_num, center_src, sigma_src, center_tgt, sigma_tgt, trans_model).to(options.device)
    result = train(model, model_name, dataset_name, train_data, valid_data, test_data, vocab_src, vocab_tgt, pretrain_used, continual_learning,epoch_num=1)
    logger.remove(log_file)
    return result

def s2s_b_task(dataset_name, tokenizer, pretrain_used=False):
    model_name = 'fuzzys2s_b'
    log_file = logger.add(options.base_path+'output/log/'+model_name+'-'+dataset_name+'-'+str(datetime.date.today()) +'.log')
    logger.info('model %s on dataset %s start...' %(model_name, dataset_name))
    setup_seed(options.seed_id)
    train_data, valid_data, test_data, vocab_src, vocab_tgt = read_data(dataset_name, tokenizer)
    src_sen_feature_map, _ = combine_sen_feature_map(train_data,vocab_src, vocab_tgt)
    logger.info("src token clustering")
    center_src,sigma_src = fcm(src_sen_feature_map, cluster_num= options.rule_num, h= options.h)
    # logger.info("tgt token clustering")
    # center_tgt,sigma_tgt = fcm(tgt_sen_feature_map, cluster_num= options.rule_num, h= options.h)
    trans_model = TransformerModel(vocab_src.n_words,
                             vocab_tgt.n_words,
                             options.trans.embedding_dim,
                             options.trans.nhead,
                             options.trans.hidden_size,
                             options.trans.nlayer,
                             options.trans.drop_out).to(options.device)
    loadmodel(trans_model, 'transformer-'+dataset_name)
    model = FuzzyS2S_B(vocab_src, vocab_tgt, options.feature_num, options.rule_num, center_src, sigma_src, trans_model).to(options.device)
    result = train(model, model_name, dataset_name, train_data, valid_data[:10], test_data, vocab_src, vocab_tgt, pretrain_used, epoch_num=3)
    logger.remove(log_file)
    return result

def rnn_task(dataset_name, tokenizer, pretrain_used=False):
    model_name = 'rnn'
    log_file = logger.add(options.base_path+'output/log/'+model_name+'-'+dataset_name+'-'+str(datetime.date.today()) +'.log')
    logger.info('model %s on dataset %s start...' %(model_name, dataset_name))
    setup_seed(options.seed_id)
    train_data, valid_data, test_data, vocab_src, vocab_tgt = read_data(dataset_name, tokenizer)
    model = RnnModel(vocab_src.n_words,
                    vocab_tgt.n_words,
                    options.rnn.hidden_size,
                    options.rnn.nlayer,
                    options.rnn.drop_out).to(options.device)
    result = train(model, model_name, dataset_name, train_data, valid_data[:10], test_data, vocab_src, vocab_tgt, pretrain_used, epoch_num=20)
    logger.remove(log_file)
    return result

def trans_task(dataset_name, tokenizer, pretrain_used=False, continual_learning=False):
    model_name = 'transformer'
    log_file = logger.add(options.base_path+'output/log/'+model_name+'-'+dataset_name+'-'+str(datetime.date.today()) +'.log')
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
    result = train(model, model_name, dataset_name, train_data, valid_data, test_data, vocab_src, vocab_tgt, pretrain_used, continual_learning, epoch_num=10)
    logger.remove(log_file)
    return result

def t5_task(model_name, dataset_name, pretrain_used=True, fine_tuning=False, task_prefix="translate English to French: "):
    log_file = logger.add(options.base_path+'output/log/'+model_name+'-'+dataset_name+'-'+str(datetime.date.today()) +'.log')
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
    if fine_tuning:
        model_path = options.base_path+'output/finetune/'+model_name+'-'+dataset_name
        # model_finetune(model,model_path, tokenizer,task_prefix, train_sen_pairs, valid_sen_pairs[:10],epoch_num=10)
        model_finetune2(model,model_path, tokenizer,task_prefix, train_sen_pairs, valid_sen_pairs[:10],
                        max_src_length=512,
                        max_tgt_length=512,
                        epoch_num=10)

    for src,tgt in tqdm(test_sen_pairs,"test data"):
        if len(src) > options.sen_len_max:
            src = src[:options.sen_len_max]
        if len(tgt) > options.sen_len_max:
            tgt = tgt[:options.sen_len_max]
        if len(src) == 7 or len(tgt) == 7:
            continue
        if len(src) == 0 or len(tgt) == 0:
            continue
        if len(src) == 1 or len(tgt) ==1:
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
    logger.info("[%s-%s]acc: %.2f, sen_bleu: %.2f, meteor: %.2f" %(result['model_name'],result['dataset_name'] ,result['acc'], result['sen_bleu'], result['meteor']))
    logger.remove(log_file)
    return result

def mt5_task(model_name, dataset_name, pretrain_used=True):
    log_file = logger.add(options.base_path+'output/log/'+model_name+'-'+dataset_name+'-'+str(datetime.date.today()) +'.log')
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
        if len(src) > options.sen_len_max:
            src = src[:options.sen_len_max]
        if len(tgt) > options.sen_len_max:
            tgt = tgt[:options.sen_len_max]
        if len(src) == 0 or len(tgt) == 0:
            continue
        if len(src) == 1 or len(tgt) ==1:
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
    logger.info("[%s-%s]acc: %.2f, sen_bleu: %.2f, meteor: %.2f" %(result['model_name'],result['dataset_name'] ,result['acc'], result['sen_bleu'], result['meteor']))
    logger.remove(log_file)
    return result

def opus_mt_task(model_name, dataset_name, pretrain_used=True, fine_tuning=False):
    log_file = logger.add(options.base_path+'output/log/'+model_name+'-'+dataset_name+'-'+str(datetime.date.today()) +'.log')
    logger.info('model %s on dataset %s start...' %(model_name, dataset_name))
    setup_seed(options.seed_id)
    train_sen_pairs, valid_sen_pairs, test_sen_pairs = read_data(dataset_name, sen_out=True)
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr").to(options.device)
    if fine_tuning:
        model_path = options.base_path+'output/finetune/'+model_name+'-'+dataset_name
        task_prefix = ''
        model_finetune2(model,model_path, tokenizer,task_prefix, train_sen_pairs, valid_sen_pairs[:10],
                        max_src_length=512,
                        max_tgt_length=512,
                        epoch_num=10)
    model_param(model)
    max_source_length = 512
    max_target_length = 128
    count = 0
    total_metrics = MetricsValue()
    evaluator = Evaluator()
    # task_prefix = "translate English to French: "
    for src,tgt in tqdm(test_sen_pairs,"test data"):
        if len(src) > options.sen_len_max:
            src = src[:options.sen_len_max]
        if len(tgt) > options.sen_len_max:
            tgt = tgt[:options.sen_len_max]
        if len(src) == 0 or len(tgt) == 0:
            continue
        if len(src) == 1 or len(tgt) ==1:
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
    logger.info("[%s-%s]acc: %.2f, sen_bleu: %.2f, meteor: %.2f" %(result['model_name'],result['dataset_name'] ,result['acc'], result['sen_bleu'], result['meteor']))
    logger.remove(log_file)
    return result

def codet5_task(model_name, dataset_name, pretrain_used=True):
    log_file = logger.add(options.base_path+'output/log/'+model_name+'-'+dataset_name+'-'+str(datetime.date.today()) +'.log')
    logger.info('model %s on dataset %s start...' %(model_name, dataset_name))
    setup_seed(options.seed_id)
    train_sen_pairs, valid_sen_pairs, test_sen_pairs = read_data(dataset_name, sen_out=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer.add_special_tokens({'pad_token': '<pad>', 'eos_token': '<eos>', 'bos_token': '<bos>', 'unk_token': '<unk>'})
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(options.device)
    model_param(model)
    # # fine tuning with dataset
    task_prefix = 'Code Generation: '
    model_path = options.base_path+'output/finetune/'+model_name+'-'+dataset_name
    model_finetune(model,model_path, tokenizer,task_prefix, train_sen_pairs, valid_sen_pairs[:10],epoch_num=1)

    # max_source_length = 512
    max_target_length = 1024
    count = 0
    total_metrics = MetricsValue()
    evaluator = Evaluator()
    for src,tgt in tqdm(test_sen_pairs,"test data"):
        if len(src) > options.sen_len_max:
            src = src[:options.sen_len_max]
        if len(tgt) > options.sen_len_max:
            tgt = tgt[:options.sen_len_max]
        if len(src) == 0 or len(tgt) == 0:
            continue
        if len(src) == 1 or len(tgt) ==1:
            continue
        tgt_ids = tokenizer(tgt, return_tensors="pt").input_ids.to(options.device)
        src_ids= tokenizer(task_prefix+src, return_tensors="pt").input_ids.to(options.device)
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
    logger.info("[%s-%s]acc: %.2f, sen_bleu: %.2f, meteor: %.2f" %(result['model_name'],result['dataset_name'] ,result['acc'], result['sen_bleu'], result['meteor']))
    logger.remove(log_file)
    return result

def codegen_task(model_name, dataset_name, pretrain_used=True):
    log_file = logger.add(options.base_path+'output/log/'+model_name+'-'+dataset_name+'-'+str(datetime.date.today()) +'.log')
    logger.info('model %s on dataset %s start...' %(model_name, dataset_name))
    setup_seed(options.seed_id)
    train_sen_pairs, valid_sen_pairs, test_sen_pairs = read_data(dataset_name, sen_out=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '<pad>',
                                  'eos_token': '<eos>',
                                  'bos_token': '<bos>',
                                  'unk_token': '<unk>',
                                  'cls_token': '<cls>',
                                  'mask_token': '<mask>',
                                  'sep_token': '<sep>'})
    model = AutoModelForCausalLM.from_pretrained(model_name).to(options.device)
    model_param(model)
    # # fine tuning with dataset
    task_prefix = 'generate python: '
    model_path = options.base_path+'output/finetune/'+model_name+'-'+dataset_name
    model_finetune(model,model_path, tokenizer,task_prefix, train_sen_pairs, valid_sen_pairs[:10],
                   max_src_length=256,
                   max_tgt_length=256,
                   epoch_num=3
                   )

    # max_source_length = 512
    max_target_length = 512
    count = 0
    total_metrics = MetricsValue()
    evaluator = Evaluator()
    for src,tgt in tqdm(test_sen_pairs,"test data"):
        if len(src) > options.sen_len_max:
            src = src[:options.sen_len_max]
        if len(tgt) > options.sen_len_max:
            tgt = tgt[:options.sen_len_max]
        if len(src) == 0 or len(tgt) == 0:
            continue
        if len(src) == 1 or len(tgt) ==1:
            continue
        tgt_ids = tokenizer(tgt, return_tensors="pt").input_ids.to(options.device)
        src_ids= tokenizer(src, return_tensors="pt").input_ids.to(options.device)
        outputs = model.generate(src_ids, max_length=max_target_length)
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
    logger.info("[%s-%s]acc: %.2f, sen_bleu: %.2f, meteor: %.2f" %(result['model_name'],result['dataset_name'] ,result['acc'], result['sen_bleu'], result['meteor']))
    logger.remove(log_file)
    return result


def codebert_task(model_name, dataset_name, pretrain_used=True):
    log_file = logger.add(options.base_path+'output/log/'+model_name+'-'+dataset_name+'-'+str(datetime.date.today()) +'.log')
    logger.info('model %s on dataset %s start...' %(model_name, dataset_name))
    setup_seed(options.seed_id)
    train_sen_pairs, valid_sen_pairs, test_sen_pairs = read_data(dataset_name, sen_out=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '<pad>',
                                  'eos_token': '<eos>',
                                  'bos_token': '<bos>',
                                  'unk_token': '<unk>',
                                  'cls_token': '<cls>',
                                  'mask_token': '<mask>',
                                  'sep_token': '<sep>'})
    model = AutoModelForCausalLM.from_pretrained(model_name).to(options.device)
    model_param(model)
    # # fine tuning with dataset
    task_prefix = ''
    model_path = options.base_path+'output/finetune/'+model_name+'-'+dataset_name
    model_finetune2(model,model_path, tokenizer,task_prefix, train_sen_pairs, valid_sen_pairs[:10],
                   max_src_length=512,
                   max_tgt_length=512,
                   epoch_num=10)

    # max_source_length = 512
    max_target_length = 512
    count = 0
    total_metrics = MetricsValue()
    evaluator = Evaluator()
    for src,tgt in tqdm(test_sen_pairs,"test data"):
        if len(src) > options.sen_len_max:
            src = src[:options.sen_len_max]
        if len(tgt) > options.sen_len_max:
            tgt = tgt[:options.sen_len_max]
        if len(src) == 0 or len(tgt) == 0:
            continue
        if len(src) == 1 or len(tgt) ==1:
            continue
        tgt_ids = tokenizer(tgt, return_tensors="pt").input_ids.to(options.device)
        src_ids= tokenizer(src, return_tensors="pt").input_ids.to(options.device)
        outputs = model.generate(src_ids, max_length=max_target_length)
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
    logger.info("[%s-%s]acc: %.2f, sen_bleu: %.2f, meteor: %.2f" %(result['model_name'],result['dataset_name'] ,result['acc'], result['sen_bleu'], result['meteor']))
    logger.remove(log_file)
    return result

def pegasus_task(model_name, dataset_name, pretrain_used=True, fine_tuning=False, task_prefix=""):
    log_file = logger.add(options.base_path+'output/log/'+model_name+'-'+dataset_name+'-'+str(datetime.date.today()) +'.log')
    logger.info('model %s on dataset %s start...' %(model_name, dataset_name))
    setup_seed(options.seed_id)
    train_sen_pairs, valid_sen_pairs, test_sen_pairs = read_data(dataset_name, sen_out=True)
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(options.device)
    model_param(model)
    max_source_length = 1024
    max_target_length = 128
    count = 0
    total_metrics = MetricsValue()
    evaluator = Evaluator()
    if fine_tuning:
        model_path = options.base_path+'output/finetune/'+model_name+'-'+dataset_name
        model_finetune(model,model_path, tokenizer,task_prefix, train_sen_pairs, valid_sen_pairs[:10],epoch_num=1)

    for src,tgt in tqdm(test_sen_pairs,"test data"):
        if len(src) > options.sen_len_max:
            src = src[:options.sen_len_max]
        if len(tgt) > options.sen_len_max:
            tgt = tgt[:options.sen_len_max]
        if len(src) == 0 or len(tgt) == 0:
            continue
        if len(src) == 1 or len(tgt) ==1:
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
    logger.info("[%s-%s]acc: %.2f, sen_bleu: %.2f, meteor: %.2f" %(result['model_name'],result['dataset_name'] ,result['acc'], result['sen_bleu'], result['meteor']))
    logger.remove(log_file)
    return result

def pegasus_x_task(model_name, dataset_name, pretrain_used=True, fine_tuning=False, task_prefix=""):
    log_file = logger.add(options.base_path+'output/log/'+model_name+'-'+dataset_name+'-'+str(datetime.date.today()) +'.log')
    logger.info('model %s on dataset %s start...' %(model_name, dataset_name))
    setup_seed(options.seed_id)
    train_sen_pairs, valid_sen_pairs, test_sen_pairs = read_data(dataset_name, sen_out=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = PegasusXForConditionalGeneration.from_pretrained(model_name).to(options.device)
    model_param(model)
    max_source_length = 1024
    max_target_length = 128
    count = 0
    total_metrics = MetricsValue()
    evaluator = Evaluator()
    if fine_tuning:
        model_path = options.base_path+'output/finetune/'+model_name+'-'+dataset_name
        model_finetune(model,model_path, tokenizer,task_prefix, train_sen_pairs, valid_sen_pairs[:10],epoch_num=1)

    for src,tgt in tqdm(test_sen_pairs,"test data"):
        if len(src) > options.sen_len_max:
            src = src[:options.sen_len_max]
        if len(tgt) > options.sen_len_max:
            tgt = tgt[:options.sen_len_max]
        if len(src) == 0 or len(tgt) == 0:
            continue
        if len(src) == 1 or len(tgt) ==1:
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
    logger.info("[%s-%s]acc: %.2f, sen_bleu: %.2f, meteor: %.2f" %(result['model_name'],result['dataset_name'] ,result['acc'], result['sen_bleu'], result['meteor']))
    logger.remove(log_file)
    return result


def run():
    # datasets =['opus_euconst', 'tatoeba','wmt14', 'ubuntu']
    # datasets =['hearthstone', 'magic', 'geo',  'spider']
    # datasets =['cnn_dailymail', 'samsum',  'billsum', 'xlsum']
    datasets= ['cnn_dailymail']
    results = []
    for dataset in datasets:
        if options.ablation.fuzzy_tokenizer:
            tokenizer = get_fuzzy_tokenizer(dataset)
        else:
            tokenizer = get_tokenizer("basic_english")
        # tokenizer = get_base_tokenizer('bert-base-uncased')
        # result = t5_task('t5-small',dataset,fine_tuning=True)
        # results.append(result)
        # result = t5_task('t5-base',dataset,fine_tuning=True)
        # results.append(result)
        # result = t5_task('t5-large',dataset, fine_tuning=True)
        # results.append(result)
        # result = t5_task('t5-small',dataset,fine_tuning=True, task_prefix="translate as to bs: ")
        # results.append(result)
        # result = t5_task('t5-base',dataset,fine_tuning=True, task_prefix="translate as to bs: ")
        # results.append(result)
        # result = t5_task('t5-large',dataset,fine_tuning=True, task_prefix="translate as to bs: ")
        # results.append(result)
        # result = mt5_task('google/mt5-small',dataset)
        # results.append(result)
        # result = opus_mt_task('Helsinki-NLP/opus-mt-en-fr', dataset, fine_tuning=True)
        # results.append(result)
        result = trans_task(dataset, tokenizer,pretrain_used=False)
        results.append(result)
        # for i in range(10):
        #     options.rule_num = i+ 1
        result = s2s_task(dataset, tokenizer,pretrain_used=False)
        results.append(result)
        # result = s2s_b_task(dataset, tokenizer,pretrain_used=False)
        # results.append(result)
        # result = rnn_task(dataset, tokenizer,pretrain_used=False)
        # results.append(result)
        # result = codet5_task('Salesforce/codet5-small',dataset)
        # results.append(result)
        # result = codet5_task('Salesforce/codet5-base',dataset)
        # results.append(result)
        # result = codet5_task('Salesforce/codet5-large',dataset)
        # results.append(result)
        # result = codegen_task('Salesforce/codegen-350M-mono',dataset)
        # results.append(result)
        # result = codebert_task('microsoft/codebert-base-mlm',dataset)
        # results.append(result)
        # result = t5_task('t5-small',dataset, task_prefix='summarize: ')
        # results.append(result)
        # result = t5_task('t5-base',dataset,task_prefix='summarize: ')
        # results.append(result)
        # result = t5_task('t5-large',dataset, task_prefix='summarize: ')
        # results.append(result)
        # result = pegasus_task('google/pegasus-xsum',dataset)
        # results.append(result)
        # result = pegasus_x_task('google/pegasus-x-large',dataset)
        # results.append(result)
    log_file = logger.add(options.base_path+'output/result/result-'+str(datetime.date.today()) +'.log')
    for result in results:
        logger.info("------------------------------------result------------------------------------------" )
        if 'epoch' in result:
            epoch = result['epoch']
        else:
            epoch = 0
        if 's2s_rule' in result:
            s2s_rule = result['s2s_rule']
        else:
            s2s_rule = 0
        if 'tokenizer_rule' in result:
            tokenizer_rule = result['tokenizer_rule']
        else:
            tokenizer_rule = 0
        logger.info("epoch: %d, embedding: %d, tokenizer_rule: %d, s2s_rule:%d" %(epoch, options.trans.embedding_dim, tokenizer_rule, s2s_rule))
        logger.info("ablation fuzzy_tokenizer: %s, fuzzy_vae: %s" %(str(options.ablation.fuzzy_tokenizer), str(options.ablation.fuzzy_vae)))
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
