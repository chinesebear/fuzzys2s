from transformers import AutoTokenizer
from torchtext.data import get_tokenizer
import numpy as np
from datasets import load_dataset,load_from_disk
import os
import datetime
from loguru import logger
from setting import options,Options
import itertools
from torch import nn
import torch
import torch.nn.functional as F
import math
from fcmeans import FCM
from loaddata import Vocab, read_raw_data, read_data
from tqdm import tqdm
import csv
import pickle

def read_dataset(name, subpath):
    dataset_path = options.base_path+"output/"+name+"/"+subpath+"/"
    if os.path.exists(dataset_path):
        dataset = load_from_disk(dataset_path)
    else :
        if subpath == '':
            dataset = load_dataset(name)
        else:
            dataset = load_dataset(name,subpath)
        dataset.save_to_disk(dataset_path)
    logger.info("read %s-%s done" %(name,subpath))
    return dataset

def read_raw_lines(dataset, data_type, src_lang, tgt_lang):
    raw_len = dataset[data_type].num_rows
    raw_iter = iter(dataset[data_type])
    raw_lines = np.empty(raw_len).tolist()
    for i in range(raw_len):
        data = next(raw_iter)
        src = data['translation'][src_lang]
        tgt = data['translation'][tgt_lang]
        raw_lines[i] =  src + ' ' + tgt
    return raw_lines

def get_wmt16_iter():
    dataset = read_dataset("wmt16",'de-en')
    train_raw_iter = iter(dataset['train'])
    valid_raw_iter = iter(dataset['validation'])
    test_raw_iter = iter(dataset['test'])
    raw_iter = itertools.chain(train_raw_iter, valid_raw_iter, test_raw_iter)
    raw_len = dataset['train'].num_rows + dataset['validation'].num_rows + dataset['test'].num_rows
    src_lang = 'en'
    tgt_lang = 'de'
    return raw_iter, raw_len, src_lang, tgt_lang

def get_wmt14_iter():
    dataset = read_dataset("wmt14",'fr-en')
    train_raw_iter = iter(dataset['train'])
    valid_raw_iter = iter(dataset['validation'])
    test_raw_iter = iter(dataset['test'])
    raw_iter = itertools.chain(train_raw_iter, valid_raw_iter, test_raw_iter)
    raw_len = dataset['train'].num_rows + dataset['validation'].num_rows + dataset['test'].num_rows
    src_lang = 'en'
    tgt_lang = 'fr'
    return raw_iter, raw_len, src_lang, tgt_lang

def get_opus100_iter():
    dataset = read_dataset("opus100",'en-fr')
    train_raw_iter = iter(dataset['train'])
    valid_raw_iter = iter(dataset['validation'])
    test_raw_iter = iter(dataset['test'])
    raw_iter = itertools.chain(train_raw_iter, valid_raw_iter, test_raw_iter)
    raw_len = dataset['train'].num_rows + dataset['validation'].num_rows + dataset['test'].num_rows
    src_lang = 'en'
    tgt_lang = 'fr'
    return raw_iter, raw_len, src_lang, tgt_lang

def get_dataset_iter(dataset):
    if dataset =="wmt16":
        return get_wmt16_iter()
    elif dataset == 'wmt14':
        return get_wmt14_iter()
    elif dataset == 'opus100':
        return get_opus100_iter()

def read_raw_lines(raw_iter, raw_len, src_lang, tgt_lang):
    raw_lines = np.empty(raw_len).tolist()
    for i in range(raw_len):
        data = next(raw_iter)
        src = data['translation'][src_lang]
        tgt = data['translation'][tgt_lang]
        raw_lines[i] =  src + ' ' + tgt
    return raw_lines

def train_tokenizer(dataset, vocab_size):
    tokenizer_path = options.base_path+"output/vs"+str(vocab_size)+"_"+dataset+"_tokenizer"
    if os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        logger.info("dataset %s, vocab size %d , tokenizer found" %(dataset, vocab_size))
        return tokenizer
    logger.info("vs%d_%s_tokenizer train start..." %(vocab_size, dataset))
    raw_iter, raw_len, src_lang, tgt_lang = get_dataset_iter(dataset)
    count = 0
    step = 20000
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    while(count < raw_len):
        if (raw_len - count) % step == 0:
            read_len = step
        else:
            read_len = (raw_len - count) % step
        raw_lines = read_raw_lines(raw_iter, read_len, src_lang, tgt_lang)
        tokenizer = tokenizer.train_new_from_iterator(raw_lines, vocab_size=vocab_size)
        count = count + read_len
    tokenizer.save_pretrained(tokenizer_path)
    logger.info("dataset %s, vocab size %d , tokenizer train done" %(dataset, vocab_size))
    return tokenizer

def train_tokenizer_on_dataset(tokenizer, dataset, vocab_size):
    logger.info("train tokenizer on %s, vocab size %d" %(dataset, vocab_size))
    train_data, valid_data,  test_data = read_data(dataset,sen_out=True)
    data = train_data+valid_data+test_data
    lines = []
    for d in tqdm(data,dataset):
        lines.append(d[0])
        lines.append(d[1])
    tokenizer.train_new_from_iterator(lines, vocab_size=vocab_size)
    return tokenizer

def train_tokenizer_on_large_dataset(tokenizer, dataset, vocab_size):
    logger.info("train tokenizer on %s, vocab size %d" %(dataset, vocab_size))
    raw_iter, raw_len, src_lang, tgt_lang = get_dataset_iter(dataset)
    count = 0
    step = 20000
    while(count < raw_len):
        if (raw_len - count) % step == 0:
            read_len = step
        else:
            read_len = (raw_len - count) % step
        raw_lines = read_raw_lines(raw_iter, read_len, src_lang, tgt_lang)
        tokenizer = tokenizer.train_new_from_iterator(raw_lines, vocab_size=vocab_size)
        count = count + read_len
    return tokenizer

def train_fuzzy_tokenizer(vocab_size):
    tokenizer_path = options.base_path+"output/fuzzy_vs"+str(vocab_size)+"_tokenizer"
    if os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        logger.info("multi-scale Tokenizer, vocab size %d , tokenizer found" %(vocab_size))
    else:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    logger.info("fuzzy_vs%d_tokenizer train start..." %(vocab_size))
    # tokenizer = train_tokenizer_on_large_dataset(tokenizer, 'wmt14', vocab_size)
    tokenizer = train_tokenizer_on_dataset(tokenizer, 'ubuntu', vocab_size)
    tokenizer = train_tokenizer_on_dataset(tokenizer, 'opus_euconst', vocab_size)
    tokenizer = train_tokenizer_on_dataset(tokenizer, 'hearthstone', vocab_size)
    tokenizer = train_tokenizer_on_dataset(tokenizer, 'magic', vocab_size)
    tokenizer = train_tokenizer_on_dataset(tokenizer, 'geo', vocab_size)
    tokenizer = train_tokenizer_on_dataset(tokenizer, 'spider', vocab_size)
    tokenizer = train_tokenizer_on_dataset(tokenizer, 'xlsum', vocab_size)
    tokenizer = train_tokenizer_on_dataset(tokenizer, 'billsum', vocab_size)
    tokenizer = train_tokenizer_on_dataset(tokenizer, 'samsum', vocab_size)
    tokenizer = train_tokenizer_on_dataset(tokenizer, 'cnn_dailymail', vocab_size)
    tokenizer.save_pretrained(tokenizer_path)
    logger.info("fuzzy_vs%d_tokenizer train done" %(vocab_size))
    return tokenizer

def get_base_tokenizer(name):
    tokenizer_path = options.base_path+"output/"+name+"/"
    if os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(name)
        tokenizer.save_pretrained(tokenizer_path)
    return tokenizer.tokenize

def gen_token_feature(token, vocab):
    size = len(token)
    if token not in vocab.word2index:
        vocab.addWord(token)
    count = vocab.word2count[token]
    if size > options.size_max:
        size = 1.0
    else:
        size = size / options.size_max
    if count > options.count_max:
        count = 1.0
    else:
        count = count/options.count_max
    feature = [size, count]
    return feature

def gen_token_features(tokens, vocab):
    tok_len = len(tokens)
    features= np.empty([tok_len, 2])
    for i in range(tok_len):
        token = tokens[i]
        features[i] = gen_token_feature(token, vocab)
    return features

def gen_raw_features(raw_iter, raw_len , src_lang, tgt_lang, vocab, tokenizer):
    tok_total = 0
    raw_tokens = np.empty(raw_len).tolist()
    for i in range(raw_len):
        data = next(raw_iter)
        src = data['translation'][src_lang]
        tgt = data['translation'][tgt_lang]
        line =  src + ' ' + tgt
        tokens = tokenizer(line)
        raw_tokens[i] = tokens
        tok_total = tok_total + len(tokens)
    tok_features = np.empty([tok_total, 2])
    offset = 0
    for i in range(raw_len):
        tokens = raw_tokens[i]
        features = gen_token_features(tokens, vocab)
        for j in range(len(features)):
            tok_features[j+offset] = features[j]
        offset = offset + len(features)
    return tok_features, tok_total

def order_point(points):
    # input type: np.array
    idxs =  np.argsort(points, axis=0)
    plen = len(points)
    pwidth = len(points[0])
    output = np.empty((plen,pwidth))
    for i in range(plen):
        output[i] = points[idxs[i][0]] # order by feature_0
    return output

def order_center(centers, sigma):
    # input type: np.array
    points = []
    for i in range(len(centers)):
        points.append([centers[i][0], centers[i][1], sigma[i][0], sigma[i][1]])
    points = order_point(points)
    for i in range(len(centers)):
        centers[i] = [points[i][0], points[i][1]]
        sigma[i] = [points[i][2], points[i][3]]
    return centers, sigma

def fcm_cluster(data, cluster_num, h):
    # input data type is numpy
    logger.info("fcm clustering...")
    feature_num = options.feature_num
    fcm = FCM(n_clusters=cluster_num,max_iter=options.iter_num)
    total = len(data)
    logger.info("tokens:%d" %(total))
    token_subclass_num = 1000
    step = int(total/token_subclass_num)
    if step > 20000:
        raw_centers = []
        for i in tqdm(range(token_subclass_num), 'sub cluster'):
            raw_data = data[i*step:i*step + step]
            fcm.fit(raw_data)
            raw_centers.extend(fcm.centers.tolist())
        data = np.array(raw_centers)
    fcm.fit(data)
    centers = fcm.centers
    centers = centers.tolist()
    logger.info("cluster center: %d" %(len(centers)))
    membership = fcm.soft_predict(data)
    data_num = len(data)
    sigma = []
    for i in range(cluster_num):
        feature_sigma = []
        for j in range(feature_num):
            a = 0
            b = 0
            for k in  range(data_num):
                x = data[k][j]
                a = a + membership[k][i]* ((x-centers[i][j]) ** 2)
                b = b + membership[k][i]
            feature_sigma.append(h*a/b)
        sigma.append(feature_sigma)
    logger.info("cluster sigma: %d" %(len(sigma)))
    centers,sigma = order_center(centers, sigma)
    sigma = np.array(sigma)
    centers = np.array(centers)
    return centers,sigma

def save_cluster_info(dataset, centers, sigma):
    info_path = options.base_path+'output/fuzzy_tokenizer/'+ dataset+'_'+str(options.tokenizer.fuzzy_rule_num)+'_rule_cluster_info.csv'
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

def load_cluster_info(dataset):
    info_path = options.base_path+'output/fuzzy_tokenizer/'+ dataset+'_'+str(options.tokenizer.fuzzy_rule_num)+'_rule_cluster_info.csv'
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
    return centers, sigma

def save_tokenizer_vocab(dataset, vocab):
    logger.info("save %s tokenizer vocab..." %(dataset))
    vocab_path = options.base_path+'output/fuzzy_tokenizer/'+ dataset+'_'+str(options.tokenizer.fuzzy_rule_num)+'_rule_tokenizer_vocab.pickle'
    with open(vocab_path, 'wb') as f: # open file with write-mode
        pickle.dump(vocab, f) # serialize and save object

def load_tokenizer_vocab(dataset):
    logger.info("load %s tokenizer vocab..." %(dataset))
    vocab_path = options.base_path+'output/fuzzy_tokenizer/'+ dataset+'_'+str(options.tokenizer.fuzzy_rule_num)+'_rule_tokenizer_vocab.pickle'
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)   # read file and build object
    return vocab

def token_cluster_iter(dataset, rule_num):
    raw_iter, raw_len, src_lang, tgt_lang = get_dataset_iter(dataset)
    vocab = Vocab(dataset)
    tokenizer  = get_tokenizer("basic_english")
    for _ in tqdm(range(raw_len)):
        data = next(raw_iter)
        src = data['translation'][src_lang]
        tgt = data['translation'][tgt_lang]
        tokens = tokenizer(src)
        vocab.addTokens(tokens)
        tokens = tokenizer(tgt)
        vocab.addTokens(tokens)
    save_tokenizer_vocab(dataset, vocab)
    raw_iter, raw_len, src_lang, tgt_lang = get_dataset_iter(dataset)
    centers = []
    count = 0
    step = 10000
    while(count < raw_len):
        if (raw_len - count) % step == 0:
            read_len = step
        else:
            read_len = (raw_len - count) % step
        count = count + read_len
        tok_features, _ = gen_raw_features(raw_iter, read_len, src_lang, tgt_lang, vocab, tokenizer)
        raw_centers,_ = fcm_cluster(tok_features, cluster_num=rule_num, h=options.h)
        centers.append(raw_centers)
    centers = np.array(centers).reshape((-1,2))
    centers, sigma = fcm_cluster(centers, cluster_num=rule_num, h=options.h)
    save_cluster_info(dataset, centers, sigma)
    logger.info('%s , centers:%s %s %s' %(dataset, str(centers[0]), str(centers[1]), str(centers[2])))
    logger.info('%s , sigma:%s %s %s' %(dataset, str(sigma[0]), str(sigma[1]), str(sigma[2])))
    return centers, sigma, vocab

def token_cluster(tokens, rule_num):
    vocab = Vocab('cluster')
    for data in tqdm(tokens, 'build vocab for cluster'):
        vocab.addTokens(data[0])
        vocab.addTokens(data[1])
    token_features = []
    for data in tqdm(tokens, 'token featurs'):
        features = gen_token_features(data[0], vocab)
        token_features.extend(features)
        features = gen_token_features(data[1], vocab)
        token_features.extend(features)
    token_features = np.array(token_features)
    centers,sigma = fcm_cluster(token_features, cluster_num=rule_num, h=options.h)
    return centers, sigma, vocab

class FuzzyTokenizer(nn.Module):
    def __init__(self,feature_in, rule_num, centers, sigma, vocab):
        super(FuzzyTokenizer, self).__init__()
        self.center =centers
        self.sigma =sigma
        self.vocab = vocab
        self.rule_num = rule_num
        self.feature_num = feature_in
        coarse_tokenzier = get_tokenizer("basic_english")
        middle_tokenizer = get_base_tokenizer('bert-base-uncased') #vs30359
        fine_tokenizer = get_base_tokenizer('fuzzy_vs4000_tokenizer')
        tokenizers = [coarse_tokenzier,
                    get_base_tokenizer('fuzzy_vs100000_tokenizer'),
                    get_base_tokenizer('fuzzy_vs80000_tokenizer'),
                    get_base_tokenizer('fuzzy_vs60000_tokenizer'),
                    get_base_tokenizer('fuzzy_vs50000_tokenizer'),
                    get_base_tokenizer('fuzzy_vs40000_tokenizer'),
                    middle_tokenizer,#vs30359
                    get_base_tokenizer('fuzzy_vs30000_tokenizer'),
                    get_base_tokenizer('fuzzy_vs20000_tokenizer'),
                    get_base_tokenizer('fuzzy_vs10000_tokenizer'),
                    get_base_tokenizer('fuzzy_vs9000_tokenizer'),
                    get_base_tokenizer('fuzzy_vs8000_tokenizer'),
                    get_base_tokenizer('fuzzy_vs7000_tokenizer'),
                    get_base_tokenizer('fuzzy_vs6000_tokenizer'),
                    get_base_tokenizer('fuzzy_vs5000_tokenizer'),
                    fine_tokenizer]
        self.tokenizers = []
        tokenizer_num = len(tokenizers)
        if rule_num == 1:
            self.tokenizers = [coarse_tokenzier]
        # elif rule_num == 2:
        #     self.tokenizers = [coarse_tokenzier, middle_tokenizer]
        # elif rule_num == 3:
        #     self.tokenizers = [coarse_tokenzier, middle_tokenizer, fine_tokenizer]
        else:
            for i in range(rule_num):
                idx = int(i * tokenizer_num / rule_num)
                self.tokenizers.append(tokenizers[idx])
    def fuzzy_layer(self, x):
        delta = x - self.center # torch tensor broadcast
        value = torch.square(torch.div(delta , self.sigma))
        membership = torch.exp(-(value /2))
        return membership
    def fire_layer(self,membership):
        # membership array
        # rule_num * feature_num
        products = torch.prod(membership, 1)
        return products.float()
    def norm_layer(self, products):
        sum = torch.sum(products)
        if sum.item() == 0:
            # print("sum is 0")
            return products
        products = products/sum
        return products
    def emit_layer(self, products, token):
        _, idx = torch.max(products,dim=-1)
        subtokens = self.tokenizers[idx](token)
        return subtokens
    def forward(self, seq):
        tokenizer  = get_tokenizer("basic_english")
        tokens = tokenizer(seq)
        output = []
        for token in tokens:
            x = gen_token_feature(token, self.vocab)
            x = torch.tensor(x).to(options.device)
            membership = self.fuzzy_layer(x)
            products = self.fire_layer(membership)
            products = self.norm_layer(products)
            subtokens = self.emit_layer(products, token)
            output.extend(subtokens)
        return output


def train_tokenizers():
    log_file = logger.add(options.base_path+'output/log/tokenizers-'+str(datetime.date.today()) +'.log')
    # train_tokenizer('wmt14', 1000)
    # train_tokenizer('wmt14', 4000)
    # train_tokenizer('wmt14', 10000)
    # train_tokenizer('wmt14', 50000)
    # train_tokenizer('wmt14', 100000)
    # train_tokenizer('wmt14', 150000)
    # train_tokenizer('wmt14', 200000)
    # train_tokenizer('wmt16', 1000)
    # train_tokenizer('wmt16', 4000)
    # train_tokenizer('wmt16', 10000)
    # train_tokenizer('opus100', 1000)
    # train_tokenizer('opus100', 4000)
    # train_tokenizer('opus100', 10000)
    epoch = 10
    for _ in range(epoch):
        train_fuzzy_tokenizer(100000)
        train_fuzzy_tokenizer(80000)
        train_fuzzy_tokenizer(60000)
        train_fuzzy_tokenizer(50000)
        train_fuzzy_tokenizer(40000)
        train_fuzzy_tokenizer(30000)
        train_fuzzy_tokenizer(20000)
        train_fuzzy_tokenizer(10000)
        train_fuzzy_tokenizer(9000)
        train_fuzzy_tokenizer(8000)
        train_fuzzy_tokenizer(7000)
        train_fuzzy_tokenizer(6000)
        train_fuzzy_tokenizer(5000)
        train_fuzzy_tokenizer(4000)
        train_fuzzy_tokenizer(3000)
        train_fuzzy_tokenizer(2000)
    train_fuzzy_tokenizer(1000)
    logger.remove(log_file)

def dataset_token_clustering():
    log_file=logger.add(options.base_path+'output/log/fuzzy-tokenizer-clustering-'+str(datetime.date.today()) +'.log')
    token_cluster_iter('wmt14', options.rule_num)
    token_cluster_iter('wmt16', options.rule_num)
    token_cluster_iter('opus100', options.rule_num)
    logger.remove(log_file)

def get_fuzzy_tokenizer(dataset):
    info_path = info_path = options.base_path+'output/fuzzy_tokenizer/'+ dataset+'_'+str(options.tokenizer.fuzzy_rule_num)+'_rule_cluster_info.csv'
    vocab_path = options.base_path+'output/fuzzy_tokenizer/'+ dataset+'_'+str(options.tokenizer.fuzzy_rule_num)+'_rule_tokenizer_vocab.pickle'
    if os.path.exists(info_path) and os.path.exists(vocab_path):
        centers, sigma = load_cluster_info(dataset)
        vocab = load_tokenizer_vocab(dataset)
    else:
        tokenizer = get_tokenizer("basic_english")
        train_tokens, valid_tokens,  test_tokens = read_raw_data(dataset, tokenizer)
        tokens =  train_tokens + valid_tokens + test_tokens
        centers, sigma,vocab = token_cluster(tokens, options.tokenizer.fuzzy_rule_num)
        save_cluster_info(dataset, centers, sigma)
        save_tokenizer_vocab(dataset, vocab)
        centers = torch.from_numpy(centers).to(options.device)
        sigma = torch.from_numpy(sigma).to(options.device)
    fuzzy_tonkenizer = FuzzyTokenizer(options.tokenizer.fuzzy_feature_num, options.tokenizer.fuzzy_rule_num, centers, sigma, vocab)
    return fuzzy_tonkenizer

def run():
    # dataset_token_clustering()
    # points = [[1,2,3,4,5,6],[7,2,3,4,5,6],[3,2,3,4,5,6],[5,2,3,4,5,6]]
    # pout = order_point(points)
    # print(pout)
    # dataset = 'atis'
    # if dataset == 'wmt14' or dataset == 'wmt16' or dataset == 'opus100' or dataset == 'tatoeba':
    #     centers, sigma = load_cluster_info(dataset)
    #     vocab = load_tokenizer_vocab(dataset)
    # else:
    #     tokenizer = get_tokenizer("basic_english")
    #     train_tokens, valid_tokens,  test_tokens = read_raw_data(dataset, tokenizer)
    #     tokens =  train_tokens + valid_tokens + test_tokens
    #     centers, sigma,vocab = token_cluster(tokens, options.rule_num)
    #     save_cluster_info(dataset, centers, sigma)
    #     save_tokenizer_vocab(dataset, vocab)
    #     centers = torch.from_numpy(centers).to(options.device)
    #     sigma = torch.from_numpy(sigma).to(options.device)
    # fuzzy_tonkenizer = FuzzyTokenizer(options.feature_num, options.rule_num, centers, sigma, vocab)
    # for data in tokens[:100]:
    #     sen = " ".join(i for i in data[0])
    #     out = fuzzy_tonkenizer(sen)
    #     print(out)
    #     sen = " ".join(i for i in data[1])
    #     out = fuzzy_tonkenizer(sen)
    #     print(out)
    train_tokenizers()
    return 0

# run()
