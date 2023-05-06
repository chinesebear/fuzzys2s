from transformers import AutoTokenizer
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
from loaddata import Vocab

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
    dataset = load_dataset("wmt16",'de-en')
    train_raw_iter = iter(dataset['train'])
    valid_raw_iter = iter(dataset['validation'])
    test_raw_iter = iter(dataset['test'])
    raw_iter = itertools.chain(train_raw_iter, valid_raw_iter, test_raw_iter)
    raw_len = dataset['train'].num_rows + dataset['validation'].num_rows + dataset['test'].num_rows
    src_lang = 'en'
    tgt_lang = 'de'
    return raw_iter, raw_len, src_lang, tgt_lang

def get_wmt14_iter():
    dataset_path = options.base_path+"output/wmt14/fr-en/"
    if os.path.exists(dataset_path):
        dataset = load_from_disk(dataset_path)
    else :
        dataset = load_dataset("wmt14",'fr-en')
        dataset.save_to_disk(dataset_path)
    train_raw_iter = iter(dataset['train'])
    valid_raw_iter = iter(dataset['validation'])
    test_raw_iter = iter(dataset['test'])
    raw_iter = itertools.chain(train_raw_iter, valid_raw_iter, test_raw_iter)
    raw_len = 100000 #dataset['train'].num_rows + dataset['validation'].num_rows + dataset['test'].num_rows
    src_lang = 'en'
    tgt_lang = 'fr'
    return raw_iter, raw_len, src_lang, tgt_lang

def get_opus100_iter():
    dataset = load_dataset("opus100",'en-fr')
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

def train_tokenizers():
    logger.add(options.base_path+'output/tokenizers-'+str(datetime.date.today()) +'.log')
    train_tokenizer('wmt14', 1000)
    train_tokenizer('wmt14', 4000)
    train_tokenizer('wmt14', 10000)
    train_tokenizer('wmt16', 1000)
    train_tokenizer('wmt16', 4000)
    train_tokenizer('wmt16', 10000)
    train_tokenizer('opus100', 1000)
    train_tokenizer('opus100', 4000)
    train_tokenizer('opus100', 10000)

def get_tokenizer(dataset, vocab_size):
    tokenizer_path = options.base_path+"output/vs"+str(vocab_size)+"_"+dataset+"_tokenizer"
    if os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        logger.info("dataset %s, vocab size %d , tokenizer found" %(dataset, vocab_size))
        return tokenizer
    else:
        return train_tokenizer(dataset, vocab_size)

# train_tokenizers()

def get_base_tokenizer(name):
    tokenizer_path = options.base_path+"output/"+name+"/"
    if os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(name)
        tokenizer.save_pretrained(tokenizer_path)
    return tokenizer


def gen_token_features(tokens, vocab):
    tok_len = len(tokens)
    features= np.empty([tok_len, 2])
    for i in range(tok_len):
        token = tokens[i]
        size = len(token)
        count = vocab.word2count[token]
        features[i] = [size, count]
    return features

def gen_raw_features(raw_iter, raw_len , src_lang, tgt_lang, vocab, tokenizer):
    tok_total = 0
    raw_tokens = np.empty(raw_len).tolist()
    for i in range(raw_len):
        data = next(raw_iter)
        src = data['translation'][src_lang]
        tgt = data['translation'][tgt_lang]
        line =  src + ' ' + tgt
        tokens = tokenizer.tokenize(line)
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

def fcm_cluster(data, cluster_num, h):
    # input data type is numpy
    logger.info("fcm clustering...")
    feature_num = len(data[0])
    fcm = FCM(n_clusters=cluster_num)
    fcm.fit(data)
    centers = fcm.centers.tolist()
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
    # print("centers:",centers )
    # print("sigma:", sigma)
    sigma = np.array(sigma)
    return centers,sigma

def token_cluster(dataset, rule_num):
    raw_iter, raw_len, src_lang, tgt_lang = get_dataset_iter(dataset)
    vocab = Vocab(dataset)
    tokenizer  = get_base_tokenizer("bert-base-uncased")
    for _ in range(raw_len):
        data = next(raw_iter)
        src = data['translation'][src_lang]
        tgt = data['translation'][tgt_lang]
        raw_line =  src + ' ' + tgt
        tokens = tokenizer.tokenize(raw_line)
        vocab.addTokens(tokens)
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
    return centers, sigma, vocab

# token_cluster('wmt14', options.rule_num)

class FuzzySystem(nn.Module):
    def __init__(self,feature_in, rule_num, center,sigma ):
        super(FuzzySystem, self).__init__()
        self.center = center
        self.sigma = sigma
        self.rule_num = rule_num
        self.feature_num = feature_in
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
            print("sum is 0")
            return products
        products = products/sum
        return products
    def forward(self, x):
        x = x.to(options.device)
        membership = self.fuzzy_layer(x)
        products = self.fire_layer(membership)
        products = self.norm_layer(products)
        output = products
        return output


class FuzzyTokenizer(nn.Module):
    def __init__(self,feature_in, rule_num, dataset):
        super(FuzzyTokenizer, self).__init__()
        center, sigma, vocab = token_cluster(dataset, options.rule_num)
        self.center = center
        self.sigma = sigma
        self.vocab = vocab
        self.rule_num = rule_num
        self.feature_num = feature_in
        self.dataset = dataset
        self.fs = FuzzySystem(feature_in, rule_num, center, sigma)
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
            print("sum is 0")
            return products
        products = products/sum
        return products
    def forward(self, seq):
        tokenizer  = get_base_tokenizer("bert-base-uncased")
        tokens = tokenizer.tokenize(seq)
        for token in tokens:
            size = len(token)
            count = self.vocab.word2count[token]
            x = [size, count]
            membership = self.fuzzy_layer(x)
            products = self.fire_layer(membership)
            products = self.norm_layer(products)
        output = products
        return output


