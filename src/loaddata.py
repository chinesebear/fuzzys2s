import csv
import os
from fcmeans import FCM
import numpy as np
import torch
import json
from torchtext.data import get_tokenizer
from transformers import AutoTokenizer
from loguru import logger
from setting import options,Options
from datasets import load_dataset,load_from_disk
from tokenizers import Tokenizer
from tokenizers.models import BPE,WordPiece,Unigram
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer,WordPieceTrainer, UnigramTrainer
from tqdm import tqdm
import random
import itertools
import pickle
import jsonlines
from translate.storage.tmx import tmxfile

class Vocab:
    def __init__(self, name):
        self.name = name
        self.word2index = {"<sos>":options.SOS, "<eos>":options.EOS, "<pad>":options.PAD,"<unk>":options.UNK}
        self.word2count = {"<sos>":1, "<eos>":1, "<pad>":1,"<unk>":1}
        self.index2word = {options.SOS: "<sos>", options.EOS: "<eos>", options.PAD:"<pad>",options.UNK: "<unk>"}
        self.n_words = 4  # Count PAD , SOS and EOS
        self.feature_max = [] # max value of feature
        self.feature_min = [] # min value of feature
        self.line_max = 0 # max length of sentence

    def addTokens(self, tokens):
        for word in tokens:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def get_base_tokenizer(name):
    tokenizer_path = options.base_path+"output/"+name+"/"
    if os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(name)
        tokenizer.save_pretrained(tokenizer_path)
    return tokenizer.tokenize

def gen_sen_feature_map(vocab, sentence):
    slen = len(sentence)
    HF = 0 # high frequency token count
    for i in range(len(sentence)):
        token_idx = sentence[i]
        word = vocab.index2word[token_idx]
        count = vocab.word2count[word]
        if count > options.high_freq_limit:
            HF = HF + 1
    return [slen, HF]

def combine_sen_feature_map(data, vocab_src, vocab_tgt):
    dlen = len(data)
    src_feature_map = np.empty((dlen, options.feature_num), dtype=int)
    tgt_feature_map = np.empty((dlen, options.feature_num), dtype=int)
    for i in range(dlen):
        src = data[i][0]
        tgt = data[i][1]
        src_feature_map[i] = gen_sen_feature_map(vocab_src, src)
        tgt_feature_map[i] = gen_sen_feature_map(vocab_tgt, tgt)
    # return type is numpy.array
    return src_feature_map, tgt_feature_map

def build_vocab(vocab_src, vocab_tgt, tokens):
    for sentence in tqdm(tokens,'build vocab'):
        src = sentence[0]
        tgt = sentence[1]
        vocab_src.addTokens(src)
        vocab_tgt.addTokens(tgt)
        if len(src) > vocab_src.line_max:
            vocab_src.line_max = len(src)
        if len(tgt) > vocab_tgt.line_max:
            vocab_tgt.line_max = len(tgt)
    return vocab_src, vocab_tgt

def attach_eos(sentence):
    sentence = sentence + [options.EOS]
    return torch.tensor(sentence).to(options.device)

def insert_sos(sentence):
    sentence = [options.SOS] + sentence
    return torch.tensor(sentence).to(options.device)

def gen_token_vectors(vocab_src, vocab_tgt, tokens):
    token_vectors =np.empty([len(tokens),2]).tolist()
    for i in tqdm(range(len(tokens)),'token vector'):
        row = tokens[i]
        src = [vocab_src.word2index[word]  for word in row[0]]
        tgt = [vocab_tgt.word2index[word]  for word in row[1]]
        token_vectors[i][0] = src
        token_vectors[i][1] = tgt
    return token_vectors

def read_raw_tokens(dataset, src_lang, tgt_lang, tokenizer, sen_out=False):
    train_len = options.tok.train_len # dataset['train'].num_rows
    test_len = dataset['test'].num_rows
    valid_len = dataset['validation'].num_rows
    train_data = np.empty([train_len], dtype=int).tolist()
    train_iter = iter(dataset['train'])
    for i in tqdm(range(train_len),'read train data'):
        data = next(train_iter)
        src = data['translation'][src_lang]
        tgt = data['translation'][tgt_lang]
        if sen_out == False:
            src = tokenizer(src)
            tgt = tokenizer(tgt)
        train_data[i] = [src, tgt]
    test_data = np.empty([test_len], dtype=int).tolist()
    test_iter = iter(dataset['test'])
    for i in range(test_len):
        data = next(test_iter)
        src = data['translation'][src_lang]
        tgt = data['translation'][tgt_lang]
        if sen_out == False:
            src = tokenizer(src)
            tgt = tokenizer(tgt)
        test_data[i] = [src, tgt]
    valid_data = np.empty([valid_len], dtype=int).tolist()
    valid_iter = iter(dataset['validation'])
    for i in range(valid_len):
        data = next(valid_iter)
        src = data['translation'][src_lang]
        tgt = data['translation'][tgt_lang]
        if sen_out == False:
            src = tokenizer(src)
            tgt = tokenizer(tgt)
        valid_data[i] = [src, tgt]
    return train_data, valid_data, test_data

def download_dataset():
    datasets = [
        ['wmt14', 'fr-en'],
        ['wmt14', 'de-en'],
        ['wmt16', 'de-en'],
        # ['wmt19', 'de-en'],
        # ['lambada',''],
        ['spider', ''],
        ['htriedman/wikisql', ''],
        ['dvitel/geo', ''],
        ['fathyshalab/atis_intents', ''],
        ['AhmedSSoliman/DJANGO', ''],
        ['neulab/conala', ''],
        ['opus100', 'en-fr'],
        ["opus_euconst",'en-fr'],
        ["cnn_dailymail", '1.0.0'],
        # ["xsum", ''],
        ["samsum", ''],
        ["gem", 'common_gen'],
        ["GEM/xlsum", 'french'],
        ['xsum', ''],
    ]
    for info in datasets:
        name = info[0]
        subpath = info[1]
        dataset_path = options.base_path+"output/"+name+"/"+subpath+"/"
        if os.path.exists(dataset_path):
            dataset = load_from_disk(dataset_path)
        else :
            if subpath == '':
                dataset = load_dataset(name)
            else:
                dataset = load_dataset(name,subpath)
            dataset.save_to_disk(dataset_path)
        print(name,'-',subpath,'done')

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
    logger.info("%s-%s done" %(name,subpath))
    return dataset

def read_line_pair(src_path, tgt_path):
    src_fd = open(src_path, "r")
    tgt_fd = open(tgt_path, "r")
    src_lines = src_fd.readlines()
    tgt_lines = tgt_fd.readlines()
    lines =[]
    for i  in range(len(src_lines)):
        src = src_lines[i]
        tgt = tgt_lines[i]
        lines.append([src, tgt])
    src_fd.close()
    tgt_fd.close()
    return lines

def gen_feature_data(train_tokens, valid_tokens,  test_tokens):
    logger.info("build vocabulary")
    vocab_src = Vocab("src en")
    vocab_tgt = Vocab("tgt fr")
    build_vocab(vocab_src, vocab_tgt, train_tokens)
    build_vocab(vocab_src, vocab_tgt, valid_tokens)
    build_vocab(vocab_src, vocab_tgt, test_tokens)
    logger.info("src vocab name:%s, size:%d" %(vocab_src.name, vocab_src.n_words))
    logger.info("tgt vocab name:%s, size:%d" %(vocab_tgt.name, vocab_tgt.n_words))
    logger.info("generate token vectors")
    train_data = gen_token_vectors(vocab_src, vocab_tgt, train_tokens)
    valid_data = gen_token_vectors(vocab_src, vocab_tgt, valid_tokens)
    test_data = gen_token_vectors(vocab_src, vocab_tgt, test_tokens)
    return train_data, valid_data, test_data, vocab_src, vocab_tgt

def read_tatoeba_data(tokenizer=None, sen_out=False):
    logger.info("read raw data")
    fd = open(options.base_path+"doc/tatoeba/fra-eng/fra.txt",encoding = "utf-8")
    # fd = open(options.base_path+"/doc/tatoeba/deu-eng/deu.txt",encoding = "utf-8")
    lines = fd.readlines()
    logger.info("dataset:tatoeba, total:%d" %(len(lines)))
    random.shuffle(lines)
    logger.info("dataset:tatoeba, data random shffle done")
    data = np.empty((len(lines),2)).tolist() #  src-tgt token pairs or sen pairs
    line_iter = iter(lines)
    for i in tqdm(range(len(lines)), 'read data'):
        line = next(line_iter)
        sen = line.split('\t')
        src = sen[0] # eng eng
        tgt = sen[1] # fra deu
        if sen_out == False:
            src = tokenizer(src)
            tgt = tokenizer(tgt)
        data[i][0] = src
        data[i][1] = tgt
    fd.close()
    total = len(data)
    part = 2000
    train_data = data[:total - part*2]
    valid_data = data[total - part*2:total - part]
    test_data = data[total - part:]
    logger.info("dataset: tatoeba, train: %d, valid: %d, test: %d" %(len(train_data),len(valid_data), len(test_data)))
    return train_data, valid_data, test_data

def read_wmt14_data(tokenizer=None, sen_out=False):
    logger.info("read wmt14 data")
    src_lang = 'en'
    tgt_lang = 'fr'
    dataset = read_dataset('wmt14', 'fr-en')
    logger.info("read raw tokens")
    train_data, valid_data,  test_data = read_raw_tokens(dataset, src_lang, tgt_lang, tokenizer, sen_out)
    logger.info("dataset: wmt14, train: %d, valid: %d, test: %d" %(len(train_data),len(valid_data), len(test_data)))
    return train_data, valid_data,  test_data

def read_wmt16_data(tokenizer=None, sen_out=False):
    logger.info("read wmt16 data")
    src_lang = 'en'
    tgt_lang = 'de'
    dataset = read_dataset('wmt16', 'de-en')
    logger.info("read raw tokens")
    train_data, valid_data,  test_data = read_raw_tokens(dataset, src_lang, tgt_lang, tokenizer, sen_out)
    logger.info("dataset: wmt16, train: %d, valid: %d, test: %d" %(len(train_data),len(valid_data), len(test_data)))
    return train_data, valid_data,  test_data

def read_opus100_data(tokenizer=None, sen_out=False):
    logger.info("read opus100")
    src_lang = 'en'
    tgt_lang = 'fr'
    dataset = read_dataset('opus100', 'en-fr')
    logger.info("read raw data")
    train_data, valid_data,  test_data = read_raw_tokens(dataset, src_lang, tgt_lang,tokenizer,sen_out)
    logger.info("dataset: opus100, train: %d, valid: %d, test: %d" %(len(train_data),len(valid_data), len(test_data)))
    test_data = test_data[:-2]
    return train_data, valid_data,  test_data

def read_ubuntu_data(tokenizer=None, sen_out=False):
    logger.info("read ubuntu data")
    src_lang = 'as'
    tgt_lang = 'bs'
    with open(options.base_path+"doc/as-bs.tmx", 'rb') as fin:
        tmx_file = tmxfile(fin, src_lang,tgt_lang)
    data = [] #  src-tgt token pairs or sen pairs
    tmx_iter = tmx_file.unit_iter()
    for node in tqdm(tmx_iter, 'read data'):
        src = node.source
        tgt = node.target
        if sen_out == False:
            src = tokenizer(src)
            tgt = tokenizer(tgt)
        data.append([src, tgt])
    total = len(data)
    part = int(total/10)
    train_data = data[:total - part*2]
    valid_data = data[total - part*2:total - part]
    test_data = data[total - part:]
    logger.info("dataset: ubuntu, train: %d, valid: %d, test: %d" %(len(train_data),len(valid_data), len(test_data)))
    return train_data, valid_data,  test_data


def read_hearthstone_data(tokenizer=None, sen_out=False):
    logger.info("read raw data")
    train_lines = read_line_pair(options.base_path+'doc/hearthstone/train_hs.in', options.base_path+'doc/hearthstone/train_hs.out')
    valid_lines = read_line_pair(options.base_path+'doc/hearthstone/dev_hs.in', options.base_path+'doc/hearthstone/dev_hs.out')
    test_lines = read_line_pair(options.base_path+'doc/hearthstone/test_hs.in', options.base_path+'doc/hearthstone/test_hs.out')
    if sen_out:
        logger.info("dataset: hearthstone, train: %d, valid: %d, test: %d" %(len(train_lines),len(valid_lines), len(test_lines)))
        return train_lines, valid_lines, test_lines
    train_data=[]
    valid_data=[]
    test_data=[]
    for src, tgt in tqdm(train_lines):
        src_tokens = tokenizer(src)
        tgt_tokens = tokenizer(tgt)
        train_data.append([src_tokens, tgt_tokens])
    for src, tgt in tqdm(valid_lines):
        src_tokens = tokenizer(src)
        tgt_tokens = tokenizer(tgt)
        valid_data.append([src_tokens, tgt_tokens])
    for src, tgt in tqdm(test_lines):
        src_tokens = tokenizer(src)
        tgt_tokens = tokenizer(tgt)
        test_data.append([src_tokens, tgt_tokens])
    logger.info("dataset: hearthstone, train: %d, valid: %d, test: %d" %(len(train_data),len(valid_data), len(test_data)))
    return train_data, valid_data,  test_data

def read_magic_data(tokenizer=None, sen_out=False):
    logger.info("read raw data")
    train_lines = read_line_pair(options.base_path+'doc/magic/train_magic.in', options.base_path+'doc/magic/train_magic.out')
    valid_lines = read_line_pair(options.base_path+'doc/magic/dev_magic.in', options.base_path+'doc/magic/dev_magic.out')
    test_lines = read_line_pair(options.base_path+'doc/magic/test_magic.in', options.base_path+'doc/magic/test_magic.out')
    if sen_out:
        logger.info("dataset: magic the gathering, train: %d, valid: %d, test: %d" %(len(train_lines),len(valid_lines), len(test_lines)))
        return train_lines, valid_lines, test_lines
    train_data=[]
    valid_data=[]
    test_data=[]
    for src, tgt in train_lines:
        src_tokens = tokenizer(src)
        tgt_tokens = tokenizer(tgt)
        train_data.append([src_tokens, tgt_tokens])
    for src, tgt in valid_lines:
        src_tokens = tokenizer(src)
        tgt_tokens = tokenizer(tgt)
        valid_data.append([src_tokens, tgt_tokens])
    for src, tgt in test_lines:
        src_tokens = tokenizer(src)
        tgt_tokens = tokenizer(tgt)
        test_data.append([src_tokens, tgt_tokens])
    logger.info("dataset: magic, train: %d, valid: %d, test: %d" %(len(train_data),len(valid_data), len(test_data)))
    return train_data, valid_data,  test_data

def read_spider_data(tokenizer=None, sen_out=False):
    logger.info("read spider data")
    dataset = read_dataset('spider', '')
    logger.info("read raw tokens")
    train_len = dataset['train'].num_rows
    valid_len = dataset['validation'].num_rows
    train_data = np.empty([train_len], dtype=int).tolist()
    train_iter = iter(dataset['train'])
    for i in range(train_len):
        data = next(train_iter)
        if sen_out:
            src = data['question']
            tgt = data['query']
        else:
            src = data['question_toks']
            tgt = data['query_toks']
        train_data[i] = [src, tgt]
    valid_data = np.empty([valid_len], dtype=int).tolist()
    valid_iter = iter(dataset['validation'])
    for i in range(valid_len):
        data = next(valid_iter)
        if sen_out:
            src = data['question']
            tgt = data['query']
        else:
            src = data['question_toks']
            tgt = data['query_toks']
        valid_data[i] = [src, tgt]
    random.shuffle(valid_data)
    test_data= valid_data[int(valid_len/2):]
    valid_data = valid_data[:int(valid_len/2)]
    logger.info("dataset: spider, train: %d, valid: %d, test: %d" %(len(train_data),len(valid_data), len(test_data)))
    return train_data, valid_data,  test_data

def read_geo_data(tokenizer=None, sen_out=False):
    logger.info("read geo data")
    dataset = read_dataset('dvitel/geo', '')
    logger.info("read raw tokens")
    train_len = dataset['train'].num_rows
    train_data = np.empty([train_len], dtype=int).tolist()
    train_iter = iter(dataset['train'])
    for i in range(train_len):
        data = next(train_iter)
        src = data['source']
        tgt = data['target']
        if sen_out == False:
            src = tokenizer(src)
            tgt = tokenizer(tgt)
        train_data[i] = [src, tgt]
    random.shuffle(train_data)
    part = int(train_len/10)
    test_data= train_data[train_len-part:]
    valid_data = train_data[train_len-part:]
    train_data= train_data[:train_len-part]
    logger.info("dataset: geo, train: %d, valid: %d, test: %d" %(len(train_data),len(valid_data), len(test_data)))
    return train_data, valid_data,  test_data

def read_django_data(tokenizer=None, sen_out=False):
    logger.info("read django data")
    dataset = read_dataset('AhmedSSoliman/DJANGO', '')
    logger.info("read raw tokens")
    train_len = dataset['train'].num_rows
    valid_len = dataset['validation'].num_rows
    test_len = dataset['test'].num_rows
    train_data = np.empty([train_len], dtype=int).tolist()
    train_iter = iter(dataset['train'])
    for i in range(train_len):
        data = next(train_iter)
        src = data['nl']
        tgt = data['code']
        if sen_out == False:
            src = tokenizer(src)
            tgt = tokenizer(tgt)
        train_data[i] = [src, tgt]
    valid_data = np.empty([valid_len], dtype=int).tolist()
    valid_iter = iter(dataset['validation'])
    for i in range(valid_len):
        data = next(valid_iter)
        src = data['nl']
        tgt = data['code']
        if sen_out == False:
            src = tokenizer(src)
            tgt = tokenizer(tgt)
        valid_data[i] = [src, tgt]
    test_data = np.empty([test_len], dtype=int).tolist()
    test_iter = iter(dataset['test'])
    for i in range(test_len):
        data = next(test_iter)
        src = data['nl']
        tgt = data['code']
        if sen_out == False:
            src = tokenizer(src)
            tgt = tokenizer(tgt)
        test_data[i] = [src, tgt]
    logger.info("dataset: django, train: %d, valid: %d, test: %d" %(len(train_data),len(valid_data), len(test_data)))
    return train_data, valid_data,  test_data

def read_conala_data(tokenizer=None, sen_out=False):
    logger.info("read conala data")
    dataset = read_dataset('neulab/conala', '')
    logger.info("read raw tokens")
    train_len = dataset['train'].num_rows
    test_len = dataset['test'].num_rows
    train_data = np.empty([train_len], dtype=int).tolist()
    train_iter = iter(dataset['train'])
    for i in range(train_len):
        data = next(train_iter)
        if data['rewritten_intent'] == None:
            src = data['intent']
        else:
            src = data['rewritten_intent']
        src = src.replace('\\', '#').replace('/', '#')
        tgt = data['snippet'].replace('\\', '#').replace('/', '#')
        if sen_out == False:
            src = tokenizer(src)
            tgt = tokenizer(tgt)
        train_data[i] = [src, tgt]
    test_data = np.empty([test_len], dtype=int).tolist()
    test_iter = iter(dataset['test'])
    for i in range(test_len):
        data = next(test_iter)
        if data['rewritten_intent'] == None:
            src = data['intent']
        else:
            src = data['rewritten_intent']
        src = src.replace('\\', '#').replace('/', '#')
        tgt = data['snippet'].replace('\\', '#').replace('/', '#')
        if sen_out == False:
            src = tokenizer(src)
            tgt = tokenizer(tgt)
        test_data[i] = [src, tgt]
    random.shuffle(test_data)
    logger.info("test data random shuffle done")
    part = int(test_len/2)
    valid_data = test_data[:part]
    test_data =  test_data[part:]
    logger.info("dataset: conala, train: %d, valid: %d, test: %d" %(len(train_data),len(valid_data), len(test_data)))
    return train_data, valid_data,  test_data

def read_opus_euconst_data(tokenizer=None, sen_out=False):
    logger.info("read opus_euconst data")
    dataset = read_dataset('opus_euconst', 'en-fr')
    logger.info("read raw tokens")
    src_lang = 'en'
    tgt_lang = 'fr'
    train_len = dataset['train'].num_rows
    logger.info("dataset:opus_euconst, total: %d"  %(train_len))
    train_raw_data = dataset['train']
    train_iter = iter(train_raw_data)
    train_data = np.empty([train_len], dtype=int).tolist()
    for i in tqdm(range(train_len)):
        data = next(train_iter)
        src = data['translation'][src_lang]
        tgt = data['translation'][tgt_lang]
        if sen_out == False:
            src = tokenizer(src)
            tgt = tokenizer(tgt)
        train_data[i] = [src, tgt]
    random.shuffle(train_data)
    logger.info("dataset:opus_euconst, data random shffle done")
    part = int(train_len/10)
    valid_data = train_data[train_len -2 *part: train_len - part]
    test_data = train_data[train_len -part:]
    train_data = train_data[:train_len -2*part]
    logger.info("dataset: opus_euconst, train: %d, valid: %d, test: %d" %(len(train_data),len(valid_data), len(test_data)))
    return train_data, valid_data,  test_data

def read_cnn_dailymail_data(tokenizer=None, sen_out=False):
    logger.info("read cnn_dailymail data")
    dataset = read_dataset('cnn_dailymail', '1.0.0')
    logger.info("read raw tokens")
    train_len = dataset['train'].num_rows
    valid_len = dataset['validation'].num_rows
    test_len = dataset['test'].num_rows
    train_data = np.empty([train_len], dtype=int).tolist()
    train_iter = iter(dataset['train'])
    for i in tqdm(range(train_len), 'read train data'):
        data = next(train_iter)
        src = data['article']
        tgt = data['highlights']
        if sen_out == False:
            src = tokenizer(src)
            tgt = tokenizer(tgt)
        train_data[i] = [src, tgt]
    valid_data = np.empty([valid_len], dtype=int).tolist()
    valid_iter = iter(dataset['validation'])
    for i in tqdm(range(valid_len), 'read valid data'):
        data = next(valid_iter)
        src = data['article']
        tgt = data['highlights']
        if sen_out == False:
            src = tokenizer(src)
            tgt = tokenizer(tgt)
        valid_data[i] = [src, tgt]
    test_data = np.empty([test_len], dtype=int).tolist()
    test_iter = iter(dataset['test'])
    for i in tqdm(range(test_len), 'read test data'):
        data = next(test_iter)
        src = data['article']
        tgt = data['highlights']
        if sen_out == False:
            src = tokenizer(src)
            tgt = tokenizer(tgt)
        test_data[i] = [src, tgt]
    logger.info("dataset: cnn dailymail, train: %d, valid: %d, test: %d" %(len(train_data),len(valid_data), len(test_data)))
    return train_data, valid_data,  test_data

def read_samsum_data(tokenizer=None, sen_out=False):
    logger.info("read samsum data")
    dataset = read_dataset('samsum', '')
    logger.info("read raw tokens")
    train_len = dataset['train'].num_rows
    valid_len = dataset['validation'].num_rows
    test_len = dataset['test'].num_rows
    train_data = np.empty([train_len], dtype=int).tolist()
    train_iter = iter(dataset['train'])
    for i in tqdm(range(train_len), 'read train data'):
        data = next(train_iter)
        src = data['dialogue']
        tgt = data['summary']
        if sen_out == False:
            src = tokenizer(src)
            tgt = tokenizer(tgt)
        train_data[i] = [src, tgt]
    valid_data = np.empty([valid_len], dtype=int).tolist()
    valid_iter = iter(dataset['validation'])
    for i in tqdm(range(valid_len), 'read valid data'):
        data = next(valid_iter)
        src = data['dialogue']
        tgt = data['summary']
        if sen_out == False:
            src = tokenizer(src)
            tgt = tokenizer(tgt)
        valid_data[i] = [src, tgt]
    test_data = np.empty([test_len], dtype=int).tolist()
    test_iter = iter(dataset['test'])
    for i in tqdm(range(test_len), 'read test data'):
        data = next(test_iter)
        src = data['dialogue']
        tgt = data['summary']
        if sen_out == False:
            src = tokenizer(src)
            tgt = tokenizer(tgt)
        test_data[i] = [src, tgt]
    logger.info("dataset: samsum, train: %d, valid: %d, test: %d" %(len(train_data),len(valid_data), len(test_data)))
    return train_data, valid_data,  test_data

def read_xsum_data(tokenizer=None, sen_out=False):
    logger.info("read xsum data")
    dataset = read_dataset('xsum', '')
    logger.info("read raw tokens")
    train_len = dataset['train'].num_rows
    valid_len = dataset['validation'].num_rows
    test_len = dataset['test'].num_rows
    train_data = np.empty([train_len], dtype=int).tolist()
    train_iter = iter(dataset['train'])
    for i in tqdm(range(train_len), 'read train data'):
        data = next(train_iter)
        src = data['document']
        tgt = data['summary']
        if sen_out == False:
            src = tokenizer(src)
            tgt = tokenizer(tgt)
        train_data[i] = [src, tgt]
    valid_data = np.empty([valid_len], dtype=int).tolist()
    valid_iter = iter(dataset['validation'])
    for i in tqdm(range(valid_len), 'read valid data'):
        data = next(valid_iter)
        src = data['document']
        tgt = data['summary']
        if sen_out == False:
            src = tokenizer(src)
            tgt = tokenizer(tgt)
        valid_data[i] = [src, tgt]
    test_data = np.empty([test_len], dtype=int).tolist()
    test_iter = iter(dataset['test'])
    for i in tqdm(range(test_len), 'read test data'):
        data = next(test_iter)
        src = data['document']
        tgt = data['summary']
        if sen_out == False:
            src = tokenizer(src)
            tgt = tokenizer(tgt)
        test_data[i] = [src, tgt]
    logger.info("dataset: xsum, train: %d, valid: %d, test: %d" %(len(train_data),len(valid_data), len(test_data)))
    return train_data, valid_data,  test_data

def read_gem_data(tokenizer=None, sen_out=False):
    logger.info("read gem data")
    dataset = read_dataset('gem', 'common_gen')
    logger.info("read raw tokens")
    train_len = dataset['train'].num_rows
    valid_len = dataset['validation'].num_rows
    test_len = dataset['test'].num_rows
    train_data = np.empty([train_len], dtype=int).tolist()
    train_iter = iter(dataset['train'])
    for i in tqdm(range(train_len), 'read train data'):
        data = next(train_iter)
        src = data['concepts']
        tgt = data['target']
        if sen_out ==False:
            tgt = tokenizer(tgt)
        train_data[i] = [src, tgt]
    valid_data = np.empty([valid_len], dtype=int).tolist()
    valid_iter = iter(dataset['validation'])
    for i in tqdm(range(valid_len), 'read valid data'):
        data = next(valid_iter)
        src = data['concepts']
        tgt = data['target']
        if sen_out ==False:
            tgt = tokenizer(tgt)
        valid_data[i] = [src, tgt]
    test_data = np.empty([test_len], dtype=int).tolist()
    test_iter = iter(dataset['test'])
    for i in tqdm(range(test_len), 'read test data'):
        data = next(test_iter)
        src = data['concepts']
        tgt = data['target']
        if sen_out ==False:
            tgt = tokenizer(tgt)
        test_data[i] = [src, tgt]
    logger.info("dataset: gem, train: %d, valid: %d, test: %d" %(len(train_data),len(valid_data), len(test_data)))
    return train_data, valid_data,  test_data

def read_billsum_data(tokenizer=None, sen_out=False):
    logger.info("read billsum data")
    logger.info("read raw tokens")
    train_path = options.base_path + 'output/billsum_v4_1/us_train_data_final_OFFICIAL.jsonl'
    test_path =  options.base_path + 'output/billsum_v4_1/us_test_data_final_OFFICIAL.jsonl'
    train_data = []
    test_data = []
    with open(train_path, "r", encoding="utf8") as f:
        for line in tqdm(jsonlines.Reader(f)):
            src = line['text']
            tgt = line['summary']
            if sen_out==False:
                src = tokenizer(src)
                tgt = tokenizer(tgt)
            train_data.append([src, tgt])
        f.close()
    with open(test_path, "r", encoding="utf8") as f:
        for line in tqdm(jsonlines.Reader(f)):
            src = line['text']
            tgt = line['summary']
            if sen_out == False:
                src = tokenizer(src)
                tgt = tokenizer(tgt)
            test_data.append([src, tgt])
        f.close()
    random.shuffle(test_data)
    part = int(len(test_data)/2)
    valid_data = test_data[:part]
    test_data = test_data[part:]
    logger.info("dataset: billsum, train: %d, valid: %d, test: %d" %(len(train_data),len(valid_data), len(test_data)))
    return train_data, valid_data,  test_data

def read_orangesum_data(tokenizer=None, sen_out=False):
    dataset = read_dataset('orange_sum', 'abstract')
    logger.info("read raw tokens")
    train_len = dataset['train'].num_rows
    valid_len = dataset['validation'].num_rows
    test_len = dataset['test'].num_rows
    train_data = np.empty([train_len], dtype=int).tolist()
    train_iter = iter(dataset['train'])
    for i in tqdm(range(train_len), 'read train data'):
        data = next(train_iter)
        src = data['text']
        tgt = data['summary']
        if sen_out ==False:
            src = tokenizer(src)
            tgt = tokenizer(tgt)
        train_data[i] = [src, tgt]
    valid_data = np.empty([valid_len], dtype=int).tolist()
    valid_iter = iter(dataset['validation'])
    for i in tqdm(range(valid_len), 'read valid data'):
        data = next(valid_iter)
        src = data['text']
        tgt = data['summary']
        if sen_out ==False:
            src = tokenizer(src)
            tgt = tokenizer(tgt)
        valid_data[i] = [src, tgt]
    test_data = np.empty([test_len], dtype=int).tolist()
    test_iter = iter(dataset['test'])
    for i in tqdm(range(test_len), 'read test data'):
        data = next(test_iter)
        src = data['text']
        tgt = data['summary']
        if sen_out ==False:
            src = tokenizer(src)
            tgt = tokenizer(tgt)
        test_data[i] = [src, tgt]
    logger.info("dataset: orangesum, train: %d, valid: %d, test: %d" %(len(train_data),len(valid_data), len(test_data)))
    return train_data, valid_data,  test_data


def read_xlsum_data(tokenizer=None, sen_out=False):
    logger.info("read GEM/xlsum data")
    dataset = read_dataset('GEM/xlsum', 'french')
    logger.info("read raw tokens")
    train_len = dataset['train'].num_rows
    valid_len = dataset['validation'].num_rows
    test_len = dataset['test'].num_rows
    train_data = np.empty([train_len], dtype=int).tolist()
    train_iter = iter(dataset['train'])
    for i in tqdm(range(train_len), 'read train data'):
        data = next(train_iter)
        src = data['text']
        tgt = data['target']
        if sen_out == False:
            src = tokenizer(src)
            tgt = tokenizer(tgt)
        train_data[i] = [src, tgt]
    valid_data = np.empty([valid_len], dtype=int).tolist()
    valid_iter = iter(dataset['validation'])
    for i in tqdm(range(valid_len), 'read valid data'):
        data = next(valid_iter)
        src = data['text']
        tgt = data['target']
        if sen_out == False:
            src = tokenizer(src)
            tgt = tokenizer(tgt)
        valid_data[i] = [src, tgt]
    test_data = np.empty([test_len], dtype=int).tolist()
    test_iter = iter(dataset['test'])
    for i in tqdm(range(test_len), 'read test data'):
        data = next(test_iter)
        src = data['text']
        tgt = data['target']
        if sen_out == False:
            src = tokenizer(src)
            tgt = tokenizer(tgt)
        test_data[i] = [src, tgt]
    logger.info("dataset: xlsum, train: %d, valid: %d, test: %d" %(len(train_data),len(valid_data), len(test_data)))
    return train_data, valid_data,  test_data

def read_atis_data(tokenizer=None, sen_out=False):
    logger.info("read atis data")
    dataset = read_dataset('fathyshalab/atis_intents', '')
    logger.info("read raw tokens")
    train_len = dataset['train'].num_rows
    test_len = dataset['test'].num_rows
    train_data = np.empty([train_len], dtype=int).tolist()
    train_iter = iter(dataset['train'])
    for i in tqdm(range(train_len), 'read train data'):
        data = next(train_iter)
        src = data['text']
        tgt = data['label text']
        if sen_out == False:
            src = tokenizer(src)
            tgt = [tgt]
        train_data[i] = [src, tgt]
    test_data = np.empty([test_len], dtype=int).tolist()
    test_iter = iter(dataset['test'])
    for i in tqdm(range(test_len), 'read test data'):
        data = next(test_iter)
        src = data['text']
        tgt = data['label text']
        if sen_out == False:
            src = tokenizer(src)
            tgt = [tgt]
        test_data[i] = [src, tgt]
    part = int(test_len/2)
    valid_data = test_data[:part]
    test_data = test_data[part:]
    logger.info("dataset: atis, train: %d, valid: %d, test: %d" %(len(train_data),len(valid_data), len(test_data)))
    return train_data, valid_data,  test_data



def read_copy_data():
    logger.info("read raw data")
    logger.info("build vocabulary")
    vocab_src = Vocab("src")
    vocab_tgt = Vocab("tgt")
    words = ["0","1","2","3","4", "5","6","7","8","9","10","11","12","13","14","15","16","17"]
    data = torch.randint(4,15, (1100,2,10)).tolist()
    for i in range(len(data)):
        for j in range(len(data[i][0])):
            data[i][0][j] = words[data[i][0][j]]
        data[i][1] = data[i][0]
    build_vocab(vocab_src, vocab_tgt, data)
    logger.info("src vocab name:%s, size:%d" %(vocab_src.name, vocab_src.n_words))
    logger.info("tgt vocab name:%s, size:%d" %(vocab_tgt.name, vocab_tgt.n_words))
    logger.info("generate token vectors")
    data = gen_token_vectors(vocab_src, vocab_tgt, data)
    train_data =data[:1000]
    valid_data =data[1000:1050]
    test_data =data[1050:]
    logger.info("dataset: copy, train: %d, valid: %d, test: %d" %(len(train_data),len(valid_data), len(test_data)))
    return train_data, valid_data, test_data, vocab_src, vocab_tgt

def read_raw_data(dataset, tokenizer=None, sen_out=False):
    if dataset =="wmt14":
        train_data, valid_data,  test_data = read_wmt14_data(tokenizer,sen_out=sen_out)
    elif dataset == "wmt16":
        train_data, valid_data,  test_data = read_wmt16_data(tokenizer,sen_out=sen_out)
    elif dataset == "tatoeba":
        train_data, valid_data,  test_data = read_tatoeba_data(tokenizer, sen_out=sen_out)
    elif dataset == 'opus100':
       train_data, valid_data,  test_data =  read_opus100_data(tokenizer,sen_out=sen_out)
    elif dataset == 'hearthstone':
        train_data, valid_data,  test_data =  read_hearthstone_data(tokenizer, sen_out=sen_out)
    elif dataset == 'magic':
        train_data, valid_data,  test_data =  read_magic_data(tokenizer, sen_out=sen_out)
    elif dataset == "spider":
        train_data, valid_data,  test_data =  read_spider_data(tokenizer, sen_out=sen_out)
    elif dataset == "geo":
        train_data, valid_data,  test_data =  read_geo_data(tokenizer, sen_out=sen_out)
    elif dataset == 'django':
        train_data, valid_data,  test_data =  read_django_data(tokenizer,sen_out=sen_out)
    elif dataset == 'conala':
        train_data, valid_data,  test_data =  read_conala_data(tokenizer, sen_out=sen_out)
    elif dataset == 'opus_euconst':
        train_data, valid_data,  test_data =  read_opus_euconst_data(tokenizer, sen_out=sen_out)
    elif dataset == 'cnn_dailymail':
        train_data, valid_data,  test_data =  read_cnn_dailymail_data(tokenizer, sen_out=sen_out)
    elif dataset == 'samsum':
        train_data, valid_data,  test_data =  read_samsum_data(tokenizer, sen_out=sen_out)
    elif dataset == 'xsum':
        train_data, valid_data,  test_data =  read_xsum_data(tokenizer, sen_out=sen_out)
    elif dataset == 'gem':
        train_data, valid_data,  test_data =  read_gem_data(tokenizer, sen_out=sen_out)
    elif dataset == 'xlsum':
        train_data, valid_data,  test_data =  read_xlsum_data(tokenizer, sen_out=sen_out)
    elif dataset == 'atis':
        train_data, valid_data,  test_data =  read_atis_data(tokenizer, sen_out=sen_out)
    elif dataset == 'billsum':
        train_data, valid_data,  test_data =  read_billsum_data(tokenizer, sen_out=sen_out)
    elif dataset == 'orangesum':
        train_data, valid_data,  test_data =  read_orangesum_data(tokenizer, sen_out=sen_out)
    elif dataset == 'ubuntu':
        train_data, valid_data,  test_data =  read_ubuntu_data(tokenizer, sen_out=sen_out)
    return train_data, valid_data,  test_data

def read_data(dataset, tokenizer=None, sen_out=False):
    if dataset == "copy":
        return  read_copy_data()
    elif sen_out:
        return read_raw_data(dataset, sen_out=True)
    else:
        train_tokens, valid_tokens,  test_tokens = read_raw_data(dataset, tokenizer)
        return gen_feature_data(train_tokens, valid_tokens,  test_tokens)
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

def save_vocab(vocab, name):
    logger.info("save %s vocab..." %(name))
    vocab_path = options.base_path+'output/vocab_'+name+'.pickle'
    with open(vocab_path, 'wb') as f: # open file with write-mode
        pickle.dump(vocab, f) # serialize and save object

def load_vocab(name):
    logger.info("load %s vocab..." %(name))
    vocab_path = options.base_path+'output/vocab_'+name+'.pickle'
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)   # read file and build object
    return vocab

def save_center_info(name, centers, sigma):
    logger.info("save %s centers..." %(name))
    info_path = options.base_path+'output/center_'+name+'.csv'
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

def load_cluster_info(name):
    logger.info("load %s centers..." %(name))
    info_path = options.base_path+'output/center_'+name+'.csv'
    centers = []
    sigma = []
    with open(info_path, encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        print(headers)
        for row in reader:
            centers.append([row[0], row[1]])
            sigma.append([row[2], row[3]])
    return centers, sigma
def translate_vocab_task():
    src_lang = 'en'
    tgt_lang = 'fr'
    tokenizer = get_tokenizer("basic_english")
    vocab_src = Vocab('src '+src_lang)
    vocab_tgt = Vocab('src '+tgt_lang)
    raw_iter, raw_len, src_lang, tgt_lang = get_wmt14_iter()
    for i in tqdm(range(raw_len),'wmt14'):
        data = next(raw_iter)
        src = data['translation'][src_lang]
        tgt = data['translation'][tgt_lang]
        src = tokenizer(src)
        tgt = tokenizer(tgt)
        vocab_src.addTokens(src)
        vocab_tgt.addTokens(tgt)
    raw_iter, raw_len, src_lang, tgt_lang = get_opus100_iter()
    for i in tqdm(range(raw_len),'opus100'):
        data = next(raw_iter)
        src = data['translation'][src_lang]
        tgt = data['translation'][tgt_lang]
        src = tokenizer(src)
        tgt = tokenizer(tgt)
        vocab_src.addTokens(src)
        vocab_tgt.addTokens(tgt)
    train_data, valid_data,  test_data = read_opus_euconst_data(tokenizer=tokenizer)
    raw_data = train_data + valid_data + test_data
    for data in tqdm(raw_data, 'opus_euconst'):
        src = data[0]
        tgt = data[1]
        vocab_src.addTokens(src)
        vocab_tgt.addTokens(tgt)
    train_data, valid_data,  test_data = read_tatoeba_data(tokenizer=tokenizer)
    raw_data = train_data + valid_data + test_data
    for data in tqdm(raw_data, 'tatoeba'):
        src = data[0]
        tgt = data[1]
        vocab_src.addTokens(src)
        vocab_tgt.addTokens(tgt)
    save_vocab(vocab_src, "translate_src_"+src_lang)
    save_vocab(vocab_tgt, "translate_tgt_"+tgt_lang)
    return 0

def translate_center_task():
    src_lang = 'en'
    tgt_lang = 'fr'
    step = 10000
    tokenizer = get_tokenizer("basic_english")
    vocab_src = load_vocab("translate_src_"+src_lang)
    vocab_tgt = load_vocab("translate_tgt_"+tgt_lang)
    raw_iter, raw_len, src_lang, tgt_lang = get_wmt14_iter()
    src_centers = []
    tgt_centers = []
    src_features = []
    tgt_features = []
    count=0
    for i in tqdm(range(raw_len),'wmt14'):
        data = next(raw_iter)
        src = data['translation'][src_lang]
        tgt = data['translation'][tgt_lang]
        src = tokenizer(src)
        tgt = tokenizer(tgt)
        src = [vocab_src.word2index[word]  for word in src]
        tgt = [vocab_tgt.word2index[word]  for word in tgt]
        src = gen_sen_feature_map(vocab_src, src)
        tgt = gen_sen_feature_map(vocab_tgt, tgt)
        src_features.append(src)
        tgt_features.append(tgt)
        count = count + 1
        if count % step == 0:
            src_data = np.array(src_features)
            centers_tensor,_ = fcm(src_data, options.cluster_num, options.h)
            src_centers.extend(centers_tensor.tolist())
            tgt_data = np.array(tgt_features)
            centers_tensor,_ = fcm(tgt_data, options.cluster_num, options.h)
            tgt_centers.extend(centers_tensor.tolist())
            src_features = []
            tgt_features = []
    count=0
    raw_iter, raw_len, src_lang, tgt_lang = get_opus100_iter()
    for i in tqdm(range(raw_len),'opus100'):
        data = next(raw_iter)
        src = data['translation'][src_lang]
        tgt = data['translation'][tgt_lang]
        src = tokenizer(src)
        tgt = tokenizer(tgt)
        src = [vocab_src.word2index[word]  for word in src]
        tgt = [vocab_tgt.word2index[word]  for word in tgt]
        src = gen_sen_feature_map(vocab_src, src)
        tgt = gen_sen_feature_map(vocab_tgt, tgt)
        src_features.append(src)
        tgt_features.append(tgt)
        count = count + 1
        if count % step == 0:
            src_data = np.array(src_features)
            centers_tensor,_ = fcm(src_data, options.cluster_num, options.h)
            src_centers.extend(centers_tensor.tolist())
            tgt_data = np.array(tgt_features)
            centers_tensor,_ = fcm(tgt_data, options.cluster_num, options.h)
            tgt_centers.extend(centers_tensor.tolist())
            src_features = []
            tgt_features = []
    count=0
    train_data, valid_data,  test_data = read_opus_euconst_data(tokenizer=tokenizer)
    raw_data = train_data + valid_data + test_data
    for data in tqdm(raw_data, 'opus_euconst'):
        src = data[0]
        tgt = data[1]
        src = [vocab_src.word2index[word]  for word in src]
        tgt = [vocab_tgt.word2index[word]  for word in tgt]
        src = gen_sen_feature_map(vocab_src, src)
        tgt = gen_sen_feature_map(vocab_tgt, tgt)
        src_features.append(src)
        tgt_features.append(tgt)
        count = count + 1
        if count % 1000 == 0:
            src_data = np.array(src_features)
            centers_tensor,_ = fcm(src_data, options.cluster_num, options.h)
            src_centers.extend(centers_tensor.tolist())
            tgt_data = np.array(tgt_features)
            centers_tensor,_ = fcm(tgt_data, options.cluster_num, options.h)
            tgt_centers.extend(centers_tensor.tolist())
            src_features = []
            tgt_features = []
    count=0
    train_data, valid_data,  test_data = read_tatoeba_data(tokenizer=tokenizer)
    raw_data = train_data + valid_data + test_data
    for data in tqdm(raw_data, 'tatoeba'):
        src = data[0]
        tgt = data[1]
        src = [vocab_src.word2index[word]  for word in src]
        tgt = [vocab_tgt.word2index[word]  for word in tgt]
        src = gen_sen_feature_map(vocab_src, src)
        tgt = gen_sen_feature_map(vocab_tgt, tgt)
        src_features.append(src)
        tgt_features.append(tgt)
        count = count + 1
        if count % step:
            src_data = np.array(src_features)
            centers_tensor,_ = fcm(src_data, options.cluster_num, options.h)
            src_centers.extend(centers_tensor.tolist())
            tgt_data = np.array(tgt_features)
            centers_tensor,_ = fcm(tgt_data, options.cluster_num, options.h)
            tgt_centers.extend(centers_tensor.tolist())
            src_features = []
            tgt_features = []
    centers_tensor,sigma_tensor = fcm(np.array(src_centers), options.cluster_num, options.h)
    save_center_info("translate_src_"+src_lang, centers_tensor.tolist(), sigma_tensor.tolist())
    centers_tensor,sigma_tensor = fcm(np.array(tgt_centers), options.cluster_num, options.h)
    save_center_info("translate_tgt_"+tgt_lang, centers_tensor.tolist(), sigma_tensor.tolist())
    return 0

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

def fcm_part(data, cluster_num,h):
    feature_num = len(data[0])
    fcm = FCM(n_clusters=cluster_num)
    fcm.fit(data)
    centers = fcm.centers
    centers = centers.tolist()
    # logger.info("cluster center: %d" %(len(centers)))
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
    return centers, sigma

def fcm(data, cluster_num, h):
    # input data type is numpy
    total = len(data)
    if total > 2000:
        part_num = 1000
        part = int(total / part_num)
        centers = []
        for i in tqdm(range(part_num),"fcm"):
            data_p = data[i*part: (i+1)*part]
            centers_p, _ = fcm_part(data_p, cluster_num, h)
            centers.extend(centers_p)
        centers = np.array(centers)
        centers,sigma = fcm_part(centers, cluster_num,h)
    else:
        centers,sigma = fcm_part(data, cluster_num,h)
    centers,sigma = order_center(centers, sigma)
    centers_tensor = torch.tensor(centers, requires_grad=True).to(options.device)
    sigma_tensor = torch.tensor(sigma, requires_grad=True).to(options.device)
    return centers_tensor,sigma_tensor

def run():
    tokenizer = get_tokenizer("basic_english")
    # read_data('atis')
    # download_dataset()
    # translate_vocab_task()
    # translate_center_task()
    read_data('xlsum', tokenizer)
    return 0

# run()