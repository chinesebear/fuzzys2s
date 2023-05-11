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
    for sentence in tokens:
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
    token_vectors =[]
    for row in tokens:
        src = [vocab_src.word2index[word]  for word in row[0]]
        tgt = [vocab_tgt.word2index[word]  for word in row[1]]
        token_vectors.append([src, tgt])
    return token_vectors

def read_raw_tokens(dataset, src_lang, tgt_lang, tokenizer):
    train_len = options.tok.train_len # dataset['train'].num_rows
    test_len = options.tok.test_len # dataset['test'].num_rows
    valid_len = options.tok.valid_len # dataset['validation'].num_rows
    train_raw_tokens = np.empty([train_len], dtype=int).tolist()
    train_iter = iter(dataset['train'])
    for i in tqdm(range(train_len),'read train data'):
        data = next(train_iter)
        src = data['translation'][src_lang]
        tgt = data['translation'][tgt_lang]
        src = tokenizer(src)
        tgt = tokenizer(tgt)
        train_raw_tokens[i] = [src, tgt]
    test_raw_tokens = np.empty([test_len], dtype=int).tolist()
    test_iter = iter(dataset['test'])
    for i in range(test_len):
        data = next(test_iter)
        src = data['translation'][src_lang]
        tgt = data['translation'][tgt_lang]
        src = tokenizer(src)
        tgt = tokenizer(tgt)
        test_raw_tokens[i] = [src, tgt]
    valid_raw_tokens = np.empty([valid_len], dtype=int).tolist()
    valid_iter = iter(dataset['validation'])
    for i in range(valid_len):
        data = next(valid_iter)
        src = data['translation'][src_lang]
        tgt = data['translation'][tgt_lang]
        src = tokenizer(src)
        tgt = tokenizer(tgt)
        valid_raw_tokens[i] = [src, tgt]
    return train_raw_tokens, test_raw_tokens, valid_raw_tokens

def download_dataset():
    datasets = [
        ['wmt14', 'fr-en'],
        ['wmt14', 'de-en'],
        ['wmt16', 'de-en'],
        # ['wmt19', 'de-en'],
        # ['lambada',''],
        ['spider', ''],
        # ['wikisql', ''],
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

def read_tatoeba_data(tokenizer):
    logger.info("read raw data")
    fd = open(options.base_path+"/doc/tatoeba/fra.txt",encoding = "utf-8")
    lines = fd.readlines()
    logger.info("dataset:tatoeba, total:%d" %(len(lines)))
    tokens = np.empty((len(lines),2)).tolist() #  src-tgt token pairs
    line_iter = iter(lines)
    for i in tqdm(range(len(lines)), 'read data'):
        line = next(line_iter)
        sen = line.split('\t')
        src = sen[0] # en
        tgt = sen[1] # fr
        src = tokenizer(src)
        tgt = tokenizer(tgt)
        tokens[i][0] = src
        tokens[i][1] = tgt
    fd.close()
    total = len(tokens)
    part = 2000
    train_tokens = tokens[:total - part*2]
    valid_tokens = tokens[total - part*2:total - part]
    test_tokens = tokens[total - part:]
    logger.info("dataset: tatoeba, train: %d, valid: %d, test: %d" %(len(train_tokens),len(valid_tokens), len(test_tokens)))
    return train_tokens, valid_tokens,  test_tokens

def read_wmt14_data(tokenizer):
    logger.info("read wmt14 data")
    src_lang = 'en'
    tgt_lang = 'de'
    dataset = read_dataset('wmt14', 'de-en')
    logger.info("read raw tokens")
    train_tokens, test_tokens,valid_tokens = read_raw_tokens(dataset, src_lang, tgt_lang, tokenizer)
    logger.info("dataset: wmt14, train: %d, valid: %d, test: %d" %(len(train_tokens),len(valid_tokens), len(test_tokens)))
    return train_tokens, valid_tokens,  test_tokens

def read_wmt16_data(tokenizer):
    logger.info("read wmt16 data")
    src_lang = 'en'
    tgt_lang = 'de'
    dataset = read_dataset('wmt16', 'de-en')
    logger.info("read raw tokens")
    train_tokens, test_tokens,valid_tokens = read_raw_tokens(dataset, src_lang, tgt_lang, tokenizer)
    logger.info("dataset: wmt16, train: %d, valid: %d, test: %d" %(len(train_tokens),len(valid_tokens), len(test_tokens)))
    return train_tokens, valid_tokens,  test_tokens

def read_opus100_data(tokenizer):
    logger.info("read opus100")
    src_lang = 'en'
    tgt_lang = 'fr'
    dataset = read_dataset('opus100', 'en-fr')
    logger.info("read raw data")
    train_tokens, test_tokens,valid_tokens = read_raw_tokens(dataset, src_lang, tgt_lang,tokenizer)
    logger.info("dataset: opus100, train: %d, valid: %d, test: %d" %(len(train_tokens),len(valid_tokens), len(test_tokens)))
    return train_tokens, valid_tokens,  test_tokens

def read_hearthstone_data(tokenizer):
    logger.info("read raw data")
    train_lines = read_line_pair(options.base_path+'doc/hearthstone/train_hs.in', options.base_path+'doc/hearthstone/train_hs.out')
    valid_lines = read_line_pair(options.base_path+'doc/hearthstone/dev_hs.in', options.base_path+'doc/hearthstone/dev_hs.out')
    test_lines = read_line_pair(options.base_path+'doc/hearthstone/test_hs.in', options.base_path+'doc/hearthstone/test_hs.out')
    train_tokens=[]
    valid_tokens=[]
    test_tokens=[]
    for src, tgt in train_lines:
        src_tokens = tokenizer(src)
        tgt_tokens = tokenizer(tgt)
        train_tokens.append([src_tokens, tgt_tokens])
    for src, tgt in valid_lines:
        src_tokens = tokenizer(src)
        tgt_tokens = tokenizer(tgt)
        valid_tokens.append([src_tokens, tgt_tokens])
    for src, tgt in test_lines:
        src_tokens = tokenizer(src)
        tgt_tokens = tokenizer(tgt)
        test_tokens.append([src_tokens, tgt_tokens])
    logger.info("dataset: hearthstone, train: %d, valid: %d, test: %d" %(len(train_tokens),len(valid_tokens), len(test_tokens)))
    return train_tokens, valid_tokens,  test_tokens

def read_magic_data(tokenizer):
    logger.info("read raw data")
    train_lines = read_line_pair(options.base_path+'doc/magic/train_magic.in', options.base_path+'doc/magic/train_magic.out')
    valid_lines = read_line_pair(options.base_path+'doc/magic/dev_magic.in', options.base_path+'doc/magic/dev_magic.out')
    test_lines = read_line_pair(options.base_path+'doc/magic/test_magic.in', options.base_path+'doc/magic/test_magic.out')
    train_tokens=[]
    valid_tokens=[]
    test_tokens=[]
    for src, tgt in train_lines:
        src_tokens = tokenizer(src)
        tgt_tokens = tokenizer(tgt)
        train_tokens.append([src_tokens, tgt_tokens])
    for src, tgt in valid_lines:
        src_tokens = tokenizer(src)
        tgt_tokens = tokenizer(tgt)
        valid_tokens.append([src_tokens, tgt_tokens])
    for src, tgt in test_lines:
        src_tokens = tokenizer(src)
        tgt_tokens = tokenizer(tgt)
        test_tokens.append([src_tokens, tgt_tokens])
    logger.info("dataset: magic, train: %d, valid: %d, test: %d" %(len(train_tokens),len(valid_tokens), len(test_tokens)))
    return train_tokens, valid_tokens,  test_tokens

def read_spider_data(tokenizer):
    logger.info("read spider data")
    dataset = read_dataset('spider', '')
    logger.info("read raw tokens")
    train_len = dataset['train'].num_rows
    valid_len = dataset['validation'].num_rows
    train_tokens = np.empty([train_len], dtype=int).tolist()
    train_iter = iter(dataset['train'])
    for i in range(train_len):
        data = next(train_iter)
        src = data['question_toks']
        tgt = data['query_toks']
        train_tokens[i] = [src, tgt]
    valid_tokens = np.empty([valid_len], dtype=int).tolist()
    valid_iter = iter(dataset['validation'])
    for i in range(valid_len):
        data = next(valid_iter)
        src = data['question_toks']
        tgt = data['query_toks']
        valid_tokens[i] = [src, tgt]
    test_tokens= valid_tokens[int(valid_len/2):]
    valid_tokens = valid_tokens[:int(valid_len/2)]
    logger.info("dataset: spider, train: %d, valid: %d, test: %d" %(len(train_tokens),len(valid_tokens), len(test_tokens)))
    return train_tokens, valid_tokens,  test_tokens

def read_geo_data(tokenizer):
    logger.info("read geo data")
    dataset = read_dataset('dvitel/geo', '')
    logger.info("read raw tokens")
    train_len = dataset['train'].num_rows
    train_tokens = np.empty([train_len], dtype=int).tolist()
    train_iter = iter(dataset['train'])
    for i in range(train_len):
        data = next(train_iter)
        src = data['source']
        tgt = data['target']
        src = tokenizer(src)
        tgt = tokenizer(tgt)
        train_tokens[i] = [src, tgt]
    part = int(train_len/10)
    # test_tokens= train_tokens[train_len-2*part:train_len-part]
    # valid_tokens = train_tokens[train_len-part:]
    # train_tokens=train_tokens[:train_len-2*part]
    test_tokens= train_tokens[train_len-part:]
    valid_tokens = train_tokens[train_len-part:]
    train_tokens=train_tokens[:train_len-part]
    logger.info("dataset: geo, train: %d, valid: %d, test: %d" %(len(train_tokens),len(valid_tokens), len(test_tokens)))
    return train_tokens, valid_tokens,  test_tokens

def read_django_data(tokenizer):
    logger.info("read django data")
    dataset = read_dataset('AhmedSSoliman/DJANGO', '')
    logger.info("read raw tokens")
    train_len = dataset['train'].num_rows
    valid_len = dataset['validation'].num_rows
    test_len = dataset['test'].num_rows
    train_tokens = np.empty([train_len], dtype=int).tolist()
    train_iter = iter(dataset['train'])
    for i in range(train_len):
        data = next(train_iter)
        src = data['nl']
        tgt = data['code']
        src = tokenizer(src)
        tgt = tokenizer(tgt)
        train_tokens[i] = [src, tgt]
    valid_tokens = np.empty([valid_len], dtype=int).tolist()
    valid_iter = iter(dataset['validation'])
    for i in range(valid_len):
        data = next(valid_iter)
        src = data['nl']
        tgt = data['code']
        src = tokenizer(src)
        tgt = tokenizer(tgt)
        valid_tokens[i] = [src, tgt]
    test_tokens = np.empty([test_len], dtype=int).tolist()
    test_iter = iter(dataset['test'])
    for i in range(test_len):
        data = next(test_iter)
        src = data['nl']
        tgt = data['code']
        src = tokenizer(src)
        tgt = tokenizer(tgt)
        test_tokens[i] = [src, tgt]
    logger.info("dataset: django, train: %d, valid: %d, test: %d" %(len(train_tokens),len(valid_tokens), len(test_tokens)))
    return train_tokens, valid_tokens,  test_tokens

def read_conala_data(tokenizer):
    logger.info("read conala data")
    dataset = read_dataset('neulab/conala', '')
    logger.info("read raw tokens")
    train_len = dataset['train'].num_rows
    test_len = dataset['test'].num_rows
    train_tokens = np.empty([train_len], dtype=int).tolist()
    train_iter = iter(dataset['train'])
    for i in range(train_len):
        data = next(train_iter)
        if data['rewritten_intent'] == None:
            src = data['intent']
        else:
            src = data['rewritten_intent']
        src = src.replace('\\', '#').replace('/', '#')
        tgt = data['snippet'].replace('\\', '#').replace('/', '#')
        src = tokenizer(src)
        tgt = tokenizer(tgt)
        train_tokens[i] = [src, tgt]
    test_tokens = np.empty([test_len], dtype=int).tolist()
    test_iter = iter(dataset['test'])
    for i in range(test_len):
        data = next(test_iter)
        if data['rewritten_intent'] == None:
            src = data['intent']
        else:
            src = data['rewritten_intent']
        src = src.replace('\\', '#').replace('/', '#')
        tgt = data['snippet'].replace('\\', '#').replace('/', '#')
        src = tokenizer(src)
        tgt = tokenizer(tgt)
        test_tokens[i] = [src, tgt]
    part = int(test_len/2)
    valid_tokens = train_tokens[:part]
    test_tokens = test_tokens[part:]
    logger.info("dataset: conala, train: %d, valid: %d, test: %d" %(len(train_tokens),len(valid_tokens), len(test_tokens)))
    return train_tokens, valid_tokens,  test_tokens

def read_opus_euconst_data(tokenizer):
    logger.info("read opus_euconst data")
    dataset = read_dataset('neulab/conala', '')
    logger.info("read raw tokens")
    train_len = dataset['train'].num_rows
    train_tokens = np.empty([train_len], dtype=int).tolist()
    train_iter = iter(dataset['train'])
    for i in range(train_len):
        data = next(train_iter)
        if data['rewritten_intent'] == None:
            src = data['intent']
        else:
            src = data['rewritten_intent']
        src = src.replace('\\', '#').replace('/', '#')
        tgt = data['snippet'].replace('\\', '#').replace('/', '#')
        src = tokenizer(src)
        tgt = tokenizer(tgt)
        train_tokens[i] = [src, tgt]
    part = int(train_len/10)
    valid_tokens = train_tokens[train_len -2 *part: train_len - part]
    test_tokens = train_tokens[train_len -part:]
    train_tokens = train_tokens[:train_len -2*part]
    logger.info("dataset: opus_euconst, train: %d, valid: %d, test: %d" %(len(train_tokens),len(valid_tokens), len(test_tokens)))
    return train_tokens, valid_tokens,  test_tokens

def read_cnn_dailymail_data(tokenizer):
    logger.info("read cnn_dailymail data")
    dataset = read_dataset('cnn_dailymail', '1.0.0')
    logger.info("read raw tokens")
    train_len = dataset['train'].num_rows
    valid_len = dataset['validation'].num_rows
    test_len = dataset['test'].num_rows
    train_tokens = np.empty([train_len], dtype=int).tolist()
    train_iter = iter(dataset['train'])
    for i in tqdm(range(train_len), 'read train data'):
        data = next(train_iter)
        src = data['article']
        tgt = data['highlights']
        src = tokenizer(src)
        tgt = tokenizer(tgt)
        train_tokens[i] = [src, tgt]
    valid_tokens = np.empty([valid_len], dtype=int).tolist()
    valid_iter = iter(dataset['validation'])
    for i in tqdm(range(valid_len), 'read valid data'):
        data = next(valid_iter)
        src = data['article']
        tgt = data['highlights']
        src = tokenizer(src)
        tgt = tokenizer(tgt)
        valid_tokens[i] = [src, tgt]
    test_tokens = np.empty([test_len], dtype=int).tolist()
    test_iter = iter(dataset['test'])
    for i in tqdm(range(test_len), 'read test data'):
        data = next(test_iter)
        src = data['article']
        tgt = data['highlights']
        src = tokenizer(src)
        tgt = tokenizer(tgt)
        test_tokens[i] = [src, tgt]
    logger.info("dataset: cnn dailymail, train: %d, valid: %d, test: %d" %(len(train_tokens),len(valid_tokens), len(test_tokens)))
    return train_tokens, valid_tokens,  test_tokens

def read_samsum_data(tokenizer):
    logger.info("read samsum data")
    dataset = read_dataset('samsum', '')
    logger.info("read raw tokens")
    train_len = dataset['train'].num_rows
    valid_len = dataset['validation'].num_rows
    test_len = dataset['test'].num_rows
    train_tokens = np.empty([train_len], dtype=int).tolist()
    train_iter = iter(dataset['train'])
    for i in tqdm(range(train_len), 'read train data'):
        data = next(train_iter)
        src = data['dialogue']
        tgt = data['summary']
        src = tokenizer(src)
        tgt = tokenizer(tgt)
        train_tokens[i] = [src, tgt]
    valid_tokens = np.empty([valid_len], dtype=int).tolist()
    valid_iter = iter(dataset['validation'])
    for i in tqdm(range(valid_len), 'read valid data'):
        data = next(valid_iter)
        src = data['dialogue']
        tgt = data['summary']
        src = tokenizer(src)
        tgt = tokenizer(tgt)
        valid_tokens[i] = [src, tgt]
    test_tokens = np.empty([test_len], dtype=int).tolist()
    test_iter = iter(dataset['test'])
    for i in tqdm(range(test_len), 'read test data'):
        data = next(test_iter)
        src = data['dialogue']
        tgt = data['summary']
        src = tokenizer(src)
        tgt = tokenizer(tgt)
        test_tokens[i] = [src, tgt]
    logger.info("dataset: samsum, train: %d, valid: %d, test: %d" %(len(train_tokens),len(valid_tokens), len(test_tokens)))
    return train_tokens, valid_tokens,  test_tokens

def read_gem_data(tokenizer):
    logger.info("read gem data")
    dataset = read_dataset('gem', 'common_gen')
    logger.info("read raw tokens")
    train_len = dataset['train'].num_rows
    valid_len = dataset['validation'].num_rows
    test_len = dataset['test'].num_rows
    train_tokens = np.empty([train_len], dtype=int).tolist()
    train_iter = iter(dataset['train'])
    for i in tqdm(range(train_len), 'read train data'):
        data = next(train_iter)
        src = data['concepts']
        tgt = data['target']
        tgt = tokenizer(tgt)
        train_tokens[i] = [src, tgt]
    valid_tokens = np.empty([valid_len], dtype=int).tolist()
    valid_iter = iter(dataset['validation'])
    for i in tqdm(range(valid_len), 'read valid data'):
        data = next(valid_iter)
        src = data['concepts']
        tgt = data['target']
        tgt = tokenizer(tgt)
        valid_tokens[i] = [src, tgt]
    test_tokens = np.empty([test_len], dtype=int).tolist()
    test_iter = iter(dataset['test'])
    for i in tqdm(range(test_len), 'read test data'):
        data = next(test_iter)
        src = data['concepts']
        tgt = data['target']
        tgt = tokenizer(tgt)
        test_tokens[i] = [src, tgt]
    logger.info("dataset: gem, train: %d, valid: %d, test: %d" %(len(train_tokens),len(valid_tokens), len(test_tokens)))
    return train_tokens, valid_tokens,  test_tokens

def read_xlsum_data(tokenizer):
    logger.info("read GEM/xlsum data")
    dataset = read_dataset('GEM/xlsum', 'french')
    logger.info("read raw tokens")
    train_len = dataset['train'].num_rows
    valid_len = dataset['validation'].num_rows
    test_len = dataset['test'].num_rows
    train_tokens = np.empty([train_len], dtype=int).tolist()
    train_iter = iter(dataset['train'])
    for i in tqdm(range(train_len), 'read train data'):
        data = next(train_iter)
        src = data['text']
        tgt = data['target']
        src = tokenizer(src)
        tgt = tokenizer(tgt)
        train_tokens[i] = [src, tgt]
    valid_tokens = np.empty([valid_len], dtype=int).tolist()
    valid_iter = iter(dataset['validation'])
    for i in tqdm(range(valid_len), 'read valid data'):
        data = next(valid_iter)
        src = data['text']
        tgt = data['target']
        src = tokenizer(src)
        tgt = tokenizer(tgt)
        valid_tokens[i] = [src, tgt]
    test_tokens = np.empty([test_len], dtype=int).tolist()
    test_iter = iter(dataset['test'])
    for i in tqdm(range(test_len), 'read test data'):
        data = next(test_iter)
        src = data['text']
        tgt = data['target']
        src = tokenizer(src)
        tgt = tokenizer(tgt)
        test_tokens[i] = [src, tgt]
    logger.info("dataset: xlsum, train: %d, valid: %d, test: %d" %(len(train_tokens),len(valid_tokens), len(test_tokens)))
    return train_tokens, valid_tokens,  test_tokens

def read_atis_data(tokenizer):
    logger.info("read atis data")
    dataset = read_dataset('fathyshalab/atis_intents', '')
    logger.info("read raw tokens")
    train_len = dataset['train'].num_rows
    test_len = dataset['test'].num_rows
    train_tokens = np.empty([train_len], dtype=int).tolist()
    train_iter = iter(dataset['train'])
    for i in tqdm(range(train_len), 'read train data'):
        data = next(train_iter)
        src = data['text']
        tgt = data['label text']
        src = tokenizer(src)
        tgt = [tgt]
        train_tokens[i] = [src, tgt]
    test_tokens = np.empty([test_len], dtype=int).tolist()
    test_iter = iter(dataset['test'])
    for i in tqdm(range(test_len), 'read test data'):
        data = next(test_iter)
        src = data['text']
        tgt = data['label text']
        tgt = [tgt]
        src = tokenizer(src)
        test_tokens[i] = [src, tgt]
    part = int(test_len/2)
    valid_tokens = test_tokens[:part]
    test_tokens = test_tokens[part:]
    logger.info("dataset: atis, train: %d, valid: %d, test: %d" %(len(train_tokens),len(valid_tokens), len(test_tokens)))
    return train_tokens, valid_tokens,  test_tokens

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

def read_raw_data(dataset, tokenizer):
    if dataset =="wmt14":
        train_tokens, valid_tokens,  test_tokens = read_wmt14_data(tokenizer)
    elif dataset == "wmt16":
        train_tokens, valid_tokens,  test_tokens = read_wmt16_data(tokenizer)
    elif dataset == "tatoeba":
        train_tokens, valid_tokens,  test_tokens = read_tatoeba_data(tokenizer)
    elif dataset == 'opus100':
        train_tokens, valid_tokens,  test_tokens =  read_opus100_data(tokenizer)
    elif dataset == 'hearthstone':
        train_tokens, valid_tokens,  test_tokens =  read_hearthstone_data(tokenizer)
    elif dataset == 'magic':
        train_tokens, valid_tokens,  test_tokens =  read_magic_data(tokenizer)
    elif dataset == "spider":
        train_tokens, valid_tokens,  test_tokens =  read_spider_data(tokenizer)
    elif dataset == "geo":
        train_tokens, valid_tokens,  test_tokens =  read_geo_data(tokenizer)
    elif dataset == 'django':
        train_tokens, valid_tokens,  test_tokens =  read_django_data(tokenizer)
    elif dataset == 'conala':
        train_tokens, valid_tokens,  test_tokens =  read_conala_data(tokenizer)
    elif dataset == 'opus_euconst':
        train_tokens, valid_tokens,  test_tokens =  read_opus_euconst_data(tokenizer)
    elif dataset == 'cnn_dailymail':
        train_tokens, valid_tokens,  test_tokens =  read_cnn_dailymail_data(tokenizer)
    elif dataset == 'samsum':
        train_tokens, valid_tokens,  test_tokens =  read_samsum_data(tokenizer)
    elif dataset == 'gem':
        train_tokens, valid_tokens,  test_tokens =  read_gem_data(tokenizer)
    elif dataset == 'xlsum':
        train_tokens, valid_tokens,  test_tokens =  read_xlsum_data(tokenizer)
    elif dataset == 'atis':
        train_tokens, valid_tokens,  test_tokens =  read_atis_data(tokenizer)
    return train_tokens, valid_tokens,  test_tokens

def read_data(dataset, tokenizer):
    if dataset == "copy":
        return  read_copy_data()
    else:
        train_tokens, valid_tokens,  test_tokens = read_raw_data(dataset, tokenizer)
        return gen_feature_data(train_tokens, valid_tokens,  test_tokens)

def fcm(data, cluster_num, h):
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
    centers_tensor = torch.tensor(centers, requires_grad=True).to(options.device)
    sigma_tensor = torch.tensor(sigma, requires_grad=True).to(options.device)
    return centers_tensor,sigma_tensor

def run():
    # read_data('atis')
    # download_dataset()
    return 0

run()