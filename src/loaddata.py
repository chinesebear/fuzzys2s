import csv
from fcmeans import FCM
import numpy as np
import torch
import json
from torchtext.data import get_tokenizer
from loguru import logger
from setting import options,Options
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE,WordPiece,Unigram
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer,WordPieceTrainer, UnigramTrainer

# def tokenizer(sentence):
#     tokenizer = get_tokenizer("basic_english")
#     tokens = tokenizer(sentence)
#     return tokens

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

    # def addSentence(self, sentence):
    #     tokens = tokenizer(sentence)
    #     for word in tokens:
    #         self.addWord(word)

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
    # copy dataset
    # features = [0,0,0,0,0,0,0,0,0,0]
    # slen = len(sentence)
    # if slen > options.feature_num:
    #     slen = options.feature_num
    # for i in range(slen):
    #     features[i] = sentence[i]
    # return features

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

def train_tokenizer(dataset,src_lang, tgt_lang, trainer,tokenizer):
    train_len = options.tok.train_len # dataset['train'].num_rows
    test_len = options.tok.test_len # dataset['test'].num_rows
    valid_len = options.tok.valid_len # dataset['validation'].num_rows
    raw_lines = np.empty([train_len + test_len + valid_len], dtype = int).tolist()
    train_iter = iter(dataset['train'])
    offset = 0
    for i in range(train_len):
        data = next(train_iter)
        src = data['translation'][src_lang]
        tgt = data['translation'][tgt_lang]
        raw_lines[i + offset] = src +' '+tgt
    test_iter = iter(dataset['test'])
    offset = offset + train_len
    for i in range(test_len):
        data = next(test_iter)
        src = data['translation'][src_lang]
        tgt = data['translation'][tgt_lang]
        raw_lines[i + offset] = src +' '+tgt
    valid_iter = iter(dataset['validation'])
    offset = offset + test_len
    for i in range(valid_len):
        data = next(valid_iter)
        src = data['translation'][src_lang]
        tgt = data['translation'][tgt_lang]
        raw_lines[i + offset] = src +' '+tgt
    tokenizer.train_from_iterator(raw_lines, trainer=trainer)

def read_raw_tokens(dataset, src_lang, tgt_lang,tokenizer):
    train_len = options.tok.train_len # dataset['train'].num_rows
    test_len = options.tok.test_len # dataset['test'].num_rows
    valid_len = options.tok.valid_len # dataset['validation'].num_rows
    # dataset = dataset.shuffle()
    train_raw_tokens = np.empty([train_len], dtype=int).tolist()
    train_iter = iter(dataset['train'])
    for i in range(train_len):
        data = next(train_iter)
        src = data['translation'][src_lang]
        tgt = data['translation'][tgt_lang]
        src = tokenizer.encode(src).tokens
        tgt = tokenizer.encode(tgt).tokens
        train_raw_tokens[i] = [src, tgt]
    test_raw_tokens = np.empty([test_len], dtype=int).tolist()
    test_iter = iter(dataset['test'])
    for i in range(test_len):
        data = next(test_iter)
        src = data['translation'][src_lang]
        tgt = data['translation'][tgt_lang]
        src = tokenizer.encode(src).tokens
        tgt = tokenizer.encode(tgt).tokens
        test_raw_tokens[i] = [src, tgt]
    valid_raw_tokens = np.empty([test_len], dtype=int).tolist()
    valid_iter = iter(dataset['validation'])
    for i in range(valid_len):
        data = next(valid_iter)
        src = data['translation'][src_lang]
        tgt = data['translation'][tgt_lang]
        src = tokenizer.encode(src).tokens
        tgt = tokenizer.encode(tgt).tokens
        valid_raw_tokens[i] = [src, tgt]
    return train_raw_tokens, test_raw_tokens, valid_raw_tokens

def read_file(path):
    fd = open(path,encoding = "utf-8")
    raw_lines = json.loads(fd.read())
    logger.info("dataset:%s, split:%s, cofig:%s" %(raw_lines['dataset'],raw_lines['split'],raw_lines['config']))
    raw_lines = raw_lines['rows']
    tokens = []
    for line in raw_lines:
        src = line['row']['translation']['en']
        tgt = line['row']['translation']['fr']
        src = "<sos> " + src + " <eos>"
        tgt = "<sos> " + tgt + " <eos>"
        src = tokenizer(src)
        tgt = tokenizer(tgt)
        tokens.append([src, tgt])
    fd.close()
    return tokens

def read_tatoeba_data():
    logger.info("read raw data")
    fd = open(options.base_path+"/doc/tatoeba/fra.txt",encoding = "utf-8")
    lines = fd.readlines()
    logger.info("dataset:tatoeba, total:%d" %(len(lines)))
    tokens = [] #  src-tgt token pairs
    for line in lines:
        sen = line.split('\t')
        src = sen[0] # en
        tgt = sen[1] # fr
        src = tokenizer(src)
        tgt = tokenizer(tgt)
        tokens.append([src, tgt])
    fd.close()
    total = len(tokens)
    part = int(total/10)
    train_tokens = tokens[:total - part*2]
    valid_tokens = tokens[total - part*2:total - part]
    test_tokens = tokens[total - part:]
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

def read_wmt14_data():
    logger.info("train tokenizer")
    src_lang = 'en'
    tgt_lang = 'fr'
    dataset = load_dataset('wmt14', 'fr-en')
    tokenizer = Tokenizer(Unigram())
    tokenizer.pre_tokenizer = Whitespace()
    trainer = UnigramTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    train_tokenizer(dataset, src_lang, tgt_lang, trainer, tokenizer)
    logger.info("read raw tokens")
    train_tokens, test_tokens,valid_tokens = read_raw_tokens(dataset, src_lang, tgt_lang,tokenizer)
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

def read_opus100_data():
    logger.info("train tokenizer")
    src_lang = 'en'
    tgt_lang = 'fr'
    dataset = load_dataset('opus100', 'en-fr')
    tokenizer = Tokenizer(Unigram())
    tokenizer.pre_tokenizer = Whitespace()
    trainer = UnigramTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    train_tokenizer(dataset, src_lang, tgt_lang, trainer, tokenizer)
    logger.info("read raw data")
    train_tokens, test_tokens,valid_tokens = read_raw_tokens(dataset, src_lang, tgt_lang,tokenizer)
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

def read_heartstone_data():
    logger.info("read raw data")
    logger.info("build vocabulary")


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
    return train_data, valid_data, test_data, vocab_src, vocab_tgt



def read_data(dataset):
    if dataset =="wmt14":
        return read_wmt14_data()
    elif dataset == "tatoeba":
        return read_tatoeba_data()
    elif dataset == "copy":
        return read_copy_data()
    elif dataset == 'opus100':
        return read_opus100_data()


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