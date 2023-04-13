import csv
from fcmeans import FCM
import numpy as np
import torch
import json
from torchtext.data import get_tokenizer
from loguru import logger
from setting import options,Options

feature_options = Options("token feature")

def tokenizer(sentence):
    tokenizer = get_tokenizer("basic_english")
    tokens = tokenizer(sentence)
    return tokens

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

    def addSentence(self, sentence):
        tokens = tokenizer(sentence)
        for word in tokens:
            self.addWord(word)

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

# map range: [low, high]
def rescaling(vocab, token_features, low_limit, high_limit):
    col_max= vocab.feature_max # max of col
    col_min= vocab.feature_min # min of col
    token_features = token_features.float()
    for i in range(len(token_features)): # feature
        feature = token_features[i]
        val = low_limit + (feature - col_min[i])*(high_limit-low_limit) /(col_max[i] - col_min[i])
        token_features[i] = val
    return token_features.tolist()


def gen_sen_feature_map(vocab, sentence):
    sen_feature_list = []
    for i in range(len(sentence)):
        token_idx = sentence[i]
        word = vocab.index2word[token_idx]
        feature_word_value = token_idx
        feature_word_position = i
        feature_word_size = len(word)
        feature_word_fequency = vocab.word2count[word]
        feature_list = [feature_word_value, feature_word_position, feature_word_size, feature_word_fequency]
        sen_feature_list.append(feature_list)
    return sen_feature_list

def gen_sen_feature_map_with_rescaling(vocab, sentence):
    sen_feature_list = gen_sen_feature_map(vocab, sentence)
    for i in range(len(sen_feature_list)):
        token_features = torch.tensor(sen_feature_list[i]).to(options.device)
        token_feature_tensor = rescaling(vocab, token_features, 0, 1)
        sen_feature_list[i] = token_feature_tensor
    return sen_feature_list



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

def del_eos(sentence):
    return sentence[:-1]

def del_sos(sentence):
    return sentence[1:]

def attach_eos(sentence):
    sentence = sentence + [options.EOS]
    return sentence

def insert_sos(sentence):
    sentence = [options.SOS] + sentence
    return sentence

def gen_token_vectors(vocab_src, vocab_tgt, tokens):
    token_vectors =[]
    for row in tokens:
        src = [vocab_src.word2index[word]  for word in row[0]]
        tgt = [vocab_tgt.word2index[word]  for word in row[1]]
        token_vectors.append([src, tgt])
    return token_vectors

def get_feature_max_min_val(token_feature_map_src, token_feature_map_tgt, vocab_src, vocab_tgt):
    src_col_max,_ = torch.max(token_feature_map_src,0) # max of col
    src_col_min,_ = torch.min(token_feature_map_src,0) # min of col
    tgt_col_max,_ = torch.max(token_feature_map_tgt,0) # max of col
    tgt_col_min,_ = torch.min(token_feature_map_tgt,0) # min of col
    vocab_src.feature_max = src_col_max
    vocab_src.feature_min = src_col_min
    vocab_tgt.feature_max = tgt_col_max
    vocab_tgt.feature_min = tgt_col_min

def combine_token_feature_map(train_data, valid_data, test_data, vocab_src, vocab_tgt):
    data = train_data + valid_data + test_data
    token_feature_map_src = torch.randn(len(data)*vocab_src.line_max, options.feature_num).long().tolist()
    token_feature_map_tgt = torch.randn(len(data)*vocab_tgt.line_max, options.feature_num).long().tolist()
    src_count = 0
    tgt_count = 0
    for d in data:
        src = gen_sen_feature_map(vocab_src, d[0])
        tgt = gen_sen_feature_map(vocab_tgt, d[1])
        for feature in src:
            token_feature_map_src[src_count] = feature
            src_count = src_count + 1
        for feature in tgt:
            token_feature_map_tgt[tgt_count] = feature
            tgt_count = tgt_count + 1
    token_feature_map_src = token_feature_map_src[:src_count]
    token_feature_map_tgt = token_feature_map_tgt[:tgt_count]
    token_feature_map_src = torch.tensor(token_feature_map_src).float()
    token_feature_map_tgt = torch.tensor(token_feature_map_tgt).float()
    get_feature_max_min_val(token_feature_map_src, token_feature_map_tgt, vocab_src, vocab_tgt)
    for i in range(len(token_feature_map_src)):
        token_features = token_feature_map_src[i]
        token_feature_map_src[i] = torch.tensor(rescaling(vocab_src, token_features, 0,1))
    for i in range(len(token_feature_map_tgt)):
        token_features = token_feature_map_tgt[i]
        token_feature_map_src[i] = torch.tensor(rescaling(vocab_tgt, token_features, 0,1))
    return token_feature_map_src, token_feature_map_tgt

def fcm(data, cluster_num, h):
    logger.info("fcm clustering...")
    feature_num = len(data[0])
    fcm = FCM(n_clusters=cluster_num)
    data = data.view(-1, feature_num)
    data = np.array(data)
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

def read_file(path):
    fd = open(path,encoding = "utf-8")
    raw_data = json.loads(fd.read())
    logger.info("dataset:%s, split:%s, cofig:%s" %(raw_data['dataset'],raw_data['split'],raw_data['config']))
    raw_lines = raw_data['rows']
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

def read_wmt14_data():
    logger.info("read raw data")
    train_tokens = read_file(options.base_path+"/doc/wmt14/wmt14-fr-en-train.json")
    valid_tokens = read_file(options.base_path+"/doc/wmt14/wmt14-fr-en-valid.json")
    test_tokens = read_file(options.base_path+"/doc/wmt14/wmt14-fr-en-test.json")
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

def read_tatoeba_data():
    logger.info("read raw data")
    fd = open(options.base_path+"/doc/tatoeba/fra.txt",encoding = "utf-8")
    lines = fd.readlines()
    logger.info("dataset:tatoeba, total:%d" %(len(lines)))
    tokens = [] #  src-tgt token pairs
    for line in lines[:10000]:
        sen = line.split('\t')
        src = sen[0] # en
        tgt = sen[1] # fr
        src = "<sos> " + src + " <eos>"
        # tgt = "<sos> " + tgt + " <eos>"
        tgt = src
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
        data[i][0] = ["<sos>"] + data[i][0] + ['<eos>']
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


