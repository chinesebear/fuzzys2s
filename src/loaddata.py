import csv
from fcmeans import FCM
import numpy as np
import torch
from nltk.tokenize import word_tokenize
from loguru import logger
from setting import options

def gen_features_map(data):
    data_list = []
    for i in range(len(data)):
        sen = data[i]
        sen_list = []
        for j in range(len(data[i])):
            d = sen[j].item()
            value = d
            position = j
            width = 2 if value > 10 else 0
            fequency = 100 if j == 0 else 10
            sen_list.append([value, position, width, fequency])
        data_list.append(sen_list)
    return torch.tensor(data_list)

# map range: [a, b]
def rescaling(x, a, b):
    data = x.view(-1, len(x[0][0]))
    col_max,_ = torch.max(data,0) # max of col
    col_min,_ = torch.min(data,0) # min of col
    x = x.float()
    for i in range(len(x)): # sentence
        for j in range(len(x[i])): # token
            for k in range(len(x[i][j])): # feature
                val = x[i][j][k]
                val = a + (val - col_min[k])*(b-a) /(col_max[k] - col_min[k])
                x[i][j][k] = val
    return x


def read_data():
    logger.info("read data...")
    with open(options.base_path+'/doc/adult.tsv', 'r', encoding='utf-8') as file_obj:
        lines_obj = csv.reader(file_obj,delimiter='\t')
        line_list =[]
        for line in lines_obj:
            if line[0] != 'age':
                for i in range(len(line)):
                    line[i] = float(line[i])
                line[-1] = int(line[-1])
            features = line[0:-2]
            target = line[-1]
            line_list.append([features, target])
        head = line_list[0]
        train_data = line_list[1:-1000]
        train_len = len(train_data)
        valid_data = line_list[-1000:-500]
        valid_len = len(valid_data)
        test_data = line_list[-500:]
        test_len = len(test_data)
        logger.info("train data: %d, valid_data:%d, test data: %d" %(train_len, valid_len, test_len))
        logger.info("data rescaling...")
        train_data = rescaling(train_data,0,1)
        valid_data = rescaling(valid_data,0,1)
        test_data = rescaling(test_data,0,1)
    return head,train_data, train_len, valid_data,valid_len, test_data, test_len

def fcm(data, cluster_num, h):
    logger.info("fcm clustering...")
    feature_num = len(data[0][0])
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

def insertSpaces(sentence, sym='.'):
    sen_list = list(sentence)
    count = sen_list.count(sym)
    if count == 0:
        return sentence
    start = 0
    for i in range(count):
        idx = sen_list.index(sym, start)
        sen_list.insert(idx, ' ')
        sen_list.insert(idx+len(sym)+1, ' ')
        start = idx+len(sym)+1 # update start postion
    return ''.join(sen_list)

def tokenizer(sentence):
    # print(sentence)
    sentence = insertSpaces(sentence, sym='.')
    sentence = insertSpaces(sentence, sym='=')
    tokens = word_tokenize(sentence)
    # ``->", ''->"
    for i in range(len(tokens)):
        token = tokens[i]
        if token == "``" or token == "''":
            tokens[i] = "\""
    return tokens

class Vocab:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {options.SOS_token: "<SOS>", options.EOS_token: "<EOS>", options.PAD_token:"<PAD>",options.UNK_token: "<UNK>"}
        self.n_words = 4  # Count PAD , SOS and EOS

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