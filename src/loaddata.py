import csv
from fcmeans import FCM
import numpy as np
import torch
from loguru import logger
from setting import options

# map range: [a, b]
def rescaling(x, a, b):
    features = [i[0] for i in x]
    features = np.array(features)
    col_max = features.max(axis=0).tolist() # max of col
    col_min = features.min(axis=0).tolist() # min of col
    data_num = len(x)
    feature_num = len(features[0])
    for i in range(data_num):
        for j in range(feature_num):
            val = x[i][0][j]
            val = a + (val - col_min[j])*(b-a) /(col_max[j] - col_min[j])
            x[i][0][j] = val
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
    fcm = FCM(n_clusters=cluster_num)
    data = [i[0] for i in data]
    data = np.array(data)
    fcm.fit(data)
    centers = fcm.centers.tolist()
    logger.info("cluster center: %d" %(len(centers)))
    membership = fcm.soft_predict(data)
    feature_num = len(data[0])
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
    centers_tensor = torch.tensor(centers, requires_grad=True)
    sigma_tensor = torch.tensor(sigma, requires_grad=True)
    return centers_tensor,sigma_tensor