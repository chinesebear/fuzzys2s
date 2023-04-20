import torch

class Options:
    def __init__(self, name) -> None:
        self.name= name
    def name(self):
        return self.name

# project gloal parameter
options = Options("Model")
options.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
options.base_path="/home/yang/github/fuzzys2s/"
options.model_parameter_path = options.base_path+"output/"
options.seed_id = 10
options.SOS = 0 # start of sentence
options.EOS = 1 # End of sentence
options.PAD = 2 # padding token
options.UNK = 3 # unknown token, word frequency low
options.epoch= 20
options.feature_num = 2 # [len, HF]
options.rule_num = 3
options.cluster_num = options.rule_num
options.h = 10.0
options.sen_len_max = 1000
options.high_freq_limit = 100
options.drop_out = 0.1
options.learning_rate = 0.0001


trans = Options("trans")
trans.embedding_dim = 128
trans.hidden_size = 128
trans.nlayer = 3
trans.nhead = 16
options.trans = trans

def setting_info():
    output = ""
    output += "epoch: "+ str(options.epoch)+", "
    output += "rule_num: "+ str(options.rule_num)+", "
    output += "h: "+ str(options.h)+", "
    output += "drop_out: "+ str(options.drop_out)+", "
    output += "learning_rate: "+ str(options.learning_rate)
    return output