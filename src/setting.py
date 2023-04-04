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
options.SOS_token = 0 # start of sentence
options.EOS_token = 1 # End of sentence
options.PAD_token = 2 # padding token
options.UNK_token = 3 # unknown token, word frequency low
options.UNK_token_count = 0
options.epoch=100
options.feature_num = 13
options.cluster_num = 32
options.rule_num = options.cluster_num
options.h = 10.0
options.drop_out = 0.1
options.learning_rate = 0.0001

def setting_info():
    output = ""
    output += "epoch: "+ str(options.epoch)+", "
    output += "rule_num: "+ str(options.rule_num)+", "
    output += "h: "+ str(options.h)+", "
    output += "drop_out: "+ str(options.drop_out)+", "
    output += "learning_rate: "+ str(options.learning_rate)
    return output