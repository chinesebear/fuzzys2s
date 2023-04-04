from torch import nn
import torch
from setting import options
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self,feature_num , rule_num, center,sigma ):
        super(MLP, self).__init__()
        self.fc = nn.Linear(feature_num, 32)
        self.relu = nn.ReLU()
        self.drop_out= nn.Dropout(0.1)
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.drop_out2= nn.Dropout(0.1)
        self.fc3 = nn.Linear(16, 2)
    def forward(self, x):
        input = self.fc(x)
        input = self.relu(input)
        input = self.drop_out(input)
        input = self.fc2(input)
        input = self.relu2(input)
        input = self.drop_out2(input)
        output = self.fc3(input)
        return output



class FNN(nn.Module):
    def __init__(self,feature_num , rule_num, center,sigma ):
        super(FNN, self).__init__()
        self.center = nn.Parameter(center)
        self.sigma = nn.Parameter(sigma)
        self.rule_num = rule_num
        self.feature_num = feature_num
        self.relu = nn.ReLU()
        self.fc = nn.Linear(rule_num,2)
    def ms_layer(self, x):
        x_arr =  x.repeat(1, self.rule_num).view(-1, self.feature_num).to(options.device)
        value = torch.square(torch.div((x_arr-self.center) , self.sigma))
        membership = torch.exp(-(value /2))
        return membership
    def rule_layer(self,membership):
        # membership array
        # rule_num * feature_num
        rule = torch.prod(membership, 1)
        return rule
    def norm_layer(self, rule):
        output = F.normalize(rule,p=2,dim=0)
        return output.float()
    def forward(self, x):
        membership = self.ms_layer(x)
        rule = self.rule_layer(membership)
        rule = self.norm_layer(rule)
        output = self.fc(rule)
        return output


class RFNN(nn.Module):
    def __init__(self,feature_num , rule_num, center,sigma ):
        super(RFNN, self).__init__()
        self.center = nn.Parameter(center)
        self.sigma = nn.Parameter(sigma)
        self.rule_num = rule_num
        self.feature_num = feature_num
        self.rule_weight = nn.Parameter(torch.randn(rule_num))
        self.fc = nn.Linear(rule_num,2)
    def ms_layer(self, x):
        x_arr =  x.repeat(1, self.rule_num).view(-1, self.feature_num).to(options.device)
        value = torch.square(torch.div((x_arr-self.center) , self.sigma))
        membership = torch.exp(-(value /2))
        return membership
    def rule_layer(self,membership, memory):
        # membership array
        # rule_num * feature_num
        rule = torch.prod(membership, 1)
        recurrent_memory = torch.mul(memory, self.rule_weight)
        rule = torch.mul(rule, recurrent_memory)
        return rule
    def norm_layer(self, rule):
        output = F.normalize(rule,p=2,dim=0)
        return output.float()
    def forward(self, x, memory):
        membership = self.ms_layer(x)
        rule = self.rule_layer(membership, memory)
        rule = self.norm_layer(rule)
        memory = rule
        output = self.fc(rule)
        return output, memory

class RFNN_Encoder(nn.Module):
    def __init__(self,feature_in, rule_num, center,sigma ):
        super(RFNN_Encoder, self).__init__()
        self.center = nn.Parameter(center)
        self.sigma = nn.Parameter(sigma)
        self.rule_num = rule_num
        self.feature_num = feature_in
        self.rule_weight = nn.Parameter(torch.randn(rule_num))
    def ms_layer(self, x):
        x_arr =  x.repeat(1, self.rule_num).view(-1, self.feature_num).to(options.device)
        value = torch.square(torch.div((x_arr-self.center) , self.sigma))
        membership = torch.exp(-(value /2))
        return membership
    def rule_layer(self,membership, memory):
        # membership array
        # rule_num * feature_num
        memeory = memory.view(-1,1).repeat(1,self.feature_num)
        membership = membership + memeory
        rule = torch.prod(membership, 1)
        return rule.float()
    def norm_layer(self, rule):
        sum = torch.sum(rule)
        if sum.item() == 0:
            print("sum is 0")
            return rule
        rule = rule/sum
        return rule
    def rfnn_node(self, x, memory):
        membership = self.ms_layer(x)
        rule = self.rule_layer(membership, memory)
        rule = self.norm_layer(rule)
        memory = rule
        return memory
    def forward(self, x):
        memory = torch.zeros((self.rule_num)).to(options.device)
        for src in x:
            memory = self.rfnn_node(src, memory)
        return memory


class RFNN_Decoder(nn.Module):
    def __init__(self,feature_in, feature_out , rule_num, center,sigma ):
        super(RFNN_Decoder, self).__init__()
        self.center = nn.Parameter(center)
        self.sigma = nn.Parameter(sigma)
        self.rule_num = rule_num
        self.feature_num = feature_in
        self.rule_weight = nn.Parameter(torch.randn(rule_num))
        self.fc = nn.Linear(rule_num,feature_out)
    def ms_layer(self, x):
        x_arr =  x.repeat(1, self.rule_num).view(-1, self.feature_num).to(options.device)
        value = torch.square(torch.div((x_arr-self.center) , self.sigma))
        membership = torch.exp(-(value /2))
        return membership
    def rule_layer(self,membership, memory):
        # membership array
        # rule_num * feature_num
        memeory = memory.view(-1,1).repeat(1,self.feature_num)
        membership = membership + memeory
        rule = torch.prod(membership, 1)
        return rule.float()
    def norm_layer(self, rule):
        sum = torch.sum(rule)
        if sum.item() == 0:
            print("sum is 0")
            return rule
        rule = rule/sum
        return rule
    def rfnn_node(self, x, memory):
        membership = self.ms_layer(x)
        rule = self.rule_layer(membership, memory)
        rule = self.norm_layer(rule)
        memory = rule
        output = self.fc(rule)
        return output, memory
    def forward(self, x, memory):
        output = torch.tensor([]).to(options.device)
        for tgt in x:
            data, memory = self.rfnn_node(tgt, memory)
            data = data.view(-1,1)
            output = torch.cat((output,data), 1)
        return output.permute(1,0)

class FuzzyS2S(nn.Module):
    def __init__(self,feature_in, feature_out , rule_num, src_center,src_sigma,  tgt_center, tgt_sigma):
        super(FuzzyS2S, self).__init__()
        self.rule_num = rule_num
        self.feature_in = feature_in
        self.feature_out = feature_out
        self.encoder = RFNN_Encoder(feature_in, rule_num, src_center, src_sigma).to(options.device)
        self.decoder = RFNN_Decoder(feature_in, feature_out, rule_num, tgt_center, tgt_sigma).to(options.device)
    def forward(self, src, tgt):
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        return output
