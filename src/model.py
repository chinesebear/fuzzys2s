from torch import nn
import torch
from setting import options
import torch.nn.functional as F

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

# class MLP(nn.Module):
#     def __init__(self,feature_num , rule_num, center,sigma ):
#         super(MLP, self).__init__()
#         self.fc = nn.Linear(feature_num, 32)
#         self.relu = nn.ReLU()
#         self.drop_out= nn.Dropout(0.1)
#         self.fc2 = nn.Linear(32, 16)
#         self.relu2 = nn.ReLU()
#         self.drop_out2= nn.Dropout(0.1)
#         self.fc3 = nn.Linear(16, 2)
#     def forward(self, x):
#         input = self.fc(x)
#         input = self.relu(input)
#         input = self.drop_out(input)
#         input = self.fc2(input)
#         input = self.relu2(input)
#         input = self.drop_out2(input)
#         output = self.fc3(input)
#         return output