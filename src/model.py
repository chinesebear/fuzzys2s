from torch import nn
import torch
from setting import options
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.ninp = ninp
        self.src_mask = None
        self.embedding = nn.Embedding(num_embeddings=ntoken, embedding_dim=ninp)
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=ninp, nhead=nhead, dim_feedforward=nhid, dropout=dropout)
        decoder_layer = nn.TransformerDecoderLayer(d_model=ninp, nhead=nhead,dim_feedforward=nhid, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=nlayers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=nlayers)
        self.linear = nn.Linear(ninp, ntoken)
        self.softmax = nn.Softmax(-1)
        self.init_weights()
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1).to(options.device)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
    def forward(self, src, tgt, src_mask, tgt_mask):
        src = src.view(-1,1)
        src = self.embedding(src) * (math.sqrt(self.ninp))
        src = self.pos_encoder(src)
        tgt = tgt.view(-1,1)
        tgt = self.embedding(tgt) * (math.sqrt(self.ninp))
        tgt = self.pos_encoder(tgt)
        memory = self.encoder(src, src_mask) # seq_len* batch* embedding_dim
        output = self.decoder(tgt, memory, tgt_mask)
        output = self.linear(output)
        return output
    def Encoder(self, src, src_mask):
        src = src.view(-1,1)
        src = self.embedding(src) * (math.sqrt(self.ninp))
        src = self.pos_encoder(src)
        memory = self.encoder(src, src_mask)
        return memory
    def Decoder(self, tgt, memory, tgt_mask):
        tgt = tgt.view(-1,1)
        tgt = self.embedding(tgt) * (math.sqrt(self.ninp))
        tgt = self.pos_encoder(tgt)
        output = self.decoder(tgt, memory, tgt_mask)
        output = self.linear(output)
        output = self.softmax(output)
        return output

class MLP(nn.Module):
    def __init__(self,feature_in , feature_out):
        super(MLP, self).__init__()
        self.fc = nn.Linear(feature_in, feature_in * 4)
        self.relu = nn.ReLU()
        self.drop_out= nn.Dropout(0.1)
        self.fc2 = nn.Linear(feature_in * 4, feature_out *4)
        self.relu2 = nn.ReLU()
        self.drop_out2= nn.Dropout(0.1)
        self.fc3 = nn.Linear( feature_out *4, feature_out)
        self.relu3 = nn.ReLU()
        self.drop_out3= nn.Dropout(0.1)
    def forward(self, x):
        input = self.fc(x)
        # input = self.relu(input)
        # input = self.drop_out(input)
        input = self.fc2(input)
        # input = self.relu2(input)
        # input = self.drop_out2(input)
        output = self.fc3(input)
        # input = self.relu3(input)
        # output = self.drop_out3(input)
        return output

class RFNN(nn.Module):
    def __init__(self,feature_in, rule_num, center,sigma ):
        super(RFNN, self).__init__()
        self.center = nn.Parameter(center)
        self.sigma = nn.Parameter(sigma)
        self.rule_num = rule_num
        self.feature_num = feature_in
    def fuzzy_layer(self, x):
        # delta = x - self.center # torch tensor broadcast
        # value = torch.square(torch.div(delta , self.sigma))
        # membership = torch.exp(-(value /2))
        x_arr =  x.repeat(1, self.rule_num).view(-1, self.feature_num).to(options.device)
        value = torch.square(torch.div((x_arr-self.center) , self.sigma))
        membership = torch.exp(-(value /2))
        return membership
    def fire_layer(self,membership, recurrent_connection):
        # membership array
        # rule_num * feature_num
        # recurrent_connection = recurrent_connection.view(-1, 1)
        # membership = membership + recurrent_connection #
        # products = torch.prod(membership, 1)
        # return products.float()
        memeory = recurrent_connection.view(-1,1).repeat(1,self.feature_num)
        membership = membership + memeory
        rule = torch.prod(membership, 1)
        return rule.float()
    def norm_layer(self, products):
        sum = torch.sum(products)
        if sum.item() == 0:
            print("sum is 0")
            return products
        products = products/sum
        return products
    def emit_layer(self, products):
        output = self.fc(products)
        return output
    def forward(self, x, memory):
        x = x.to(options.device)
        membership = self.fuzzy_layer(x)
        recurrent_connection = memory #torch.mul(memory,self.recurrent_weight)
        products = self.fire_layer(membership, recurrent_connection)
        products = self.norm_layer(products)
        # output = self.emit_layer(products)
        output = products
        return output

class RFS_Encoder(nn.Module):
    def __init__(self,feature_in, rule_num, center,sigma ):
        super(RFS_Encoder, self).__init__()
        self.rule_num = rule_num
        self.feature_num = feature_in
        self.rfs_block = RFNN(feature_in, rule_num, center, sigma)
    def forward(self, src):
        memory = torch.zeros((self.rule_num)).to(options.device)
        for x  in src:
            memory = self.rfs_block(x, memory)
        output = memory
        return output


class RFS_Decoder(nn.Module):
    def __init__(self,feature_in, feature_out , rule_num, center,sigma ):
        super(RFS_Decoder, self).__init__()
        self.rule_num = rule_num
        self.feature_num = feature_in
        self.feature_out = feature_out
        self.rfs_block = RFNN(feature_in, rule_num, center, sigma)
        self.mlp = nn.Linear(rule_num,feature_out)
        # self.mlp = MLP(rule_num, feature_out)
    def forward(self, tgt, memory):
        output = torch.tensor([]).view(-1, self.feature_out).to(options.device)
        for x in tgt:
            memory = self.rfs_block(x, memory)
            data = self.mlp(memory).view(1,-1)
            output = torch.cat((output, data),0)
        return output

class FuzzyS2S(nn.Module):
    def __init__(self,feature_in, feature_out , rule_num, src_center,src_sigma,  tgt_center, tgt_sigma):
        super(FuzzyS2S, self).__init__()
        self.rule_num = rule_num
        self.feature_in = feature_in
        self.feature_out = feature_out
        self.encoder = RFS_Encoder(feature_in, rule_num, src_center, src_sigma).to(options.device)
        self.decoder = RFS_Decoder(feature_in, feature_out, rule_num, tgt_center, tgt_sigma).to(options.device)
    def forward(self, src, tgt):
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        return output
