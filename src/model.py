from torch import nn
import torch
from setting import options
import torch.nn.functional as F
import math
from loaddata import attach_eos, insert_sos,gen_sen_feature_map

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

class TransEncoder(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.1):
        super(TransEncoder, self).__init__()
        self.name = "transencoder"
        self.ninp = ninp
        self.embedding = nn.Embedding(num_embeddings=ntoken, embedding_dim=ninp)
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=ninp, nhead=nhead, dim_feedforward=nhid, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=nlayers)
        self.init_weights()
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1).to(options.device)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
    def forward(self, src):
        src_mask = self._generate_square_subsequent_mask(len(src))
        src = src.view(-1,1)
        src = self.embedding(src) * (math.sqrt(self.ninp))
        src = self.pos_encoder(src)
        memory = self.encoder(src, src_mask) # seq_len* batch* embedding_dim
        output = memory
        return output

class TransDecoder(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.1):
        super(TransDecoder, self).__init__()
        self.name = "transdecoder"
        self.ninp = ninp
        self.embedding = nn.Embedding(num_embeddings=ntoken, embedding_dim=ninp)
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        decoder_layers = nn.TransformerDecoderLayer(d_model=ninp, nhead=nhead, dim_feedforward=nhid, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layers, num_layers=nlayers)
        self.fc = nn.Linear(ninp, ntoken)
        self.init_weights()
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1).to(options.device)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
    def forward(self, tgt, memory):
        tgt_mask = self._generate_square_subsequent_mask(len(tgt))
        tgt = tgt.view(-1,1)
        tgt = self.embedding(tgt) * (math.sqrt(self.ninp))
        tgt = self.pos_encoder(tgt)
        output = self.decoder(tgt, memory, tgt_mask)
        output= self.fc(output)
        return output


class TransformerModel(nn.Module):
    def __init__(self, vocab_src_size, vocab_tgt_size, ninp, nhead, nhid, nlayers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.name = "transformer"
        self.encoder = TransEncoder(vocab_src_size, ninp, nhead, nhid, nlayers,dropout)
        self.decoder = TransDecoder(vocab_tgt_size, ninp, nhead, nhid, nlayers,dropout)
    def forward(self, src, tgt):
        src_with_eos = attach_eos(src)
        tgt_with_sos = insert_sos(tgt)
        memory = self.encoder(src_with_eos) # seq_len* batch* embedding_dim
        output = self.decoder(tgt_with_sos, memory)
        return output

class FuzzySystem(nn.Module):
    def __init__(self,feature_in, rule_num, center,sigma ):
        super(FuzzySystem, self).__init__()
        self.center = center
        self.sigma = sigma
        self.rule_num = rule_num
        self.feature_num = feature_in
    def fuzzy_layer(self, x):
        delta = x - self.center # torch tensor broadcast
        value = torch.square(torch.div(delta , self.sigma))
        membership = torch.exp(-(value /2))
        return membership
    def fire_layer(self,membership):
        # membership array
        # rule_num * feature_num
        products = torch.prod(membership, 1)
        return products.float()
    def norm_layer(self, products):
        sum = torch.sum(products)
        if sum.item() == 0:
            print("sum is 0")
            return products
        products = products/sum
        return products
    def forward(self, x):
        x = x.to(options.device)
        membership = self.fuzzy_layer(x)
        products = self.fire_layer(membership)
        products = self.norm_layer(products)
        output = products
        return output

class FuzzyEncoder(nn.Module):
    def __init__(self,src_vocab_size, feature_num, rule_num, center,sigma ):
        super(FuzzyEncoder, self).__init__()
        self.rule_num = rule_num
        self.fs = FuzzySystem(feature_num, rule_num, center, sigma)
        encoder_list = []
        for _ in range(rule_num):
            encoder_list.append(TransEncoder(src_vocab_size,
                                             options.trans.embedding_dim,
                                             options.trans.nhead,
                                             options.trans.hidden_size,
                                             options.trans.nlayer,
                                             options.trans.drop_out))
        self.encoder = nn.ModuleList(encoder_list)
    def forward(self, src, src_features):
        products = self.fs(src_features)
        output = 0
        for i in range(self.rule_num):
            product = products[i]
            if math.isnan(product):
                product = 1.0
            output = output + product * self.encoder[i](src)
        return output

class FuzzyDecoder(nn.Module):
    def __init__(self,tgt_vocab_size, feature_num, rule_num, center,sigma ):
        super(FuzzyDecoder, self).__init__()
        self.rule_num = rule_num
        self.feature_num = feature_num
        self.fs = FuzzySystem(feature_num, rule_num, center, sigma)
        decoder_list = []
        for _ in range(rule_num):
            decoder_list.append(TransDecoder(tgt_vocab_size,
                                             options.trans.embedding_dim,
                                             options.trans.nhead,
                                             options.trans.hidden_size,
                                             options.trans.nlayer,
                                             options.trans.drop_out))
        self.decoder = nn.ModuleList(decoder_list)
    def forward(self, tgt, tgt_features, memory):
        products = self.fs(tgt_features)
        output = 0
        for i in range(self.rule_num):
            product = products[i]
            if math.isnan(product):
                product = 1.0
            output = output + product * self.decoder[i](tgt, memory)
        return output

class FuzzyS2S(nn.Module):
    def __init__(self,vocab_src, vocab_tgt , feature_num, rule_num, src_center,src_sigma,  tgt_center, tgt_sigma):
        super(FuzzyS2S, self).__init__()
        self.name = 'fuzzys2s'
        self.rule_num = rule_num
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt
        src_vocab_size = vocab_src.n_words
        tgt_vocab_size = vocab_tgt.n_words
        self.encoder = FuzzyEncoder(src_vocab_size, feature_num, rule_num, src_center, src_sigma).to(options.device)
        self.decoder = FuzzyDecoder(tgt_vocab_size, feature_num, rule_num, tgt_center, tgt_sigma).to(options.device)
    def forward(self, src, tgt):
        src_with_eos = attach_eos(src)
        tgt_with_sos = insert_sos(tgt)
        src_features = torch.tensor(gen_sen_feature_map(self.vocab_src, src)).to(options.device)
        tgt_features = torch.tensor(gen_sen_feature_map(self.vocab_tgt, tgt)).to(options.device)
        memory = self.encoder(src_with_eos, src_features)
        output = self.decoder(tgt_with_sos, tgt_features, memory)
        return output
