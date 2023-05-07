from sklearn.metrics import accuracy_score
import torch
import math
from setting import options
from nltk.translate.bleu_score import sentence_bleu

def padding1d(input, limit_size):
    size = limit_size - len(input)
    output = F.pad(input, (0,size), mode="constant", value=options.PAD_token).long()
    return output

def truncat1d(input, limit_size):
    return input[0:limit_size]

def align1d(input, limit_size):
    if len(input) < limit_size:
        return padding1d(input, limit_size)
    elif len(input) > limit_size:
        return truncat1d(input, limit_size)
    else:
        return input

def calc_bleu(candidate, reference):
    bleu = sentence_bleu([reference], candidate)
    return bleu

def calc_acc(predict, target):
    target_len = len(target)
    predict_aligned = align1d(predict, target_len)
    acc = accuracy_score(target, predict_aligned)
    return acc

def calc_ppl(loss):
    ppl = math.exp(min(loss, 100.0))
    return ppl

def idx2word(vocab, source):
   corpus_list =  [vocab.index2word[idx] for idx in source]
   return corpus_list

def evaluate(output, target, vocab_tgt):
    predict  = torch.argmax(output,dim=-1)
    predict = predict.tolist()
    target = target.tolist()
    acc = calc_acc(predict, target)
    candidate = idx2word(vocab_tgt, predict)
    reference = idx2word(vocab_tgt, target)
    bleu = calc_bleu(candidate=candidate, reference=reference)
    return acc,bleu