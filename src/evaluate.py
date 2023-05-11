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

def getlen1d(input):
    """get sentence length without padding"""
    # count=0
    # for i in range(len(input)):
    #     if input[i] == options.EOS or (input[i] == options.PAD and input[i+1] == options.PAD and input[i+2] == options.PAD):
    #         return count
    #     else:
    #         count += 1
    # return count
    return len(input)

def calc_bp(candidate, reference):
    r = getlen1d(candidate)
    c = getlen1d(reference)
    bp = 0
    if c > r :
        bp = 1
    else:
        bp = math.exp(1 - r/c)
    return bp
def count_ngram(ngram, sentence, n):
    limit = len(sentence) - n + 1
    count = 0
    for i in range(limit):
        item = str(sentence[i:i+n])
        if ngram == item:
            count+=1
    return count
def calc_ngram_score(candidate, reference, n):
    count = 0
    candidate_len = getlen1d(candidate)
    reference_len = getlen1d(reference)
    candidate_ngram_dict = {}
    limit = candidate_len -n + 1
    for i in range(limit):
        ngram = str(candidate[i:i+n])
        if ngram in candidate_ngram_dict.keys():
            candidate_ngram_dict[ngram] += 1
        else:
            candidate_ngram_dict[ngram] = 1
    hc = sum(candidate_ngram_dict.values())
    limit = reference_len - n + 1
    min_hc_hs = 0
    for ngram in candidate_ngram_dict.keys():
        count = count_ngram(ngram, reference, n)
        min_hc_hs += min(count, candidate_ngram_dict[ngram])
    if min_hc_hs == 0 or hc == 0:
        return 0.0
    Pn = min_hc_hs / hc
    return Pn

def calc_bleu_val(candidate, reference):
    N = 4
    log_Pn = [0.0,0.0,0.0,0.0]
    for i in range(N):
        Pn = calc_ngram_score(candidate, reference, i+1)
        if Pn == 0:
            return 0.0
        else:
            log_Pn[i] = math.log(Pn)
    bp = calc_bp(candidate, reference)
    bleu = bp * math.exp(sum(log_Pn) / N)
    return bleu

def calc_bleu(candidate, reference):
    # bleu = sentence_bleu([reference], candidate)
    bleu = calc_bleu_val(candidate, reference)
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