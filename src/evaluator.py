from sklearn.metrics import accuracy_score
from torch import nn
import torch.nn.functional as F
import math
from setting import options
from nltk.translate.bleu_score import sentence_bleu
import evaluate

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

# def calc_bleu(candidate, reference):
#     bleu = calc_bleu_val(candidate, reference)
#     return bleu

# def calc_sen_bleu(candidate, reference, vocab):
#     candidate = idx2word(vocab, candidate)
#     reference = idx2word(vocab, reference)
#     bleu = sentence_bleu([reference], candidate)
#     return bleu

# def calc_acc(predict, target):
#     target_len = len(target)
#     predict_aligned = align1d(predict, target_len)
#     acc = accuracy_score(target, predict_aligned)
#     return acc

# def calc_ppl(loss):
#     ppl = math.exp(min(loss, 100.0))
#     return ppl

# def idx2word(vocab, source):
#    corpus_list =  [vocab.index2word[idx] for idx in source]
#    return corpus_list

# def evaluate(output, target, vocab_tgt):
#     predict  = torch.argmax(output,dim=-1)
#     predict = predict.tolist()
#     target = target.tolist()
#     acc = calc_acc(predict, target)
#     candidate = predict
#     reference = target
#     bleu = calc_bleu(candidate=candidate, reference=reference)
#     return acc,bleu

class Evaluator(nn.Module):
    def __init__(self):
        super(Evaluator, self).__init__()
        metrics_path = options.evaluate_path+'metrics/'
        self.meteor_metric = evaluate.load(metrics_path+'meteor')
        self.rouge_metric = evaluate.load(metrics_path+'rouge')
        self.accuracy_metric = evaluate.load(metrics_path+"accuracy")
        self.precision_metric = evaluate.load(metrics_path+"precision")
        self.recall_metric = evaluate.load(metrics_path+"recall")
        self.f1_metric = evaluate.load(metrics_path+"f1")
        self.mae_metric = evaluate.load(metrics_path+"mae")
        self.mape_metric = evaluate.load(metrics_path+"mape")
        self.smape_metric = evaluate.load(metrics_path+"smape")
        self.acc = 0.0
        self.accuracy = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.f1 = 0.0
        self.mae = 0.0
        self.mape = 0.0
        self.smape = 0.0
        self.ppl = 0.0
        self.bleu = 0.0
        self.sen_bleu = 0.0
        self.rouge = [0.0,0.0,0.0,0.0]
        self.meteor = 0.0
    def calc_acc(self, predict, target):
        target_len = len(target)
        predict_aligned = align1d(predict, target_len)
        acc = accuracy_score(target, predict_aligned)
        self.acc = acc
        return self.acc
    def calc_accuracy(self, predict, target):
        target_len = len(target)
        predict_aligned = align1d(predict, target_len)
        accuracy = self.accuracy_metric.compute(predictions=predict_aligned, references=target)
        self.accuracy = accuracy['accuracy']
        return self.accuracy
    def calc_precision(self, predict, target):
        precision = self.precision_metric.compute(predictions=predict, references=target,average='micro')
        self.precision = precision['precision']
        return self.precision
    def calc_recall(self, predict, target):
        recall = self.recall_metric.compute(predictions=predict, references=target,average='micro')
        self.recall = recall['recall']
        return self.recall
    def calc_f1(self, predict, target):
        f1 = self.f1_metric.compute(predictions=predict, references=target,average='micro')
        self.f1 = f1['f1']
        return self.f1
    def calc_mae(self, predict, target):
        mae = self.mae_metric.compute(predictions=predict, references=target)
        self.mae = mae['mae']
        return self.mae
    def calc_mape(self, predict, target):
        mape = self.mape_metric.compute(predictions=predict, references=target)
        self.mape = mape['mape']
        return self.mape
    def calc_smape(self, predict, target):
        smape = self.smape_metric.compute(predictions=predict, references=target)
        self.smape = smape['smape']
        return self.smape
    def calc_ppl(self, loss):
        ppl = math.exp(min(loss, 100.0))
        self.ppl = ppl
        return self.ppl
    def calc_bleu(self, candidate, reference):
        bleu = calc_bleu_val(candidate, reference)
        self.belu = bleu
        return self.bleu
    def calc_sen_bleu(self, candidate, reference):
        bleu = sentence_bleu([reference], candidate)
        self.sen_belu = bleu
        return self.sen_bleu
    def calc_rouge(self, prediction, reference):
        #['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        rouge = self.rouge_metric.compute(predictions=prediction,references=reference,use_aggregator=False)
        self.rouge = [rouge['rouge1'][0], rouge['rouge2'][0],rouge['rougeL'][0],rouge['rougeLsum'][0]]
        return self.rouge
    def calc_meteor(self, prediction, reference):
        meteor = self.meteor_metric.compute(predictions=prediction, references=reference)
        self.meteor = meteor['meteor']
        return self.meteor
    def results(self):
        return [self.acc, self.accuracy,self.precision, self.recall,
                self.f1, self.mae, self.mape, self.smape, self.ppl, self.bleu, self.sen_bleu,
                self.rouge[0], self.rouge[1], self.rouge[2],self.rouge[3], self.meteor]
    def forward(self, predict, target):
        acc = self.calc_acc(predict, target)
        candidate = predict
        reference = target
        bleu = self.calc_bleu(candidate=candidate, reference=reference)
        return acc,bleu

def run():
    # evaluator = Evaluator()
    # predict = [1,2,3,4,5,6,7,8,9]
    # reference = [1,2,3,4,5,6,7,8,8]
    # acc = evaluator.calc_acc(predict, reference)
    # accuarcy = evaluator.calc_accuracy(predict, reference)
    # precision = evaluator.calc_precision(predict, reference)
    # recall = evaluator.calc_recall(predict, reference)
    # f1 = evaluator.calc_f1(predict, reference)
    # mae = evaluator.calc_mae(predict, reference)
    # mape = evaluator.calc_mape(predict, reference)
    # smape = evaluator.calc_smape(predict, reference)
    # bleu = evaluator.calc_bleu(predict, reference)
    # prediction = ["It is a guide to action which ensures that the military always obeys the commands of the party"]
    # reference = ["It is a guide to action that ensures that the military will forever heed Party commands"]
    # sen_bleu =evaluator.calc_sen_bleu(candidate=prediction[0].split(' '), reference=reference[0].split(' '))
    # rouge =evaluator.calc_rouge(prediction=prediction, reference=reference)
    # meteor= evaluator.calc_meteor(prediction=prediction[0].split(' '), reference=prediction[0].split(' '))
    # print(evaluator.results())
    return 0

run()