from sklearn.metrics import accuracy_score
from torch import nn
import torch
import torch.nn.functional as F
import math
from setting import options
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sacrebleu.metrics import BLEU, CHRF, TER
import evaluate
from torchtext.data import get_tokenizer
from setting import options
def padding1d(input, limit_size):
    size = limit_size - len(input)
    input = torch.Tensor(input).to(options.device)
    output = F.pad(input, (0,size), mode="constant", value=options.PAD).long()
    return output.tolist()

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

class MetricsValue(nn.Module):
    def __init__(self):
        super(MetricsValue, self).__init__()
        self.metrics_dict = {
            'acc': 0.0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'mae': 0.0,
            'mape': 0.0,
            'smape': 0.0,
            'ppl': 0.0,
            'bleu': 0.0,
            'sen_bleu': 0.0,
            'sacre_bleu': 0.0,
            'google_bleu':0.0,
            'chrf2': 0.0,
            'ter':0.0,
            'rouge1': 0.0,
            'rouge2': 0.0,
            'rougeL': 0.0,
            'rougeLsum': 0.0,
            'meteor': 0.0
        }
        self.count = 0
    def add(self, result_dict):
        for key in result_dict.keys():
            val = result_dict[key]
            if isinstance(val,str):
                continue
            val = self.metrics_dict[key] + val
            self.metrics_dict[key] = val
        self.count = self.count + 1
    def average(self, count):
        for key in self.metrics_dict.keys():
            val = self.metrics_dict[key]
            if isinstance(val,str):
                continue
            val = self.metrics_dict[key] / count
            self.metrics_dict[key] = val
        return self.metrics_dict
    def clear(self):
        for key in self.metrics_dict.keys():
            val = self.metrics_dict[key]
            if isinstance(val,str):
                self.metrics_dict[key] = ''
            else:
                self.metrics_dict[key] = 0.0
        self.count = 0
    def forward(self):
        self.average(self.count)


class Evaluator(nn.Module):
    def __init__(self):
        super(Evaluator, self).__init__()
        metrics_path = options.evaluate_path+'metrics/'
        self.google_bleu_metrics = evaluate.load(metrics_path+'google_bleu')
        # self.bleu_metrics = evaluate.load(metrics_path+'bleu')
        self.meteor_metrics = evaluate.load(metrics_path+'meteor')
        self.rouge_metrics = evaluate.load(metrics_path+'rouge')
        self.accuracy_metrics = evaluate.load(metrics_path+"accuracy")
        self.precision_metrics = evaluate.load(metrics_path+"precision")
        self.recall_metrics = evaluate.load(metrics_path+"recall")
        self.f1_metrics = evaluate.load(metrics_path+"f1")
        self.mae_metrics = evaluate.load(metrics_path+"mae")
        self.mape_metrics = evaluate.load(metrics_path+"mape")
        self.smape_metrics = evaluate.load(metrics_path+"smape")
        self.metrics_dict = MetricsValue().metrics_dict
        self.tokenizer = get_tokenizer("basic_english")
    def calc_acc(self, predict, target):
        target_len = len(target)
        predict_aligned = align1d(predict, target_len)
        acc = accuracy_score(target, predict_aligned)
        acc = acc * 100
        self.metrics_dict['acc'] = acc
        return acc
    def calc_accuracy(self, predict, target):
        target_len = len(target)
        predict_aligned = align1d(predict, target_len)
        accuracy = self.accuracy_metrics.compute(predictions=predict_aligned, references=target)
        accuracy = accuracy['accuracy'] * 100
        self.metrics_dict['accuracy'] = accuracy
        return accuracy
    def calc_precision(self, predict, target):
        target_len = len(target)
        predict_aligned = align1d(predict, target_len)
        precision = self.precision_metrics.compute(predictions=predict_aligned, references=target,average='micro')
        precision = precision['precision'] * 100
        self.metrics_dict['precision'] = precision
        return precision
    def calc_recall(self, predict, target):
        target_len = len(target)
        predict_aligned = align1d(predict, target_len)
        recall = self.recall_metrics.compute(predictions=predict_aligned, references=target,average='micro')
        recall = recall['recall'] * 100
        self.metrics_dict['recall'] = recall
        return recall
    def calc_f1(self, predict, target):
        target_len = len(target)
        predict_aligned = align1d(predict, target_len)
        f1 = self.f1_metrics.compute(predictions=predict_aligned, references=target,average='micro')
        f1 = f1['f1'] * 100
        self.metrics_dict['f1'] = f1
        return f1
    def calc_mae(self, predict, target):
        target_len = len(target)
        predict_aligned = align1d(predict, target_len)
        mae = self.mae_metrics.compute(predictions=predict_aligned, references=target)
        mae = mae['mae']
        self.metrics_dict['mae'] = mae
        return mae
    def calc_mape(self, predict, target):
        target_len = len(target)
        predict_aligned = align1d(predict, target_len)
        mape = self.mape_metrics.compute(predictions=predict_aligned, references=target)
        mape = mape['mape']
        self.metrics_dict['mape'] = mape
        return mape
    def calc_smape(self, predict, target):
        target_len = len(target)
        predict_aligned = align1d(predict, target_len)
        smape = self.smape_metrics.compute(predictions=predict_aligned, references=target)
        smape = smape['smape']
        self.metrics_dict['smape'] = smape
        return smape
    def calc_ppl(self, loss):
        ppl = math.exp(min(loss, 100.0))
        self.metrics_dict['ppl'] = ppl
        return ppl
    def calc_bleu(self, candidate, reference):
        bleu = calc_bleu_val(candidate, reference)
        bleu = bleu * 100
        self.metrics_dict['bleu'] = bleu
        return bleu
    def calc_sen_bleu(self, candidate, reference):
        candidate = self.tokenizer(candidate)
        reference = self.tokenizer(reference)
        sen_bleu = sentence_bleu([reference], candidate, smoothing_function=SmoothingFunction().method1)
        sen_bleu = sen_bleu * 100
        self.metrics_dict['sen_bleu'] = sen_bleu
        return sen_bleu
    def calc_google_bleu(self, candidate, reference):
        google_bleu = self.google_bleu_metrics.compute(predictions=[candidate], references=[[reference]])
        google_bleu = google_bleu['google_bleu'] * 100
        self.metrics_dict['google_bleu'] = google_bleu
        return google_bleu
    def calc_sacre_bleu(self, candidate, reference):
        sacrebleu_metrics = BLEU()
        sacre_bleu = sacrebleu_metrics.corpus_score([candidate], [[reference]])
        sacre_bleu = sacre_bleu.score
        self.metrics_dict['sacre_bleu'] = sacre_bleu
        return sacre_bleu
    def calc_chrf2(self, candidate, reference):
        chrf2_metrics = CHRF()
        chrf2 = chrf2_metrics.corpus_score([candidate], [[reference]])
        chrf2 = chrf2.score
        self.metrics_dict['chrf2'] = chrf2
        return chrf2
    def calc_ter(self, candidate, reference):
        ter_metrics = TER()
        ter = ter_metrics.corpus_score([candidate], [[reference]])
        ter = ter.score
        self.metrics_dict['ter'] = ter
        return ter
    def calc_rouge(self, prediction, reference):
        #['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        rouge = self.rouge_metrics.compute(predictions=[prediction],references=[reference],use_aggregator=False)
        rouge1 = rouge['rouge1'][0] * 100
        rouge2 =rouge['rouge2'][0] * 100
        rougeL = rouge['rougeL'][0] * 100
        rougeLsum = rouge['rougeLsum'][0] * 100
        self.metrics_dict['rouge1'] = rouge1
        self.metrics_dict['rouge2'] = rouge2
        self.metrics_dict['rougeL'] = rougeL
        self.metrics_dict['rougeLsum'] = rougeLsum
        result = [rouge1, rouge2,rougeL ,rougeLsum]
        return result
    def calc_meteor(self, prediction, reference):
        meteor = self.meteor_metrics.compute(predictions=[prediction], references=[reference])
        meteor = meteor['meteor'] * 100
        self.metrics_dict['meteor'] = meteor
        return meteor
    def metrics(self):
        return self.metrics_dict
    def forward(self, predict, target):
        acc = self.calc_acc(predict, target)
        bleu = self.calc_bleu(candidate=predict, reference=target)
        return acc,bleu

def run():
    evaluator = Evaluator()
    predict = [1,2,3,4,5,6,7,8,9]
    reference = [1,2,3,4,5,6,7,8,8]
    acc = evaluator.calc_acc(predict, reference)
    accuarcy = evaluator.calc_accuracy(predict, reference)
    precision = evaluator.calc_precision(predict, reference)
    recall = evaluator.calc_recall(predict, reference)
    f1 = evaluator.calc_f1(predict, reference)
    mae = evaluator.calc_mae(predict, reference)
    mape = evaluator.calc_mape(predict, reference)
    smape = evaluator.calc_smape(predict, reference)
    bleu = evaluator.calc_bleu(predict, reference)
    # prediction = "It is a guide to action which ensures that the military always obeys the commands of the party"
    # reference = "It is a guide to action that ensures that the military will forever heed Party commands"
    prediction = "the cat sat on the mat"
    reference = "the cat ate the mat" ## google bleu 0.3333
    # hypothesis1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which', 'ensures',
    #                 'that', 'the', 'military', 'always', 'obeys', 'the',
    #                 'commands', 'of', 'the', 'party']
    # reference1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that', 'ensures',
    #               'that', 'the', 'military', 'will', 'forever', 'heed',
    #               'Party', 'commands']  ## sen_bleu 0.4118
    # prediction = "1 2 3"
    # reference = "1 2 3"
    sen_bleu =evaluator.calc_sen_bleu(candidate=prediction, reference=reference)
    sacre_bleu = evaluator.calc_sacre_bleu(candidate=prediction, reference=reference)
    google_bleu = evaluator.calc_google_bleu(candidate=prediction, reference=reference)
    chrf2 = evaluator.calc_chrf2(candidate=prediction, reference=reference)
    ter = evaluator.calc_ter(candidate=prediction, reference=reference)
    rouge =evaluator.calc_rouge(prediction=prediction, reference=reference)
    meteor= evaluator.calc_meteor(prediction=prediction, reference=prediction)
    print(evaluator.metrics())
    metrics = MetricsValue()
    for i in range(100):
        metrics.add(evaluator.metrics())
    results = metrics.average(i)
    print(results)
    return 0

# run()