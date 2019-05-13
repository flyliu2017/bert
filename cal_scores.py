#coding=utf8
import numpy as np
import json
from rouge import Rouge
import time
from tensorflow.contrib import predictor

UNK_ID = 0
SOS_ID = 1
EOS_ID = 2

def padding(tokens_list, max_len):
    ret = []
    for i,t in enumerate(tokens_list):
        t = t + (max_len-len(t)) * [EOS_ID]
        ret.append(t)
    return ret


def read_vocab(vocab_filepath):
    word2idx_dict = {}
    with open(vocab_filepath) as fp:
        for idx, line in enumerate(fp.readlines()):
            word = line.strip()
            word2idx_dict[word] = idx

    return word2idx_dict


class CalScore(object):
    def __init__(self, model_path='unigram_probs_model.json'):
        self.rouge = Rouge()  # codes from https://github.com/pltrdy/rouge
        self.unigram_probs = json.load(open(model_path))
        self.w2i = read_vocab('/data/xueyou/car/comment/vocab.txt')
        self.lm=predictor.from_saved_model("/data/xueyou/car/comment/lm/score/0/")

    def get_tokens(self, content):
        tokens = content.split()
        ids = []
        for t in tokens:
            if t in self.w2i:
                ids.append(self.w2i[t])
            else:
                for c in t:
                    ids.append(self.w2i.get(c,UNK_ID))
        ids = (ids + [EOS_ID])[-100:]
        return (ids)[:-1], (ids)[1:]

    # def get_ppl_from_lm(self, rewrite_tokens):
    #     """通过预训练的语言模型计算ppl"""
    #     print(time.strftime("%H:%M:%S"))
    #     rewrite_tokens = rewrite_tokens.strip('，')
    #     sources, target_tokens = self.get_tokens(rewrite_tokens)
    #
    #     len_tokens = len(sources)
    #     sources = padding([sources], len_tokens)
    #     target_tokens = padding([target_tokens], len_tokens)
    #
    #     instances = [{"sources": x, "sequence_length": l, "target_tokens": y} for x, l, y in
    #                  zip(sources, [len_tokens], target_tokens)]
    #     data = {
    #         "signature_name": "predict",
    #         "instances": instances
    #     }
    #
    #     ret_item = requests.post("https://car-comment-score-tf.aidigger.com/v1/models/car-comment-score:predict",
    #                              json=data).json()['predictions']
    #     ppl = ret_item[0]['ppl']
    #     return ppl

    def get_ppl_from_lm(self, rewrite_tokens):
        """通过预训练的语言模型计算ppl"""

        rewrite_tokens = [t.strip('，') for t in rewrite_tokens]
        tokens=map(self.get_tokens,rewrite_tokens)
        source_tokens, target_tokens = zip(*tokens)

        

        # instances = [{"sources": x, "sequence_length": l, "target_tokens": y} for x, l,y in
        #              zip(sources,len_tokens, target_tokens)]


        # data = {
        #     "signature_name": "predict",
        #     "instances": instances
        # }
        # try:
        #     response = requests.post("http://192.168.3.11:9193/predict",json=instances).json()
        # except Exception as e:
        #     raise ConnectionError(e)
        # response = requests.post("https://car-comment-score-tf.aidigger.com/v1/models/car-comment-score:predict", json=data).json()

        # if 'error' in response:
        #     raise BaseException(response['error'])

        ppls=[]
        batch_size=200
        loops=int(np.ceil(len(source_tokens)/batch_size))
        for i in range(loops):
            sources=source_tokens[batch_size*i:batch_size*i+batch_size]
            targets=target_tokens[batch_size*i:batch_size*i+batch_size]

            len_tokens = [len(t) for t in sources]
            max_len = max(len_tokens)
            sources = padding(sources, max_len)
            targets = padding(targets, max_len)
            print(time.strftime("%H:%M:%S"))
            instances = {"source_tokens": sources,
                         "sequence_length": len_tokens,
                         "target_tokens": targets}

            ppls.extend(self.lm(instances)['ppl'])

        return ppls

    def cal_slor_with_ppl(self, rewrite_tokens, ppl):
        len_tokens = len(rewrite_tokens.split())
        ## 计算SLOR分数: -ln(ppl)-ln(P(S))/|S|
        unigram_probs = 0
        for token in rewrite_tokens.split():
            token = token.lower()
            if token in self.unigram_probs:
                token_prob = self.unigram_probs[token]
            else:
                token_prob = self.unigram_probs['<unk>']
                print('assert token: {} not found...'.format(token))

            unigram_probs += np.log(token_prob)
        slor_score = -np.log(ppl) - unigram_probs / len_tokens
        return slor_score

    def cal_slor_with_entropy(self, rewrite_tokens, entropy_loss):
        len_tokens = len(rewrite_tokens.split())
        ## 计算SLOR分数: (-entropy_los - ln(P(S)))/|S|
        unigram_probs = 1.0
        for token in rewrite_tokens.split():
            token = token.lower()
            if token in self.unigram_probs:
                token_prob = self.unigram_probs[token]
            else:
                token_prob = self.unigram_probs['<unk>']
                print('assert token: {} not found...'.format(token))

            unigram_probs *= token_prob
        slor_score = -entropy_loss - np.log(unigram_probs) / len_tokens
        return slor_score

    def cal_rouge(self, rewrite_tokens, original_tokens):
        rouge_scores = self.rouge.get_scores(rewrite_tokens, original_tokens, avg=True)  # rouge scores
        rouge_score = rouge_scores['rouge-2']['f']
        return rouge_score

    def editDistance(self, s1, s2):
        """最小编辑距离"""
        if len(s1) > len(s2):
            s1, s2 = s2, s1

        distances = range(len(s1.split()) + 1)
        for i2, c2 in enumerate(s2.split()):
            distances_ = [i2 + 1]
            for i1, c1 in enumerate(s1.split()):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances[-1]


def demo():
    # original_tokens = "， 至于 外观 是 不 惊艳 但是 比较 耐 看 的 那种"  # 原始评论，单词之间以空格相隔
    # rewrite_tokens = "外观 大气 ，  动力 还 不错"  # 模型改写后评论，单词之间以空格相隔
    rewrite_tokens = ["至于 外观 是 不 惊艳 但是 比较 耐 看 的 那种","性价比 不错"]  # 模型改写后评论，单词之间以空格相隔
    # rewrite_tokens = ["性价比 不错"]  # 模型改写后评论，单词之间以空格相隔

    # load unigram model
    unigram_probs_filepath = './unigram_probs_model.json'

    model = CalScore(unigram_probs_filepath)

    ## rouge score
    # rouge_score = model.cal_rouge(rewrite_tokens, original_tokens)

    ## SLOR score
    # 根据预训练的语言模型计算ppl
    ppls = model.get_ppl_from_lm(rewrite_tokens)
    slor_score =[ model.cal_slor_with_ppl(t, ppl) for t,ppl in zip(rewrite_tokens,ppls)]

    ## minimum edit distance
    # edit_dis = model.editDistance(original_tokens, rewrite_tokens)
    # print('ROUGE-2 score: {}, SLOR score: {}'.format(rouge_score, slor_score))
    print(slor_score)
# demo()
