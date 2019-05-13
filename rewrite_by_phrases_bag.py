# pip install simplex_sdk
import json
import os
import re
import time

import jieba as jieba
from simplex_sdk import SimplexClient
import tensorflow as tf
import numpy as np
import multiprocessing
from cal_scores import CalScore

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The data dir.")
flags.DEFINE_string(
    "input_file", None,
    "The input file. ")
flags.DEFINE_string(
    "output_file", None,
    "The output file. ")
flags.DEFINE_string(
    "device", None,
    "CUDA_VISIBLE_DEVICES ")
flags.DEFINE_integer(
    "rewrite_num", None,
    "number of inputs to rewrite.")

def get_tags(inputs,batch_size=10):
    client = SimplexClient('BertCarMultiLabelsExtractTopK')
    results=[]
    iter=(len(inputs)+batch_size-1)//batch_size
    for i in range(iter):
        data=[{"content":n} for n in inputs[i*batch_size:(i+1)*batch_size]]
        ret = client.predict(data)
        print(ret)
        ret=[n['tags'] for n in ret]
        ret=[[n['tag'] for n in l] for l in ret]
        results.extend(ret)
    return results

def get_phrases(pred_tags,raws):
    lengths = [len(n) for n in pred_tags]
    repeat_inputs = []
    for i in range(len(lengths)):
        repeat_inputs.extend([raws[i]] * lengths[i])

    flatten = [n for l in pred_tags for n in l]
    combine = [i + ' | ' + tag for i, tag in zip(repeat_inputs, flatten)]
    combined_file = os.path.join(data_dir, 'combined_' + FLAGS.output_file)
    with open(combined_file, 'w', encoding='utf8') as f:
        f.write('\n'.join(combine))

    cmd = 'python run_token_level_classifier_from_file.py    ' \
          '--task_name=fromfile    ' \
          '--do_predict=true   ' \
          '--data_dir=/data/share/liuchang/comments_dayu/tag_prediction/rewrite   ' \
          '--vocab_file=/data/share/ludezheng/bert/chinese_L-12_H-768_A-12/vocab.txt    ' \
          '--bert_config_file=/data/share/ludezheng/bert/chinese_L-12_H-768_A-12/bert_config.json    ' \
          '--init_checkpoint=/data/share/ludezheng/bert/chinese_L-12_H-768_A-12/bert_model.ckpt    ' \
          '--max_seq_length=128    ' \
          '--train_batch_size=32    ' \
          '--learning_rate=2e-5    ' \
          '--num_train_epochs=6.0 ' \
          '--output_dir=/data/share/liuchang/comments_dayu/tag_prediction/rewrite/ ' \
          '--from_file {} '.format(combined_file)
    ret=os.system(cmd)
    if ret:
        raise SystemError('get_phrase failed.')       



def get_candidates(phrases_bag:dict,tags,old_phrases):
    candidate_bags={}
    for phrase,tag in zip(old_phrases,tags):
        if phrase in candidate_bags:
            candidate_bags[phrase]=candidate_bags[phrase]+phrases_bag[tag]
        else:
            candidate_bags[phrase]=phrases_bag[tag]
    return candidate_bags

def generate_all_rewrite_candidates(sentence,candidate_bags):
    if not candidate_bags:
        return [sentence.lower()]
    keys=list(candidate_bags.keys())
    semi_rewrites=generate_all_rewrite_candidates(sentence,{key:candidate_bags[key] for key in keys[1:]})
    key=keys[0]
    rewrites=[semi_rewrite.replace(key,'***' + s + '***') for s in candidate_bags[key] for semi_rewrite in semi_rewrites]
    return rewrites

def choose_rewrite_by_ppl(lm,sentence,candidate_bags,candidate_num=5,strategy='beamsearch'):
    for phrase in candidate_bags:
        bag=candidate_bags[phrase]
        if len(bag)>candidate_num:
            candidate_bags[phrase]=np.random.choice(bag,candidate_num)

    def choose_rewrite(sentence,candidate_bags):
        candidates = generate_all_rewrite_candidates(sentence, candidate_bags)
        seged_candidates = [' '.join(list(jieba.cut(n.replace('***', '')))) for n in candidates]
        ppls = lm.get_ppl_from_lm(seged_candidates)
        sentence = candidates[ppls.index(min(ppls))]
        return sentence

    if strategy=='beamsearch':
        return choose_rewrite(sentence,candidate_bags)
    # elif strategy=='greedy':
    else:
        for key in candidate_bags:
            sentence=choose_rewrite(sentence,{key:candidate_bags[key]})

        return sentence



def marks_to_phrases(marks,tags,sentence):
    old_phrases = re.split(r'[，。；！？]', sentence.lower())
    old_phrases = [n for n in old_phrases if n.strip() != '']

    phrases = []
    new_tags=[]
    for mark,tag in zip(marks,tags):
        if mark=='':
            continue
        if '，' in mark:
            phrases.append(mark)
            new_tags.append(tag)
        else:
            for phrase in old_phrases:
                if mark in phrase:
                    phrases.append(phrase)
                    new_tags.append(tag)
                    break

    return phrases,new_tags

def rewrite(phrases_bag:dict,sentence,tags,old_phrases,language_model:CalScore,candidate_num=5,strategy='beamsearch'):
    candidate_bags=get_candidates(phrases_bag,tags,old_phrases)

    if strategy in ['beamsearch','greedy']:
        return choose_rewrite_by_ppl(language_model,sentence,candidate_bags,candidate_num=candidate_num,strategy=strategy)

    if strategy=='random':
        for phrase in candidate_bags:
            new=np.random.choice(candidate_bags[phrase])
            sentence = sentence.replace(phrase, '***' + new + '***')
        return sentence

    raise ValueError('strategy must be one of "beamsearch,greedy,random".')

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']=FLAGS.device

    data_dir=FLAGS.data_dir
    with open(os.path.join(data_dir,FLAGS.input_file), 'r', encoding='utf8') as f:
        inputs = f.read().splitlines()
        if FLAGS.rewrite_num:
            inputs=inputs[:FLAGS.rewrite_num]
        raws,true_tags=list(zip(*[n.split(' | ',1) for n in inputs]))

    output_file=os.path.join(data_dir,FLAGS.output_file)


    if not os.path.isfile(output_file):
        pred_tags=get_tags(raws)
        outstr=[' | '.join(n) for n in pred_tags]
        print(pred_tags)
        with open(os.path.join(data_dir,FLAGS.output_file), 'w', encoding='utf8') as f:
            f.write('\n'.join(outstr))
    else:
        with open(os.path.join(data_dir, FLAGS.output_file), 'r', encoding='utf8') as f:
            outstr=f.read().splitlines()
            pred_tags=[n.split(' | ') for n in outstr]

    # get_phrases(pred_tags,raws)
    
    with open(os.path.join(data_dir,'phrases_bag.json'), 'r', encoding='utf8') as f:
        phrases_bag = json.load(f)
    with open(os.path.join('/data/share/liuchang/comments_dayu/tag_prediction/rewrite/test_results.tsv'), 'r', encoding='utf8') as f:
        marks = f.read().splitlines()

    count=0
    rewrites=[]
    lm=CalScore()

    lengths = [len(n) for n in pred_tags]
    for sentence,tags,length in zip(raws, pred_tags, lengths):
        phrases,new_tags=marks_to_phrases(marks[count:count+length],tags,sentence)
        count+=length
        new_sentence=rewrite(phrases_bag,sentence,tags,phrases,lm,strategy='beamsearch')
        print(new_sentence)
        rewrites.append(new_sentence)

    timestr = time.strftime("%y-%m-%d_%H-%M-%S")
    with open(os.path.join('/data/share/liuchang/comments_dayu/tag_prediction/rewrite','rewirtes_' + timestr), 'w', encoding='utf8') as f:
        f.write('\n'.join(rewrites))

    with open(os.path.join('/data/share/liuchang/comments_dayu/tag_prediction/rewrite','compare_' + timestr), 'w', encoding='utf8') as f:
        for rewrite,tagstr,input in zip(rewrites,outstr,inputs):
            f.write(input+'\n'+rewrite+' | '+tagstr+'\n\n')

