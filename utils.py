from sklearn.metrics import precision_recall_fscore_support
import numpy as np

def report_metrics(preds,trues):
    metric_results=[]
    count=0
    for pred,true in zip(preds,trues):
        metric_results.append(precision_recall_fscore_support(true,pred,average='binary'))
        if pred==true:
            count+=1

    total=len(metric_results)
    precision, recall, fscore, _ = list(zip(*metric_results))
    report = 'accuracy: {}\n'.format(count / total) + \
             'precision: {}\n'.format(sum(precision) / total) + \
             'recall: {}\n'.format(sum(recall) / total) + \
             'fscore: {}\n'.format(sum(fscore) / total)
    return report

def precision_recall_f1score(y_true,probs,threthold_num=20):
    epsilon=1e-9
    y_true = np.array(y_true)==1
    total_true=sum(y_true)
    probs=np.array(probs)
    thretholds=list(np.arange(0.,1.,1/threthold_num))+[1+epsilon]
    true=np.array([probs>=threthold for threthold in thretholds])
    tps=np.sum(y_true & true,-1)
    precisions=tps/(np.sum(true,-1)+epsilon)
    recalls=tps/total_true
    f1scores=2*precisions*recalls/(precisions+recalls+epsilon)

    return precisions,recalls,f1scores

