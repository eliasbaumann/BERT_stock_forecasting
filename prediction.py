import pandas as pd
import numpy as np

def convert_to_sidx(data):
    data['article_time'] = pd.to_datetime(data['article_time'])
    data_t = data.set_index(data['article_time'])
    temp = data_t[['neg','neut','pos']].resample('D').sum()
    temp = temp.fillna(0)
    sidx = (temp['pos']-temp['neg'])/(temp['pos']+temp['neut']+temp['neg'])
    return sidx

def process_res(path, cat=True):
    svm_res = pd.read_csv(path+'aapl_svm.csv')
    svm_sent_idx = convert_to_sidx(svm_res)
    print(svm_sent_idx)
    

def process_bert_res(results,cat=True):
    results['pred_cat'] = results['pred_proba'].apply(np.argmax)
    results = results[["article_time","Text","pred_proba","pred_cat"]]
    # TODO consider a different alignment: articles written between closed market hours are relevant for next day, not prev day
    results['article_time'] = pd.to_datetime(results['article_time']).dt.floor('D')
    if(cat):
        inter = results.groupby(['article_time','pred_proba']).size().reset_index(name='counts')
        results_final = inter.pivot_table('counts', ['article_time'], 'pred_proba')
        results_final = results_final.fillna(0)
        results_final["sentiment_index"] = (results_final["label_positive"] - results_final["label_negative"])/(results_final["label_positive"] + results_final["label_neutral"] + results_final["label_negative"])
        results_final = results_final[["date","sentiment_index"]]
