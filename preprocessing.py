import pandas as pd
import numpy as np
import sys
import csv
import re
from nltk.tokenize import sent_tokenize

csv.field_size_limit(1000000)


def pre_process_pb(path):
    df = pd.read_csv(path+'FinancialPhraseBank-v1.0/Sentences_66Agree.txt',sep='\@',engine='python',header=None,names=['sentence','label'])
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    df.to_csv(path+'Sentences_AllAgree_preprocessed_baseline.csv',index=False)
    df = pd.get_dummies(df, columns=['label'])
    df.to_csv(path+'Sentences_AllAgree_preprocessed.csv',index=False)
    return df

# dont know exactly what it does, but we'll apply it anyways
def clean_5(x):
    words = re.findall('\.[A-Z][a-z]*', x)
    for i in words:
        x = x.replace(i,". " + i[1:])
    return x 

def pre_process_news(path):
    df = pd.read_csv(path+'all_upto_2019-08-29.csv', sep=',', engine='python',parse_dates=['article_time'], encoding='utf-8') # TODO This is baseline manuel -> replace with better approach?
    df['symbols'] = df['symbols'].fillna(value='None')
    df = df[df['symbols'].str.contains('AAPL')]
    df = df[df['article_time'].dt.year == 2018]
    df['text'] = df['article_title']+'. '+df['article_content']
    df['text'].replace('\xa0', ' ', inplace=True)
    #df['text'].replace('\x92', '\'', inplace=True)
    df['text'].replace('\n', ' ', inplace=True)
    df['text'].replace('Inc.', '', inplace=True)
    df['text'].replace('"', '', inplace=True)
    df['text'].apply(clean_5)
    df = df[['article_time', 'text']]
    sentences = df['text'].apply(sent_tokenize)
    sentences = sentences.apply(pd.Series).stack()
    sentences.index = sentences.index.droplevel(-1)
    sentences.name = 'text'
    sentences = pd.DataFrame(sentences)
    df.drop('text', axis=1, inplace=True)
    df = df.join(sentences)
    df.to_csv(path+'/pre_processed_aapl_sentences.csv')


def pre_process_vars(path):
    nasdaq = pd.read_csv(path+'AAPL_daily.csv',usecols=["Date", "Close"],parse_dates=['Date'])
    nasdaq.columns = ["date", "nasdaq_index_close"]
    fama = pd.read_csv(path+'fama5.csv',parse_dates=[0],skiprows=2)
    fama['date'] = fama['Unnamed: 0']
    fama = fama[["date", 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
    libor = pd.read_csv(path+'LIBOR USD.csv',parse_dates=[0])
    libor = libor[["Date", "1M", "3M"]]
    libor.columns = ["date", "libor_1M","libor_3M"]
    libor["risk_premium"] = libor["libor_3M"] - libor["libor_1M"]
    usd_index = pd.read_csv(path+'USDX.csv', parse_dates=[0], usecols=['Date', 'Close'])
    usd_index.columns = ['date', 'usd_index_close']
    [x.set_index('date', inplace=True) for x in [nasdaq, fama, libor, usd_index]]
    return nasdaq, fama, libor, usd_index


