import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from sklearn.model_selection import train_test_split

from preprocessing import pre_process_vars


def convert_to_sidx(data, name):
    data['article_time'] = pd.to_datetime(data['article_time'])
    data_t = data.set_index(data['article_time'])
    temp = data_t[['neg','neut','pos']].resample('D').sum()
    temp = temp.fillna(0)
    sidx = np.divide(temp['pos']-temp['neg'],temp['pos']+temp['neut']+temp['neg'])
    res = pd.DataFrame(sidx,index=temp.index, columns=[name])
    return res

def process_res(path, cat=True): #TODO alternative approach when using different index
    svm_res = pd.read_csv(path+'aapl_svm.csv')
    svm_sent_idx = convert_to_sidx(svm_res,'svm_sidx')
    dict_res = pd.read_csv(path+'aapl_lex.csv')
    lex_sent_idx = convert_to_sidx(dict_res,'lex_sidx')
    bert_res = pd.read_csv(path+'aapl_bert.csv')
    bert_sent_idx = convert_to_sidx(bert_res,'bert_sidx')
    nasdaq, fama, libor, usd_index = pre_process_vars(path)
    combined = nasdaq.join([fama,libor,usd_index,svm_sent_idx,lex_sent_idx,bert_sent_idx], how='outer')
    combined = combined[combined.index.year==2018]
    combined = combined[combined.index.dayofweek < 5]
    combined.drop(combined.index[0],inplace=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    combined[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'libor_1M', 'libor_3M', 'risk_premium', 'usd_index_close']] = scaler.fit_transform(combined[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'libor_1M', 'libor_3M','risk_premium', 'usd_index_close']])
    # TODO this explicitly does it for two steps back shift:
    combined_t_1 = combined.shift(-1).add_suffix('(t-1)')
    combined_t_2 = combined.shift(-2).add_suffix('(t-2)')
    df_cmb = pd.concat([combined,combined_t_1,combined_t_2],axis=1)
    df_cmb['log_ret'] = np.log(df_cmb['nasdaq_index_close']/df_cmb['nasdaq_index_close(t-1)'])
    df_cmb['log_ret(t-1)'] = np.log(df_cmb['nasdaq_index_close(t-1)']/df_cmb['nasdaq_index_close(t-2)'])
    dropcols = ~df_cmb.columns.str.contains('nasdaq',case=False, na=False)
    final_data = df_cmb.loc[:,dropcols]
    final_data.dropna(axis=0, inplace=True)
    return final_data

def create_subsets(data):
    train_bert = (data.drop(['log_ret', 'svm_sidx(t-2)', 'lex_sidx(t-2)', 'svm_sidx(t-1)', 'lex_sidx(t-1)', 'svm_sidx', 'lex_sidx'],axis=1),data['log_ret'].values,'bert')
    train_lex = (data.drop(['log_ret', 'svm_sidx(t-2)', 'bert_sidx(t-2)', 'svm_sidx(t-1)', 'bert_sidx(t-1)', 'svm_sidx', 'bert_sidx'],axis=1),data['log_ret'].values,'dict')
    train_svm = (data.drop(['log_ret', 'bert_sidx(t-2)', 'lex_sidx(t-2)', 'bert_sidx(t-1)', 'lex_sidx(t-1)', 'bert_sidx', 'lex_sidx'],axis=1),data['log_ret'].values,'svm')
    train_none = (data.drop(['log_ret', 'svm_sidx(t-2)', 'lex_sidx(t-2)', 'bert_sidx(t-2)', 'svm_sidx(t-1)', 'lex_sidx(t-1)', 'bert_sidx(t-1)', 'svm_sidx', 'lex_sidx', 'bert_sidx'],axis=1),data['log_ret'].values,'None')
    return [train_bert, train_lex, train_svm, train_none]

def predict(data):
    subsets = create_subsets(data)
    for j in subsets:
        x_train, x_test, y_train, y_test = train_test_split(j[0],j[1],test_size=0.2, random_state=42)
        x_train = np.expand_dims(x_train,axis=1)
        x_test = np.expand_dims(x_test,axis=1)
        rmse= []
        for i in range(10):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.LSTM(50, input_shape=(x_train.shape[1], x_train.shape[2]),activation=None))
            model.add(tf.keras.layers.Dense(1,activation=None))
            model.compile(loss='mae', optimizer='adam')
            model.fit(x_train, y_train, epochs=100, batch_size=100, validation_data=(x_test, y_test),verbose=0, shuffle=False)
            yhat = model.predict(x_test)
            rmse_i = np.sqrt(tf.keras.losses.MSE(y_test, yhat).numpy())
            rmse.append(rmse_i)
        print(j[2]+': ')
        print(np.mean(rmse))

    