import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from preprocessing import pre_process_vars


def convert_to_sidx(data, name):
    '''
    Function to create a sentiment index column
    data: DataFrame with 'article_time' and 'neg', 'neut', 'pos' columns
    name: name of output column
    '''
    data['article_time'] = pd.to_datetime(data['article_time'])
    data_t = data.set_index(data['article_time'])
    temp = data_t[['neg', 'neut', 'pos']].resample('D').sum()
    temp = temp.fillna(0)
    sidx = np.divide(temp['pos']-temp['neg'],temp['pos']+temp['neut']+temp['neg'])
    res = pd.DataFrame(sidx, index=temp.index, columns=[name])
    return res

def process_res(path, inc_lag=True):
    '''
    Process different sentiment predictions.
    Function requires:
        - aapl_svm.csv (baselines.py)
        - aapl_lex.csv (baselines.py)
        - aapl_bert.csv (bert_model.py)
    inc_lag: whether to implicitly include lagged information in every row. In this instance it is hard coded to two steps
    This function also currently is hardcoded to work with one method of sentiment index creation.
    '''
    svm_res = pd.read_csv(path+'aapl_svm.csv')
    svm_sent_idx = convert_to_sidx(svm_res, 'svm_sidx')
    dict_res = pd.read_csv(path+'aapl_lex.csv')
    lex_sent_idx = convert_to_sidx(dict_res, 'lex_sidx')
    bert_res = pd.read_csv(path+'aapl_bert.csv')
    bert_sent_idx = convert_to_sidx(bert_res, 'bert_sidx')
    nasdaq, fama, libor, usd_index = pre_process_vars(path)
    combined = nasdaq.join([fama, libor, usd_index, svm_sent_idx, lex_sent_idx, bert_sent_idx], how='outer')
    combined = combined[combined.index.year == 2018]
    combined = combined[combined.index.dayofweek < 5]
    combined.drop(combined.index[0], inplace=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    combined[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA',
              'libor_1M', 'libor_3M', 'risk_premium', 'usd_index_close'
              ]] = scaler.fit_transform(combined[['Mkt-RF', 'SMB', 'HML', 'RMW',
                                                  'CMA', 'libor_1M', 'libor_3M',
                                                  'risk_premium', 'usd_index_close']])
    
    if inc_lag:
        # this explicitly does it for two steps back shift:
        combined_t_1 = combined.shift(-1).add_suffix('(t-1)')
        combined_t_2 = combined.shift(-2).add_suffix('(t-2)')
        df_cmb = pd.concat([combined, combined_t_1, combined_t_2], axis=1)
        df_cmb['log_ret'] = np.log(df_cmb['nasdaq_index_close']/df_cmb['nasdaq_index_close(t-1)'])
        df_cmb['log_ret(t-1)'] = np.log(df_cmb['nasdaq_index_close(t-1)']/df_cmb['nasdaq_index_close(t-2)'])
    else:
        combined['log_ret'] = np.log(combined['nasdaq_index_close']).diff()
        df_cmb = combined
    dropcols = ~df_cmb.columns.str.contains('nasdaq',case=False, na=False)
    df_cmb = df_cmb.loc[:, dropcols].copy()
    df_cmb.dropna(axis=0, inplace=True)
    return df_cmb

def create_subsets(data, convert_log_ret=False):
    '''
    Create subsets for every sentiment index where the dataset includes lag
    convert_log_ret: can be used to create a binary target variable for economic indicator prediction
    '''
    if convert_log_ret:
        data['log_ret'] = np.array(list(map(lambda x: 1. if x>=1 else 0., np.exp(data['log_ret']))))
        data['log_ret(t-1)'] = np.array(list(map(lambda x: 1. if x>=1 else 0., np.exp(data['log_ret(t-1)']))))
    train_bert = (data.drop(['log_ret', 'svm_sidx(t-2)', 'lex_sidx(t-2)', 'svm_sidx(t-1)', 'lex_sidx(t-1)', 'svm_sidx', 'lex_sidx'], axis=1), data['log_ret'].values, 'bert')
    train_lex = (data.drop(['log_ret', 'svm_sidx(t-2)', 'bert_sidx(t-2)', 'svm_sidx(t-1)', 'bert_sidx(t-1)', 'svm_sidx', 'bert_sidx'], axis=1), data['log_ret'].values, 'dict')
    train_svm = (data.drop(['log_ret', 'bert_sidx(t-2)', 'lex_sidx(t-2)', 'bert_sidx(t-1)', 'lex_sidx(t-1)', 'bert_sidx', 'lex_sidx'], axis=1), data['log_ret'].values, 'svm')
    train_none = (data.drop(['log_ret', 'svm_sidx(t-2)', 'lex_sidx(t-2)', 'bert_sidx(t-2)', 'svm_sidx(t-1)', 'lex_sidx(t-1)', 'bert_sidx(t-1)', 'svm_sidx', 'lex_sidx', 'bert_sidx'], axis=1), data['log_ret'].values, 'None')
    return [train_bert, train_lex, train_svm, train_none]

def create_subsets_nolag(data, convert_log_ret=False):
    '''
    Create subsets for every sentiment index where the dataset does not include lagged variables
    convert_log_ret: can be used to create a binary target variable for economic indicator prediction
    '''
    if convert_log_ret:
        data['log_ret'] = np.array(list(map(lambda x: 1. if x>=1 else 0., np.exp(data['log_ret']))))
    train_bert = (data.drop(['log_ret', 'svm_sidx', 'lex_sidx'], axis=1), data['log_ret'].values, 'bert')
    train_lex = (data.drop(['log_ret', 'svm_sidx', 'bert_sidx'], axis=1), data['log_ret'].values, 'dict')
    train_svm = (data.drop(['log_ret', 'bert_sidx', 'lex_sidx'], axis=1), data['log_ret'].values, 'svm')
    train_none = (data.drop(['log_ret', 'svm_sidx', 'lex_sidx', 'bert_sidx'], axis=1), data['log_ret'].values, 'No_sidx')
    return [train_bert, train_lex, train_svm, train_none]

def rmse_loss(targets, predictions):
    '''
    Root mean squared error
    targets and predictions should have the same shape and same datatype
    '''
    return np.sqrt(np.mean((predictions-targets)**2))

def predict_lstm(data, pred_bin=False):
    '''
    Predict target variable using LSTM
    To avoid problems with random initialization, a mean over 10 training iterations is used.
    data: DataFrame, should have been created using the other functions available in this code
    pred_bin: True: predict binary target variable with AUC as evaluation metric and binary crossentropy loss
              False: predict continuous target variable with RMSE as evaluation metric and MAE loss
    '''
    subsets = create_subsets(data, convert_log_ret=pred_bin)
    activation = None
    loss_func = 'mae'
    eval_loss = rmse_loss
    if pred_bin:
        activation = tf.keras.activations.sigmoid
        loss_func = tf.keras.losses.BinaryCrossentropy()
        eval_loss = roc_auc_score
    for j in subsets:
        x_train, x_test, y_train, y_test = train_test_split(j[0], j[1], test_size=0.2, random_state=42)
        x_train = np.expand_dims(x_train, axis=1)
        x_test = np.expand_dims(x_test, axis=1)
        acc_eval_loss=[]
        for _ in range(10):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.LSTM(50, input_shape=(x_train.shape[1], x_train.shape[2]), activation=None))
            model.add(tf.keras.layers.Dense(1, activation=activation))
            model.compile(loss=loss_func, optimizer='adam')
            model.fit(x_train, y_train, epochs=100, batch_size=100, validation_data=(x_test, y_test),verbose=0, shuffle=False)
            yhat = model.predict(x_test)
            eval_loss_i = eval_loss(y_test, yhat) #np.sqrt(tf.keras.losses.MSE(y_test, yhat).numpy())
            acc_eval_loss.append(eval_loss_i)
        print(j[2]+': ')
        print(np.mean(acc_eval_loss))

def diff_n_times(data, n):
    '''
    Function to differentiate a Dataframe n times
    data: Dataframe
    n: how many times to differentiate
    '''
    print('differencing '+str(n)+' times')
    for _ in range(n):
        data = data.diff().dropna()
    return data

def eval_for_var(data):
    '''
    Evaluate data to make sure that it fits requirements for Vector autoregression
    data: Dataframe from preprocessing steps
    '''
    data.drop(['risk_premium'], axis=1, inplace=True)
    data = diff_n_times(data, 1) # TODO: You should probably not difference all columns
                                 # but selectively the once which are not stationary.
                                 # This code includes a number of evaluation functions to check for stationarity
    subsets = create_subsets_nolag(data)
    for j in subsets:
        y = pd.DataFrame(j[1])
        y.index = j[0].index
        y.columns = ['log_return']
        df = pd.concat([j[0], y], axis=1)
        print(grangers_causation_matrix(df, variables=df.columns))
        cointegration_test(df)
        for _, column in df.iteritems():
            adfuller_test(column, name=column.name)
            print('\n')

def predict_var(data):
    ''' Predict continuous data using vector autoregression. Evaluation metric: RMSE
        As the Summary print is too long for some editors, it is stored as a Text file
        (var_res.txt)
    '''
    data.drop(['risk_premium'], axis=1, inplace=True) # TODO Statsmodels VAR only seems to work with 10 variables
                                                      #      I randomly chose to drop risk_premium, if you leave it in
                                                      #      you will get an error message.
    data = diff_n_times(data, 1) # TODO: You should probably not difference all columns
                                 # but selectively the once which are not stationary.
                                 # This code includes a number of evaluation functions to check for stationarity
    subsets = create_subsets_nolag(data)
    for j in subsets:
        cmb = np.concatenate((j[0], np.expand_dims(j[1], 1)), axis=1)
        nobs = int(0.2*len(j[1]))
        train = cmb[:-nobs] # Simple train, validation split
        valid = cmb[-nobs:]
        train = pd.DataFrame(train, columns=list(j[0].columns)+['log_return'])
        model = VAR(train)
        results = model.fit(2)
        file1 = open("var_res.txt", "a")
        file1.write(j[2])
        file1.write(str(results.summary()))
        file1.close()
        print(j[2])
        get_durbin_watson(list(j[0].columns)+['log_return'], results)
        res = []
        for x, y in zip(valid[:-2], valid[1:-1]):
            pred = results.forecast([x, y], 1)
            res.append(pred)
        res = np.vstack(res)
        df_res = undiff_once(train, valid, res)
        print('RMSE: ')
        print(rmse_loss(df_res['log_return'], df_res['log_return_forecast']))

# All following from https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/
def grangers_causation_matrix(data, variables, test='ssr_chi2test', maxlag=12):
    ''' Create grangers causation matrix '''
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose = False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

def cointegration_test(df, alpha=0.05): 
    ''' Perform the Johansen cointegration test '''
    out = coint_johansen(df, -1, 1)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length=6): return str(val).ljust(length)
    # Summary
    print('Johansen cointegration test')
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace, 2), 9), ">", adjust(cvt, 8), ' =>  ', trace > cvt)

def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue']

    def adjust(val, length=6): 
        return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")  


def get_durbin_watson(cols, model):
    """
    Check for Serial Correlation of Residuals (Errors)
    """
    out = durbin_watson(model.resid)
    for col, val in zip(cols, out):
        print(col, ':', round(val, 2))

def undiff_once(train, valid, forecast):
    """
    undiff once with the lag 2, very specific function, won't work properly in any other case
    """
    val = pd.DataFrame(valid, columns=train.columns)
    df_fc = pd.DataFrame(forecast.copy(), columns = train.columns)
    for col in val.columns:
        df_fc[str(col)+'_forecast'] = val[col].iloc[1] + df_fc[col].cumsum()
    return df_fc



    