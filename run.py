''' Run your code from here. 
    pre_process_pb(path), pre_process_news(path), pre_process_vars(path), train_bert(path) need to be ran at least once
    but will all create files, such that you can work with the results and do not have to run everything again '''
from preprocessing import pre_process_news, pre_process_pb, pre_process_vars
from bert_model import train_bert
from prediction import process_res, predict_lstm, predict_var, eval_for_var


if __name__ == "__main__":
    path = 'C:/Users/Elias/Desktop/BERT/Data/'
    use_VAR = False

    pre_process_pb(path)
    pre_process_news(path)
    pre_process_vars(path)
    train_bert(path)
    if use_VAR:
        data = process_res(path, inc_lag=False)
        eval_for_var(data)
        predict_var(data)
    else:
        data = process_res(path, inc_lag=True)
        predict_lstm(data, pred_bin=True) # currently set to predict binary target
    