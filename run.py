from preprocessing import pre_process_news, pre_process_pb, pre_process_vars
from bert_model import train_bert
from prediction import process_res


if __name__ == "__main__":
    path = 'C:/Users/Elias/Desktop/BERT/Data/'
    # pre_process_pb(path)
    # pre_process_news(path)
    # pre_process_vars(path)
    # train_bert(path)
    #res1 =  tfidf_svm(path)
    process_res(path)