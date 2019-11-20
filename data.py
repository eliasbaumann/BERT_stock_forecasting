import pandas as pd

def pre_process_pb(path):
    df = pd.read_csv(path+'FinancialPhraseBank-v1.0/Sentences_66Agree.txt',sep='\@',engine='python',header=None,names=['sentence','label'])
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    df.to_csv(path+'Sentences_AllAgree_preprocessed_baseline.csv',index=False)
    df = pd.get_dummies(df, columns=['label'])
    df.to_csv(path+'Sentences_AllAgree_preprocessed.csv',index=False)
    print(df.head())

def pre_process_aapl(path):
    print(path+'all_upto_2019-08-29.csv')
    df = pd.read_csv(path+'all_upto_2019-08-29.csv', sep=',', engine='python',) # TODO how? What are the correct import params?
    print(df.head())


