import pandas as pd
import re
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier

from Data.lexicon.TXTfpblexical import predict_sentences

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

def load_aapl(path):
    test = pd.read_csv(path+'pre_processed_aapl_sentences.csv', index_col=None, engine='python')
    test.dropna(inplace=True)
    return test

def load_malo(path):
    data = pd.read_csv(path+'/Sentences_AllAgree_preprocessed_baseline.csv')
    return data

def lexicon_based(path):
    data = load_aapl(path)
    pred = predict_sentences(data, path)
    pred.to_csv(path_or_buf=path+'aapl_lex.csv')
    return pred

def clean_text(text):
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = text.replace('\n', ' ')
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    return text

def tokenize(txt):
    """Tokenize by whitespace."""
    return txt.split()

def tfidf_svm(path):
    train = load_malo(path)
    test = load_aapl(path)
    train['sentence'] = train['sentence'].apply(clean_text)
    codes = {'neutral':0, 'positive':1, 'negative':-1}
    train['label'] = train['label'].map(codes)
    X_train, X_test, y_train, y_test = train_test_split(train['sentence'], train['label'], test_size=0.2, random_state=42)

    piper = Pipeline([("vect", CountVectorizer(tokenizer = tokenize, ngram_range=(1,2),min_df = 0.0, max_df=0.85)),
                  ("tfidf", TfidfTransformer(norm="l2",sublinear_tf=True,use_idf= True)),
                  ("clf", SGDClassifier(shuffle = True,n_iter_no_change = 80, random_state = 123,alpha = 0.0001,loss="log",penalty="l1"))])
    piper.fit(X_train,y_train)
    res_p = piper.predict_proba(test['text'])
    res_p = pd.DataFrame(res_p)
    res = piper.predict(test['text'])
    res = pd.get_dummies(res)
    f_res = pd.concat([test, res_p, res], axis=1)
    f_res.columns = ['idx','article_time', 'text', 'neg_prob','neut_prob','pos_prob', 'neg','neut','pos']
    f_res.to_csv(path_or_buf=path+'aapl_svm.csv')
    return f_res