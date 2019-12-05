from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.modeling import BertConfig, BertForSequenceClassification, BertForNextSentencePrediction, BertForMaskedLM
from sklearn.model_selection import train_test_split
from functools import partial
import fastai
import fastai.text
import pandas as pd
import torch
#from fastai.text import BaseTokenizer

class FastAiBertTokenizer(fastai.text.BaseTokenizer):
    """Wrapper around BertTokenizer to be compatible with fast.ai"""
    def __init__(self, tokenizer: BertTokenizer, max_seq_len: int=128, **kwargs):
        self._pretrained_tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, *args, **kwargs):
        return self

    def tokenizer(self, t:str):
        """Limits the maximum sequence length"""
        return ["[CLS]"] + self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2] + ["[SEP]"]

def train_bert(path):
    train = pd.read_csv(path+'Sentences_AllAgree_preprocessed.csv')
    test = pd.read_csv(path+'pre_processed_aapl_sentences.csv', index_col=None, engine='python')
    test.dropna(inplace=True)
    test2 = pd.DataFrame(test['text'])
    test2.columns = ['sentence']
    train_1, val = train_test_split(train, shuffle=True, test_size=0.2, random_state=42)

    bert_tok = BertTokenizer.from_pretrained("bert-base-uncased",)
    fastai_bert_vocab = fastai.text.Vocab(list(bert_tok.vocab.keys()))
    fastai_tokenizer = fastai.text.Tokenizer(tok_func=FastAiBertTokenizer(bert_tok, max_seq_len=256), pre_rules=[], post_rules=[])
    label_cols = ["label_negative","label_neutral","label_positive"]

    databunch_1 = fastai.text.TextDataBunch.from_df(".", train_1, val,
                                                    test_df=test2,
                                                    tokenizer=fastai_tokenizer,
                                                    vocab=fastai_bert_vocab,
                                                    include_bos=False,
                                                    include_eos=False,
                                                    text_cols="sentence",
                                                    label_cols=label_cols,
                                                    bs=32,
                                                    collate_fn=partial(fastai.text.pad_collate, pad_first=False, pad_idx=0),
                                                    )
    bert_model_class = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    learner = fastai.text.Learner(databunch_1,
                                  bert_model_class,
                                  loss_func=torch.nn.BCEWithLogitsLoss(),
                                  model_dir=path+'temp/model',
                                  metrics=partial(fastai.text.accuracy_thresh, thresh=.25)
                                  )
    def bert_class_split(model):
        embedder = model.bert.embeddings
        pooler = model.bert.pooler
        encoder = model.bert.encoder
        classifier = [model.dropout, model.classifier]
        n = len(encoder.layer) // 3
        return [[embedder], list(encoder.layer[:n]), list(encoder.layer[n+1:2*n]), list(encoder.layer[(2*n)+1:]), [pooler], classifier]
    x = bert_class_split(bert_model_class)
    learner.split([x[0], x[1], x[2], x[3], x[5]])
    learner.fit_one_cycle(2, slice(5e-6, 5e-5), moms=(0.8,0.7), pct_start=0.2, wd =(1e-7, 1e-5, 1e-4, 1e-3,1e-1))
    preds = learner.get_preds(ds_type=fastai.basic_data.DatasetType.Test)
    test_res = pd.DataFrame(preds[0].tolist())
    test = pd.concat([test,test_res],axis=1) 
    test = test[['article_time','text',0,1,2]]
    tmp = pd.get_dummies(test[[0,1,2]].idxmax(axis=1))
    tmp.columns = ['neg','neut','pos']
    test.columns = ['article_time','text', 'neg_prob','neut_prob','pos_prob']
    test = pd.concat([test,tmp],axis=1)
    test.to_csv(path+'aapl_bert.csv')
    return test
    

