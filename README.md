# Financial forecasting using a BERT based sentiment index
This work is based on a Master Thesis by [Manuel Tonneau](https://github.com/mananeau) which in turn is based on:

Hiew, Joshua & Huang, Xin & Mou, Hao & Li, Duan & Wu, Qi & Xu, Yabo. (2019). [BERT-based Financial Sentiment Index and LSTM-based Stock Return Predictability](https://arxiv.org/pdf/1906.09024.pdf). 

You may use this repository as a foundation to develop alternative approaches and additional functionalities. To run this code, execute run.py or edit the file to configure it to your liking. I recommend using a code editor such as Visual Studio Code with the python plugin or any other python capable editor to work on this project. The application also requires a Dataset which is not public and will have to be requested if you work on this topic as your master thesis.

### Requirements
The following package versions have been used to develop this work. Newer versions will likely work as well, older versions might not.
```
Python 3.6.8

tensorflow-gpu==2.0.0 (you may use regular tensorflow, tf 2.1 should be fine as well)
pandas==0.24.2
nltk==3.4.5
scikit-learn==0.21.2
statsmodels==0.10.2
pytorch-pretrained-bert==0.6.2
torch==1.3.1
fastai==1.0.59
beautifulsoup4==4.8.1
```

### Acknowledgements
#### Data:
Malo, Pekka & Sinha, Ankur & Takala, Pyry & Korhonen, Pekka & Wallenius, Jyrki. (2013). [FinancialPhraseBank-v1.0](https://www.researchgate.net/publication/251231364_FinancialPhraseBank-v10). 

http://iborate.com/usd-libor/ (USD libor)

https://mba.tuck.dartmouth.edu/ (Fama/French 5 Factors)

https://finance.yahoo.com/ (AAPL and USDX)

#### Code: 
Elisabeth Bommes (2016)
https://github.com/QuantLet/TXT/tree/master/TXTfpblexical 

https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/



