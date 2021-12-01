# Financial Sentiment Analysis
Academic project as part of the "Deep Learning" course at MVA.<br>
Goal was to get familiar with NLP methods & try different models up to the now popular Transformer architecture.<br>
3-class classification task.

# Dataset
I'm using the public dataset Financial PhraseBank (https://arxiv.org/abs/1307.5336).<br>
It consists of 4845 english sentences selected randomly from financial news, labeled according to how the information contained might affect the mentioned company stock price: positive, negative or neutral

# How to run code?
To run training and testing using different recurrent neural networks (RNNs, LSTMs, GRUs), modifiy the rnn config file and run:
```
python train_rnn.py --config rnn_config.json
```

To run training and testing using BERT model, modifiy the BERT config file and run:
```
python train_bert.py --config bert_config.json
```
