import os
import json
import random
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def open_config_file(filepath):
    with open(filepath) as jsonfile:
        pdict = json.load(jsonfile)
        params = AttrDict(pdict)
    return params


def epoch_time(start_time, end_time):
  elapsed_time = end_time - start_time
  elapsed_mins = int(elapsed_time / 60)
  elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
  return elapsed_mins, elapsed_secs


def categorical_accuracy(preds, y):
    '''
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    '''
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum() / torch.FloatTensor([y.shape[0]])


def evaluate_rnn(model, val_loader, criterion):
    
  epoch_loss = 0
  epoch_acc = 0
  model.eval()
  
  with torch.no_grad():
  
    for batch in val_loader:

      headline, headline_lengths = batch.headline
      predictions = model(headline, headline_lengths.cpu()).squeeze(1)
      loss = criterion(predictions, batch.sentiment.long())
      acc = categorical_accuracy(predictions.cpu(), batch.sentiment.cpu())

      epoch_loss += loss.item()
      epoch_acc += acc.item()
    
  return epoch_loss / len(val_loader), epoch_acc / len(val_loader)


def train_rnn(model, train_loader, optimizer, criterion):
    
  epoch_loss = 0
  epoch_acc = 0
  model.train()
  
  for batch in train_loader:
      
    optimizer.zero_grad()
    headline, headline_lengths = batch.headline
    predictions = model(headline, headline_lengths.cpu()).squeeze(1)
    loss = criterion(predictions, batch.sentiment.long())
    acc = categorical_accuracy(predictions.cpu(), batch.sentiment.cpu())
    
    loss.backward()
    optimizer.step()
    
    epoch_loss += loss.item()
    epoch_acc += acc.item()
      
  return epoch_loss / len(train_loader), epoch_acc / len(train_loader)


def get_predictions_rnn(model, data_loader):
  
  model = model.eval()
  
  headline_texts = []
  predictions = []
  prediction_probs = []
  real_values = []

  with torch.no_grad():

    for batch in data_loader:

      headline, headline_lengths = batch.headline
      outputs = model(headline, headline_lengths.cpu()).squeeze(1)
      probs = F.softmax(outputs, dim=1)
      _, preds = torch.max(outputs, dim=1)
      
      predictions.extend(preds)
      prediction_probs.extend(probs)
      targets = batch.sentiment
      real_values.extend(targets)

  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  real_values = torch.stack(real_values).cpu()
  return predictions, prediction_probs, real_values


def show_confusion_matrix(confusion_matrix):
  
  plt.figure(figsize=(8,7))
  hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
  hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
  hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
  plt.ylabel('True sentiment')
  plt.xlabel('Predicted sentiment')


def train_bert(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
  
  model = model.train()
  losses = []
  correct_predictions = 0
  
  for d in data_loader:
    
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    sentiments = d["sentiment"]

    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

    _, preds = torch.max(outputs, dim=1)
    targets = torch.LongTensor([class_names[s] for s in sentiments]).to(device)
    loss = loss_fn(outputs, targets)

    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

  return correct_predictions.double() / n_examples, np.mean(losses)


def eval_bert(model, data_loader, loss_fn, device, n_examples):
  
  model = model.eval()
  losses = []
  correct_predictions = 0

  with torch.no_grad():
    for d in data_loader:
      
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      sentiments = d["sentiment"]

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      targets = torch.LongTensor([class_names[s] for s in sentiments]).to(device)
      loss = loss_fn(outputs, targets)

      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(losses)


def get_predictions_bert(model, data_loader):
  
  model = model.eval()
  
  headline_texts = []
  predictions = []
  prediction_probs = []
  real_values = []

  with torch.no_grad():

    for d in data_loader:

      headlines = d["headline"]
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      sentiments = d["sentiment"]

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      probs = F.softmax(outputs, dim=1)

      headline_texts.extend(headlines)
      predictions.extend(preds)
      prediction_probs.extend(probs)
      targets = torch.LongTensor([class_names[s] for s in sentiments]).to(device)
      real_values.extend(targets)

  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  real_values = torch.stack(real_values).cpu()
  return headline_texts, predictions, prediction_probs, real_values