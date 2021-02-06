import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data
from textwrap import wrap
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import random
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from utils import *

history = defaultdict(list)
best_accuracy = 0

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default="bert_default.json", metavar='N', help='config file')
args = parser.parse_args()
params = open_config_file(args.config)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

class_names = {'neutral':0, 'positive':1, 'negative':2}
model_desc = f'{params.model}_pretrained={params.pretrained}_nlayer={params.num_layers}'

# Load data
df_train = pd.read_csv('train.csv')
df_val = pd.read_csv('val.csv')
df_test = pd.read_csv('test.csv')

print(f'df_train: {df_train.shape}')
print(f'df_val: {df_val.shape}')
print(f'df_test: {df_test.shape}')

tokenizer = BertTokenizer.from_pretrained(p.pre_trained_model_name)

train_loader = create_data_loader(df_train, tokenizer, params.max_len, params.train_batch_size)
val_loader = create_data_loader(df_val, tokenizer, params.max_len, params.val_batch_size)
test_loader = create_data_loader(df_test, tokenizer, params.max_len, params.test_batch_size)

# Model
model = create_model(params)
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_loader) * params.nepochs
scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss().to(device)


for epoch in range(params.nepochs):

  print(f'Epoch {epoch + 1}/{params.nepochs}')
  print('-' * 10)

  train_acc, train_loss = train_epoch(
    model,
    train_loader,    
    loss_fn, 
    optimizer, 
    device, 
    scheduler, 
    len(df_train)
  )

  print(f'Train loss {train_loss} accuracy {train_acc}')

  val_acc, val_loss = eval_model(
    model,
    val_loader,
    loss_fn, 
    device, 
    len(df_val)
  )

  print(f'Val loss {val_loss} accuracy {val_acc}')
  print()

  history['train_acc'].append(train_acc)
  history['train_loss'].append(train_loss)
  history['val_acc'].append(val_acc)
  history['val_loss'].append(val_loss)

  if val_acc > best_accuracy:
    torch.save(model.state_dict(), 'best_model_state.pth')
    best_accuracy = val_acc

model.load_state_dict(torch.load('best_model_state.pth'))

test_acc, _ = eval_model(
  model,
  test_loader,
  loss_fn,
  device,
  len(df_test)
)

print(f'test accuracy: {test_acc.item()}')

y_headlines_texts, y_pred, y_pred_probs, y_test = get_predictions(model, test_data_loader)
print(classification_report(y_test, y_pred, target_names=class_names))
cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
show_confusion_matrix(df_cm)
plt.tight_layout()
plt.savefig(os.path.join(params.result_dir, f'{model_desc}_cm.pdf'))