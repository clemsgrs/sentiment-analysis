import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data
from torchtext import datasets
import random
import argparse
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default="default.json", metavar='N', help='config file')
args = parser.parse_args()
params = open_config_file(args.config)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

class_names = {'neutral':0, 'positive':1, 'negative':2}
model_desc = f'{params.model}_bidir={params.bidirectional}_nlayer={params.num_layers}'

headline = data.Field(tokenize='spacy', include_lengths=True)
sentiment = data.LabelField(dtype=torch.float)

f = {'headline': ('headline', headline), 'sentiment': ('sentiment', sentiment)}

train_dataset = data.TabularDataset(path=os.path.join(params.data_dir, 'train.csv'), format='csv', fields=f)
val_dataset = data.TabularDataset(path=os.path.join(params.data_dir, 'val.csv'), format='csv', fields=f)
test_dataset = data.TabularDataset(path=os.path.join(params.data_dir, 'test.csv'), format='csv', fields=f)

print(f'Number of training examples: {len(train_dataset)}')
print(f'Number of validation examples: {len(val_dataset)}')
print(f'Number of testing examples: {len(test_dataset)}')

# Load GloVe embeddings
print(f'Loading GloVe embeddings...')
headline.build_vocab(train_dataset, vectors='glove.6B.100d', unk_init=torch.Tensor.normal_)
print(f'GloVe embeddings loaded!')
sentiment.build_vocab(train_dataset)

print(f'Unique tokens in headline vocabulary: {len(headline.vocab)}')
print(f'Unique tokens in sentiment vocabulary: {len(sentiment.vocab)}')

# Create DataLoaders
train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train_dataset, val_dataset, test_dataset), 
    batch_sizes=(params.train_batch_size, params.val_batch_size, params.test_batch_size),
    sort_key=lambda x: len(x.headline), sort_within_batch=True, device=device)

# Instantiale model
vocab_size = len(headline.vocab)
pad_idx = headline.vocab.stoi[headline.pad_token]
unk_idx = headline.vocab.stoi[headline.unk_token]

model = create_model(params, vocab_size, pad_idx)

# Each word (i.e each index between 1 and len(vocab)) is represented by a 100-dim distributed vector
pretrained_embeddings = headline.vocab.vectors
# Load pre-trained embeddings
model.embedding.weight.data.copy_(pretrained_embeddings)
model.embedding.weight.data[unk_idx] = torch.zeros(params.embedding_dim)
model.embedding.weight.data[pad_idx] = torch.zeros(params.embedding_dim)

# Optimizer & Loss
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)
model = model.to(device)

# TRAINING
best_valid_loss = float('inf')
for epoch in range(params.nepochs):

  start_time = time.time()
  
  train_loss, train_acc = train(model, train_iter, optimizer, criterion)
  valid_loss, valid_acc = evaluate(model, val_iter, criterion)
  
  end_time = time.time()
  epoch_mins, epoch_secs = epoch_time(start_time, end_time)
  
  if valid_loss < best_valid_loss:
    best_valid_loss = valid_loss
    torch.save(model.state_dict(), os.path.join(params.checkpoint_dir, f'best_{model_desc}.pt'))

  print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
  print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
  print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

# TESTING
model.load_state_dict(torch.load(os.path.join(params.checkpoint_dir, f'best_{model_desc}.pt')))
test_loss, test_acc = evaluate(model, test_iter, criterion)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

y_pred, y_pred_probs, y_test = get_predictions(model, test_iter)
print(classification_report(y_test, y_pred, target_names=class_names))

cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
show_confusion_matrix(df_cm)
plt.tight_layout()
plt.savefig(os.path.join(params.result_dir, f'{model_desc}_cm.pdf'))