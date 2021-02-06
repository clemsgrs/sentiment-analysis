import torch
import torch.nn as nn

def create_model(p, vocab_size, pad_idx):

    model = None
    print(p.model)
    
    if p.model == 'rnn':
        model = RNN(p, vocab_size, pad_idx)
    elif p.model == 'lstm':
        model = LSTM(p, vocab_size, pad_idx)
    elif p.model == 'gru':
        model = GRU(p, vocab_size, pad_idx)
    elif p.model == 'bert':
        model = BERT(p, vocab_size, pad_idx)
    else:
        raise ValueError(f'Model {p.model} not recognized')

    print(f'model {model.name()} was created')
    
    return model

class RNN(nn.Module):
  def __init__(self, p, vocab_size, pad_idx):
    
    super().__init__()
    self.p = p
    self.bidirectional = p.bidirectional
    self.embedding = nn.Embedding(vocab_size, p.embedding_dim, padding_idx=pad_idx)
    
    self.rnn = nn.RNN(p.embedding_dim, 
                      p.hidden_dim, 
                      num_layers=p.num_layers, 
                      bidirectional=p.bidirectional, 
                      dropout=p.dropout_p)
    
    fc_input_dim = 2*p.hidden_dim if self.p.bidirectional else p.hidden_dim
    self.fc = nn.Linear(fc_input_dim, p.num_classes)
    self.dropout = nn.Dropout(p.dropout_p)
    
  def forward(self, headline, headline_lengths):
    
    # headline = [sent len, batch size]
    
    embedded = self.dropout(self.embedding(headline))
    
    #embedded = [sent len, batch size, emb dim]
    
    # pack sequence
    packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, headline_lengths)
    packed_output, hidden = self.rnn(packed_embedded)
    
    # unpack sequence
    output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

    # output = [sent len, batch size, hid dim * num directions]
    # output over padding tokens are zero tensors
    
    # hidden = [num layers * num directions, batch size, hid dim]
    # cell = [num layers * num directions, batch size, hid dim]
    
    # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
    # and apply dropout
    
    if self.bidirectional:
      hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
    else:
      hidden = self.dropout(hidden[-1,:,:])
            
    # hidden = [batch size, hid dim * num directions]
        
    return self.fc(hidden)


class LSTM(nn.Module):
  def __init__(self, p, vocab_size, pad_idx):
    
    super().__init__()
    self.p = p
    self.bidirectional = p.bidirectional
    self.embedding = nn.Embedding(vocab_size, p.embedding_dim, padding_idx=pad_idx)
    
    self.rnn = nn.LSTM(p.embedding_dim, 
                      p.hidden_dim, 
                      num_layers=p.num_layers, 
                      bidirectional=p.bidirectional, 
                      dropout=p.dropout_p)
    
    fc_input_dim = 2*p.hidden_dim if self.p.bidirectional else p.hidden_dim
    self.fc = nn.Linear(fc_input_dim, p.num_classes)
    self.dropout = nn.Dropout(p.dropout_p)
    
  def forward(self, headline, headline_lengths):
    
    # headline = [sent len, batch size]
    
    embedded = self.dropout(self.embedding(headline))
    
    #embedded = [sent len, batch size, emb dim]
    
    # pack sequence
    packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, headline_lengths)
    packed_output, (hidden, cell) = self.rnn(packed_embedded)
    
    # unpack sequence
    output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

    # output = [sent len, batch size, hid dim * num directions]
    # output over padding tokens are zero tensors
    
    # hidden = [num layers * num directions, batch size, hid dim]
    # cell = [num layers * num directions, batch size, hid dim]
    
    # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
    # and apply dropout
    
    if self.bidirectional:
      hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
    else:
      hidden = self.dropout(hidden[-1,:,:])
            
    # hidden = [batch size, hid dim * num directions]
        
    return self.fc(hidden)


class GRU(nn.Module):
  def __init__(self, p, vocab_size, pad_idx):
    
    super().__init__()
    self.p = p
    self.bidirectional = p.bidirectional
    self.embedding = nn.Embedding(vocab_size, p.embedding_dim, padding_idx=pad_idx)
    
    self.rnn = nn.GRU(p.embedding_dim, 
                      p.hidden_dim, 
                      num_layers=p.num_layers, 
                      bidirectional=p.bidirectional, 
                      dropout=p.dropout_p)
    
    fc_input_dim = 2*p.hidden_dim if self.p.bidirectional else p.hidden_dim
    self.fc = nn.Linear(fc_input_dim, p.num_classes)
    self.dropout = nn.Dropout(p.dropout_p)
    
  def forward(self, headline, headline_lengths):
    
    # headline = [sent len, batch size]
    
    embedded = self.dropout(self.embedding(headline))
    
    #embedded = [sent len, batch size, emb dim]
    
    # pack sequence
    packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, headline_lengths)
    packed_output, hidden = self.rnn(packed_embedded)
    
    # unpack sequence
    output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

    # output = [sent len, batch size, hid dim * num directions]
    # output over padding tokens are zero tensors
    
    # hidden = [num layers * num directions, batch size, hid dim]
    # cell = [num layers * num directions, batch size, hid dim]
    
    # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
    # and apply dropout
    
    if self.bidirectional:
      hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
    else:
      hidden = self.dropout(hidden[-1,:,:])
            
    # hidden = [batch size, hid dim * num directions]
        
    return self.fc(hidden)