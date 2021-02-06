import torch

class FinancialPhraseBankDataset(torch.utils.dataDataset):

  def __init__(self, headlines, sentiments, tokenizer, max_len):
    self.headlines = headlines.
    self.sentiments = sentiments
    self.tokenizer = tokenizer
    self.max_len = max_len
  
  def __len__(self):
    return len(self.headlines)
  
  def __getitem__(self, item):
    headline = str(self.headlines[item])
    sentiment = self.sentiments[item]

    encoding = self.tokenizer.encode_plus(
      headline,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return {
      'headline': headline,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'sentiment': sentiment
    }

def create_data_loader(df, tokenizer, max_len, batch_size):
  
  ds = FinancialPhraseBankDataset(
    headlines=df['headline'].to_numpy(),
    sentiments=df['sentiment'].to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )

  dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=4)
  
  return dl