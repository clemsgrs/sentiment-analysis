import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import os

parser = argparse.ArgumentParser(description='Sentiment analyzer')
parser.add_argument('--file_path', type=str, help='Path to the database file.')
args = parser.parse_args()

data_dir = 'data/'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

data = pd.read_csv(args.file_path, sep='.@', names=['headline','sentiment'])

train, test = train_test_split(data, test_size=0.2, random_state=0)
train, val = train_test_split(train, test_size=0.1, random_state=0)

train.to_csv(os.path.join(data_dir, 'train.csv'), index=False)
test.to_csv(os.path.join(data_dir, 'val.csv'), index=False)
val.to_csv(os.path.join(data_dir, 'test.csv'), index=False)