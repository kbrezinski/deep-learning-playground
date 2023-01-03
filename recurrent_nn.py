
import torch
import torch.nn as nn
import pandas as pd
from torchtext.legacy import data
import spacy

df = pd.read_csv('data/imdb-dataset.csv')

# Create the training and test sets
df.replace({'sentiment': {'positive': 1, 'negative': 0}}, inplace=True)
random_idx = torch.randperm(len(df))
train_idx = random_idx[:int(len(df)*.8)]
test_idx = random_idx[int(len(df)*.8):]

train_df = df.iloc[train_idx]
test_df = df.iloc[test_idx]

def create_dataset(df):
    sentences = df['review'].tolist()
    labels = df['sentiment'].tolist()

    # Create a torchtext Field to process the text
    text_field = data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm')
    label_field = data.Field(sequential=False, use_vocab=False, dtype=torch.long)
    fields = [('text', text_field), ('label', label_field)]

    # Create a torchtext dataset from the sentences and labels
    examples = [data.Example.fromlist([sentence, label], fields=fields) \
                     for sentence, label in zip(sentences, labels)]
    dataset = data.Dataset(examples=examples, fields=fields)

    # Build a vocabulary using the sentences
    text_field.build_vocab(dataset, max_size=10_000, min_freq=2)
    # Use the `BucketIterator` to batch the dataset
    iterator = data.BucketIterator(dataset, batch_size=256, train=False, sort_within_batch=False, sort_key=lambda x: len(x.text))

    return text_field, label_field, dataset, iterator
   

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        input = self.embedding(input)
        # input = [sentence length, batch size, embedding dim]

        output, (hidden, cell) = self.rnn(input)
        # output = [sentence length, batch size, hidden size]
        # hidden = [1, batch size, hidden size]
        # cell = [1, batch size, hidden size]

        output = self.fc(hidden.squeeze(0))
        return output

text_field, label_field, dataset, dataloader = create_dataset(test_df)

layer = RNN(len(text_field.vocab), hidden_size=128, output_size=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(layer.parameters(), lr=0.001)

for i in range(15):
    epoch_loss = 0
    for batch in dataloader:    
        optimizer.zero_grad()
        logits = layer(batch.text)
        loss = criterion(logits, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if not i % 10:
        print(f'Epoch {i} loss: {epoch_loss / len(dataloader):.4f}')

