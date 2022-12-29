
import re
import unidecode
import numpy as np

import torch
import torch.nn as nn

class CharRNN(nn.Module):
    def __init__(self, tokens, n_hidden=256, n_layers=2,
                 drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        self.chars = tuple(set(tokens))
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        self.embed = nn.Embedding(len(self.chars), n_hidden)
        self.lstm = nn.LSTM(n_hidden, n_hidden, n_layers,
                            dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden, len(self.chars))

    def forward(self, x, hidden=None):

        if hidden is None:
            hidden = self.init_hidden(x.size(0))

        embed = self.embed(x)
        r_output, hidden = self.lstm(embed, hidden)
        out = self.dropout(r_output)
        out = out.contiguous().view(-1, self.n_hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        return hidden

# Load data
with open('data/covid19-faq.txt', encoding="utf8") as f:
    text = f.read()
textfile = unidecode.unidecode(text)
textfile = re.sub(" +", " ", textfile)

# Init model
model = CharRNN(textfile)
x = torch.tensor([model.char2int[ch] for ch in textfile[:100]]).view(1, -1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train model
for i in range(100):
    optimizer.zero_grad()
    out, hidden = model.forward(x, hidden=None if i == 0 else hidden)
    loss = criterion(out, x.view(-1))
    loss.backward()
    optimizer.step()
    print(loss.item())



