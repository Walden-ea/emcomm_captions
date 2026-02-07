import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers=2, pad_id=None):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, batch_first=True)

    def forward(self, src):
        # src: [B, T]
        embedded = self.emb(src)  # [B, T, emb_dim]
        outputs, (h, c) = self.rnn(embedded)  # outputs ignored for vanilla seq2seq
        return h, c  # [num_layers, B, hid_dim]


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers=2, pad_id=None):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hid_dim, vocab_size)

    def forward(self, tgt, h, c):
        # tgt: [B, T] (with <sos> prepended)
        embedded = self.emb(tgt)  # [B, T, emb_dim]
        outputs, (h, c) = self.rnn(embedded, (h, c))
        logits = self.fc(outputs)  # [B, T, vocab_size]
        return logits, h, c
