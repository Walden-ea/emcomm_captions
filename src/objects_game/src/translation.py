import torch
import torch.nn as nn


# ==================== RNN Models ====================
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers=2, pad_id=None, dropout=0.0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)

    def forward(self, src):
        # src: [B, T]
        embedded = self.emb(src)  # [B, T, emb_dim]
        outputs, (h, c) = self.rnn(embedded)  # outputs ignored for vanilla seq2seq
        return h, c  # [num_layers, B, hid_dim]


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers=2, pad_id=None, dropout=0.0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hid_dim, vocab_size)

    def forward(self, tgt, h, c):
        # tgt: [B, T] (with <sos> prepended)
        embedded = self.emb(tgt)  # [B, T, emb_dim]
        outputs, (h, c) = self.rnn(embedded, (h, c))
        logits = self.fc(outputs)  # [B, T, vocab_size]
        return logits, h, c


# ==================== Transformer Models ====================
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers=2, pad_id=None, dropout=0.0, num_heads=8):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.pos_enc = nn.Parameter(self._positional_encoding(5000, emb_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=hid_dim,
            dropout=dropout,
            batch_first=True,
            activation="relu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pad_id = pad_id

    def _positional_encoding(self, max_len, d_model):
        """Generate positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, src):
        # src: [B, T]
        embedded = self.emb(src)  # [B, T, emb_dim]
        
        # Add positional encoding
        seq_len = embedded.size(1)
        embedded = embedded + self.pos_enc[:seq_len, :].to(embedded.device)
        
        # Create padding mask
        src_key_padding_mask = (src == self.pad_id) if self.pad_id is not None else None
        
        # Pass through transformer
        output = self.transformer(embedded, src_key_padding_mask=src_key_padding_mask)  # [B, T, emb_dim]
        
        # Return output sequence (compatible with decoder input)
        # Return dummy h, c for compatibility with RNN decoder interface
        return output, None


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers=2, pad_id=None, dropout=0.0, num_heads=8):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.pos_enc = nn.Parameter(self._positional_encoding(5000, emb_dim))
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=hid_dim,
            dropout=dropout,
            batch_first=True,
            activation="relu"
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(emb_dim, vocab_size)
        self.pad_id = pad_id

    def _positional_encoding(self, max_len, d_model):
        """Generate positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, tgt, encoder_output, h=None, c=None):
        # tgt: [B, T]
        # encoder_output: [B, S, emb_dim]
        embedded = self.emb(tgt)  # [B, T, emb_dim]
        
        # Add positional encoding
        seq_len = embedded.size(1)
        embedded = embedded + self.pos_enc[:seq_len, :].to(embedded.device)
        
        # Create causal mask for autoregressive decoding
        tgt_len = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(tgt.device)
        
        # Create padding masks
        tgt_key_padding_mask = (tgt == self.pad_id) if self.pad_id is not None else None
        memory_key_padding_mask = (torch.zeros(encoder_output.size(0), encoder_output.size(1), dtype=torch.bool).to(tgt.device))
        
        # Pass through transformer decoder
        output = self.transformer(
            embedded,
            encoder_output,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )  # [B, T, emb_dim]
        
        logits = self.fc(output)  # [B, T, vocab_size]
        return logits, None, None
