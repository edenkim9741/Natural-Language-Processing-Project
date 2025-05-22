import torch
import torch.nn as nn

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=8)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt_input, memory, tgt_mask=None, memory_mask=None):
        embedded = self.embedding(tgt_input)  # [B, T, D]
        output = self.decoder(embedded.transpose(0,1), memory.transpose(0,1), tgt_mask=tgt_mask, memory_mask=memory_mask)
        return self.fc_out(output.transpose(0,1))  # [B, T, vocab]
