import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import sys
sys.path.insert(0, '..')
from transformer.positional_encoding import PositionalEncoding
from transformer.layers import EncoderLayer, DecoderLayer

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
    # end init

    def generate_mask(self, src, tgt, pad_token_idx=0):
        src_mask = (src != pad_token_idx).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != pad_token_idx).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask
    # end generate_mask

    def forward(self, src, tgt, pad_token_idx=0):
        src_mask, tgt_mask = self.generate_mask(src, tgt, pad_token_idx)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output
    # end forward
# end class Transformer

class EncoderModel(nn.Module):
    def __init__(self, src_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(EncoderModel, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
    # end init

    def generate_mask(self, src, pad_token_idx=0):
        src_mask = (src != pad_token_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
    # end generate_mask

    def forward(self, src, pad_token_idx=0):
        src_mask = self.generate_mask(src, pad_token_idx)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
        return enc_output
    # end forward
# end class

class DecoderModel(nn.Module):
    def __init__(self, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(DecoderModel, self).__init__()
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
    # end init

    def generate_mask(self, src, tgt, pad_token_idx=0):
        src_mask = (src != pad_token_idx).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != pad_token_idx).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask
    # end generate_mask

    def forward(self, enc_output, src, tgt, pad_token_idx=0):
        src_mask, tgt_mask = self.generate_mask(src, tgt, pad_token_idx)
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output
    # end forward
# end class

class TransformerFromModels(nn.Module):
    def __init__(self, enc_model, dec_model):
        super(TransformerFromModels, self).__init__()
        self.encoderModel = enc_model
        self.decoderModel = dec_model
    # end init

    def forward(self, src, tgt, pad_token_idx=0):
        encoder_output = self.encoderModel(src, pad_token_idx)
        decoder_output = self.decoderModel(encoder_output, src, tgt, pad_token_idx)
        return decoder_output
    # end forward
# end class TransformerFromModels

# MLM encoder wrapper based on the MLM component of BERT found here:
# https://neptune.ai/blog/how-to-code-bert-using-pytorch-tutorial
# It says that the output "decoding" part has the same weights
# as the input embedding.
# For training, check out this repository:
# https://github.com/lucidrains/mlm-pytorch/tree/master
class MLMEncoderWrapper(nn.Module):
    def __init__(self, encoderModel):
        super(MLMEncoderWrapper, self).__init__()
        self.encoder_model = encoderModel
        # decoder is shared with embedding layer
        n_vocab, n_dim = self.encoder_model.encoder_embedding.weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        # tie weights
        self.decoder.weight = self.encoder_model.encoder_embedding.weight
        # self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))
    # end init

    def forward(self, input_ids):
        output = self.encoder_model(input_ids)
        # adding bias again is not correct
        logits_vocab_size = self.decoder(output) # + self.decoder_bias # [batch_size, max_pred, n_vocab]
        return logits_vocab_size
    # end forward
# end class MLMEncoderWrapper