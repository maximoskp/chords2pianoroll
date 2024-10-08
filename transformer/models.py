from transformers import RobertaModel, RobertaTokenizerFast, BartForConditionalGeneration, BartConfig
from miditok import REMI, TokenizerConfig
from pathlib import Path
import torch.nn as nn

'''
# text
text_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
text_encoder = RobertaModel.from_pretrained('roberta-base')

# midi
remi_tokenizer = REMI(params=Path('/media/datadisk/data/pretrained_models/midis_REMI_BPE_tokenizer.json'))
roberta_tokenizer_midi = RobertaTokenizerFast.from_pretrained('/media/datadisk/data/pretrained_models/midi_mlm_tiny/midi_wordlevel_tokenizer')
midi_model = RobertaModel.from_pretrained('/media/datadisk/data/pretrained_models/midi_mlm_tiny/checkpoint-5120')

# chroma
roberta_tokenizer_chroma = RobertaTokenizerFast.from_pretrained('/media/datadisk/data/pretrained_models/chroma_mlm_tiny/chroma_wordlevel_tokenizer')
chroma_model = RobertaModel.from_pretrained('/media/datadisk/data/pretrained_models/chroma_mlm_tiny/checkpoint-14336')

bart_config = BartConfig(
    vocab_size=roberta_tokenizer_midi.vocab_size,
    pad_token_id=roberta_tokenizer_midi.pad_token_id,
    bos_token_id=roberta_tokenizer_midi.bos_token_id,
    eos_token_id=roberta_tokenizer_midi.eos_token_id,
    decoder_start_token_id=roberta_tokenizer_midi.bos_token_id,
    forced_eos_token_id=roberta_tokenizer_midi.eos_token_id,
    max_position_embeddings=4096,
    encoder_layers=4,
    encoder_attention_heads=4,
    encoder_ffn_dim=512,
    decoder_layers=4,
    decoder_attention_heads=4,
    decoder_ffn_dim=512,
    d_model=512,
    encoder_layerdrop=0.2,
    decoder_layerdrop=0.2,
    dropout=0.2
)
model = BartForConditionalGeneration(bart_config)

'''

class MelCAT_base(nn.Module):
    def __init__(self, bart_config):
        super(MelCAT_base, self).__init__()
        self.text_encoder = RobertaModel.from_pretrained('roberta-base')
        self.midi_encoder = RobertaModel.from_pretrained('/media/datadisk/data/pretrained_models/midi_mlm_tiny/checkpoint-5120')
        self.chroma_encoder = RobertaModel.from_pretrained('/media/datadisk/data/pretrained_models/chroma_mlm_tiny/checkpoint-14336')
        self.bart_model = BartForConditionalGeneration(bart_config)
    # end init

    def forward(self, text_ids, melody_ids, chroma_ids):
        text_embeds = self.text_encoder( text_ids )
        midi_embeds = self.chroma_encoder( text_ids )
        chroma_embeds = self.midi_encoder( text_ids )
    # end forward
# end class