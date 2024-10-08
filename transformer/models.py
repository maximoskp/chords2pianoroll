from transformers import RobertaModel, RobertaTokenizerFast, BartForConditionalGeneration, BartConfig
from miditok import REMI, TokenizerConfig
from pathlib import Path
import torch.nn as nn
import torch

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
        self.text_lstm = nn.LSTM(input_size=768, 
                            hidden_size=256, 
                            batch_first=True)
        self.midi_encoder = RobertaModel.from_pretrained('/media/datadisk/data/pretrained_models/midi_mlm_tiny/checkpoint-5120')
        self.chroma_encoder = RobertaModel.from_pretrained('/media/datadisk/data/pretrained_models/chroma_mlm_tiny/checkpoint-14336')
        self.bart_model = BartForConditionalGeneration(bart_config)
        print('initialized')
    # end init

    def forward(self, text, melody, chroma): # TODO: add optional accomp input (for continuing composition) and labels (for loss calculation)
        print('in forward')
        text_embeds = self.text_encoder( input_ids=text['input_ids'], attention_mask=text['attention_mask'], output_hidden_states=True )
        text_lstm_output, (_,_) = self.text_lstm(text_embeds.last_hidden_state)
        melody_embeds = self.midi_encoder( input_ids=melody['input_ids'], attention_mask=melody['attention_mask'], output_hidden_states=True )
        chroma_embeds = self.chroma_encoder( input_ids=chroma['input_ids'], attention_mask=chroma['attention_mask'], output_hidden_states=True )
        print(text_embeds.last_hidden_state.shape)
        print(text_lstm_output.shape)
        print(melody_embeds.last_hidden_state.shape)
        print(chroma_embeds.last_hidden_state.shape)
        bart_encoder_input = torch.cat( (text_lstm_output[:,-1:,:], melody_embeds.last_hidden_state, chroma_embeds.last_hidden_state), 1 )
        print(bart_encoder_input.shape)
        encoder_outputs = self.bart_model.model.encoder( inputs_embeds=bart_encoder_input )
        print(encoder_outputs.last_hidden_state.shape)
        decoder_outputs = self.bart_model.model.decoder(
            # inputs_embeds=decoder_input_embeds,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            # encoder_attention_mask=attention_mask,
            # attention_mask=decoder_attention_mask,
            # **kwargs
        )
        return decoder_outputs
    # end forward
# end class