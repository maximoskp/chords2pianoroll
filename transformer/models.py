from transformers import RobertaModel, RobertaTokenizerFast, BartForConditionalGeneration, BartConfig
from miditok import REMI, TokenizerConfig
from pathlib import Path
import torch.nn as nn
import torch
import torch.nn.functional as F

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

CHROMA_PAD = 550

models_folder = '/media/maindisk/maximos/data/pretrained_models/'
# models_folder = '/media/datadisk/data/pretrained_models/'

class MelCAT_base(nn.Module):
    def __init__(self, bart_config, gpu=None):
        super(MelCAT_base, self).__init__()
        if gpu is None:
            self.dev = torch.device("cpu")
        else:
            self.dev = torch.device("cuda:"+str(gpu) if torch.cuda.is_available() else "cpu")
        self.to(self.dev)
        self.text_encoder = RobertaModel.from_pretrained('roberta-base').to(self.dev)
        self.text_lstm = nn.LSTM(input_size=768, 
                            hidden_size=256, 
                            batch_first=True).to(self.dev)
        # self.midi_encoder = RobertaModel.from_pretrained('/media/datadisk/data/pretrained_models/pop_midi_mlm_base/checkpoint-28680').to(self.dev)
        self.midi_encoder = RobertaModel.from_pretrained(models_folder+'pop_midi_mlm_base/checkpoint-28680').to(self.dev)
        self.midi_rescaler = nn.Linear(512, 256).to(self.dev)
        # self.chroma_encoder = RobertaModel.from_pretrained('/media/datadisk/data/pretrained_models/chroma_mlm_tiny/checkpoint-14336').to(self.dev)
        self.chroma_encoder = RobertaModel.from_pretrained(models_folder+'chroma_mlm_tiny/checkpoint-14336').to(self.dev)
        self.bart_model = BartForConditionalGeneration(bart_config).to(self.dev)
        # print('initialized')
    # end init

    def forward(self, text, melody, chroma, accomp): # TODO: add optional accomp input (for continuing composition) and labels (for loss calculation)
        # print('in forward')
        text_embeds = self.text_encoder( input_ids=text['input_ids'].to(self.dev), attention_mask=text['attention_mask'].to(self.dev), output_hidden_states=True )
        text_lstm_output, (_,_) = self.text_lstm(text_embeds.last_hidden_state)
        # print('ids:', melody['input_ids'])
        # print('att:', melody['attention_mask'])
        melody_embeds = self.midi_encoder( input_ids=melody['input_ids'].to(self.dev), attention_mask=melody['attention_mask'].to(self.dev), output_hidden_states=True )
        melody_embeds = self.midi_rescaler(melody_embeds.last_hidden_state)
        chroma_embeds = self.chroma_encoder( input_ids=chroma['input_ids'].to(self.dev), attention_mask=chroma['attention_mask'].to(self.dev), output_hidden_states=True )
        chroma_embeds = chroma_embeds.last_hidden_state
        # pad chroma if needed - leave CHROMA_PAD time steps to be sure
        if CHROMA_PAD-chroma_embeds.shape[1] > 0:
            chroma_embeds_shape = chroma_embeds.shape[1]
            chroma_embeds = F.pad( chroma_embeds , ( 0,0,0, CHROMA_PAD-chroma_embeds_shape ), 'constant', -1 )
            chroma_attention_mask = F.pad( chroma['attention_mask'], ( 0,CHROMA_PAD-chroma_embeds_shape ), 'constant', 0 )
        # print(text_embeds.last_hidden_state.shape)
        # print(text_lstm_output.shape)
        # print(melody_embeds.shape)
        # print(chroma_embeds.shape)
        # print(chroma_attention_mask.shape)
        # print(test_embeds.shape)
        # print(accomp['input_ids'].shape)
        bart_encoder_input = torch.cat( (text_lstm_output[:,-1:,:], chroma_embeds, melody_embeds), 1 )
        # bart_encoder_input = torch.cat( (text_lstm_output[:,-1:,:], melody_embeds.last_hidden_state, chroma_embeds.last_hidden_state), 1 )
        # make masks
        # bart_encoder_mask = torch.cat( (torch.full( (text_lstm_output[:,-1:,:].shape[0], 1), self.bart_model.config.pad_token_id ), \
        #                                 chroma['attention_mask'], melody['attention_mask'] ), 1 ).to(self.dev)
        bart_encoder_mask = torch.cat( (torch.full( (text_lstm_output[:,-1:,:].shape[0], 1), self.bart_model.config.pad_token_id ), \
                                        chroma_attention_mask, melody['attention_mask'] ), 1 ).to(self.dev)
        # print(bart_encoder_input.shape)
        encoder_outputs = self.bart_model.model.encoder( inputs_embeds=bart_encoder_input[:,:self.bart_model.config.max_position_embeddings,:], attention_mask=bart_encoder_mask[:,:self.bart_model.config.max_position_embeddings] )
        # print(encoder_outputs.last_hidden_state.shape)
        decoder_outputs = self.bart_model.model.decoder(
            # inputs_embeds=decoder_input_embeds,
            # input_ids=torch.full( (text_embeds.last_hidden_state.shape[0], 1), self.bart_model.config.eos_token_id ),
            input_ids=accomp['input_ids'][:,:self.bart_model.config.max_position_embeddings].to(self.dev),
            attention_mask=accomp['attention_mask'][:,:self.bart_model.config.max_position_embeddings].to(self.dev),
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            # encoder_attention_mask=attention_mask,
            # attention_mask=decoder_attention_mask,
            return_dict=True
            # **kwargs
        )
        vocab_output = self.bart_model.lm_head(decoder_outputs.last_hidden_state)
        return vocab_output
    # end forward
# end class

class MelCAT_base_tokens(nn.Module):
    def __init__(self, bart_config, gpu=None):
        super(MelCAT_base_tokens, self).__init__()
        if gpu is None:
            self.dev = torch.device("cpu")
        else:
            self.dev = torch.device("cuda:"+str(gpu) if torch.cuda.is_available() else "cpu")
        self.to(self.dev)
        # self.text_encoder = RobertaModel.from_pretrained('roberta-base').to(self.dev)
        # self.text_lstm = nn.LSTM(input_size=768, 
        #                     hidden_size=256, 
        #                     batch_first=True).to(self.dev)
        # self.midi_encoder = RobertaModel.from_pretrained('/media/datadisk/data/pretrained_models/midi_mlm_tiny/checkpoint-46080').to(self.dev)
        # self.chroma_encoder = RobertaModel.from_pretrained('/media/datadisk/data/pretrained_models/chroma_mlm_tiny/checkpoint-14336').to(self.dev)
        self.bart_model = BartForConditionalGeneration(bart_config).to(self.dev)
        # print('initialized')
    # end init

    def forward(self, melody, chroma, accomp, labels): # TODO: add optional accomp input (for continuing composition) and labels (for loss calculation)
        # pad chroma if needed - leave CHROMA_PAD time steps to be sure
        if CHROMA_PAD-chroma['input_ids'].shape[1] > 0:
            chroma_embeds_shape = chroma['input_ids'].shape[1]
            chroma_ids = F.pad( chroma['input_ids'] , ( 0, CHROMA_PAD-chroma_embeds_shape ), 'constant', self.bart_model.config.pad_token_id )
            chroma_attention_mask = F.pad( chroma['attention_mask'], ( 0,CHROMA_PAD-chroma_embeds_shape ), 'constant', 0 )
        bart_encoder_input = torch.cat( (chroma_ids, melody['input_ids']), 1 ).to(self.dev)
        bart_encoder_mask = torch.cat( (chroma_attention_mask, melody['attention_mask'] ), 1 ).to(self.dev)

        vocab_output = self.bart_model(
            input_ids=bart_encoder_input[:,:self.bart_model.config.max_position_embeddings],
            attention_mask=bart_encoder_mask[:,:self.bart_model.config.max_position_embeddings],
            decoder_input_ids=accomp['input_ids'].to(self.dev),
            decoder_attention_mask=accomp['attention_mask'].to(self.dev),
            labels=labels,
            return_dict=True
        )
        return vocab_output
    # end forward
# end class

class MelCAT_GCT(nn.Module):
    def __init__(self, bart_config, gpu=None):
        super(MelCAT_GCT, self).__init__()
        if gpu is None:
            self.dev = torch.device("cpu")
        else:
            self.dev = torch.device("cuda:"+str(gpu) if torch.cuda.is_available() else "cpu")
        self.to(self.dev)
        self.text_encoder = RobertaModel.from_pretrained('roberta-base').to(self.dev)
        self.text_lstm = nn.LSTM(input_size=768, 
                            hidden_size=256, 
                            batch_first=True).to(self.dev)
        # self.midi_encoder = RobertaModel.from_pretrained('/media/datadisk/data/pretrained_models/pop_midi_mlm_base/checkpoint-28680').to(self.dev)
        self.midi_encoder = RobertaModel.from_pretrained(models_folder+'pop_midi_mlm_base/checkpoint-28680').to(self.dev)
        self.midi_rescaler = nn.Linear(512, 256).to(self.dev)
        # self.chroma_encoder = RobertaModel.from_pretrained('/media/datadisk/data/pretrained_models/chroma_mlm_tiny/checkpoint-14336').to(self.dev)
        self.chroma_encoder = RobertaModel.from_pretrained(models_folder+'gct_mlm_tiny/checkpoint-16930').to(self.dev)
        self.bart_model = BartForConditionalGeneration(bart_config).to(self.dev)
        # print('initialized')
    # end init

    def forward(self, text, melody, chroma, accomp): # TODO: add optional accomp input (for continuing composition) and labels (for loss calculation)
        # print('in forward')
        text_embeds = self.text_encoder( input_ids=text['input_ids'].to(self.dev), attention_mask=text['attention_mask'].to(self.dev), output_hidden_states=True )
        text_lstm_output, (_,_) = self.text_lstm(text_embeds.last_hidden_state)
        # print('ids:', melody['input_ids'])
        # print('att:', melody['attention_mask'])
        melody_embeds = self.midi_encoder( input_ids=melody['input_ids'].to(self.dev), attention_mask=melody['attention_mask'].to(self.dev), output_hidden_states=True )
        melody_embeds = self.midi_rescaler(melody_embeds.last_hidden_state)
        chroma_embeds = self.chroma_encoder( input_ids=chroma['input_ids'].to(self.dev), attention_mask=chroma['attention_mask'].to(self.dev), output_hidden_states=True )
        chroma_embeds = chroma_embeds.last_hidden_state
        # pad chroma if needed - leave CHROMA_PAD time steps to be sure
        if CHROMA_PAD-chroma_embeds.shape[1] > 0:
            chroma_embeds_shape = chroma_embeds.shape[1]
            chroma_embeds = F.pad( chroma_embeds , ( 0,0,0, CHROMA_PAD-chroma_embeds_shape ), 'constant', -1 )
            chroma_attention_mask = F.pad( chroma['attention_mask'], ( 0,CHROMA_PAD-chroma_embeds_shape ), 'constant', 0 )
        # print(text_embeds.last_hidden_state.shape)
        # print(text_lstm_output.shape)
        # print(melody_embeds.shape)
        # print(chroma_embeds.shape)
        # print(chroma_attention_mask.shape)
        # print(test_embeds.shape)
        # print(accomp['input_ids'].shape)
        bart_encoder_input = torch.cat( (text_lstm_output[:,-1:,:], chroma_embeds, melody_embeds), 1 )
        # bart_encoder_input = torch.cat( (text_lstm_output[:,-1:,:], melody_embeds.last_hidden_state, chroma_embeds.last_hidden_state), 1 )
        # make masks
        # bart_encoder_mask = torch.cat( (torch.full( (text_lstm_output[:,-1:,:].shape[0], 1), self.bart_model.config.pad_token_id ), \
        #                                 chroma['attention_mask'], melody['attention_mask'] ), 1 ).to(self.dev)
        bart_encoder_mask = torch.cat( (torch.full( (text_lstm_output[:,-1:,:].shape[0], 1), self.bart_model.config.pad_token_id ), \
                                        chroma_attention_mask, melody['attention_mask'] ), 1 ).to(self.dev)
        # print(bart_encoder_input.shape)
        encoder_outputs = self.bart_model.model.encoder( inputs_embeds=bart_encoder_input[:,:self.bart_model.config.max_position_embeddings,:], attention_mask=bart_encoder_mask[:,:self.bart_model.config.max_position_embeddings] )
        # print(encoder_outputs.last_hidden_state.shape)
        decoder_outputs = self.bart_model.model.decoder(
            # inputs_embeds=decoder_input_embeds,
            # input_ids=torch.full( (text_embeds.last_hidden_state.shape[0], 1), self.bart_model.config.eos_token_id ),
            input_ids=accomp['input_ids'][:,:self.bart_model.config.max_position_embeddings].to(self.dev),
            attention_mask=accomp['attention_mask'][:,:self.bart_model.config.max_position_embeddings].to(self.dev),
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            # encoder_attention_mask=attention_mask,
            # attention_mask=decoder_attention_mask,
            return_dict=True
            # **kwargs
        )
        vocab_output = self.bart_model.lm_head(decoder_outputs.last_hidden_state)
        return vocab_output
    # end forward
# end class