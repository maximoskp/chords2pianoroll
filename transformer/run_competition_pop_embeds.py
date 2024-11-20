from transformers import BartForConditionalGeneration, BartConfig
from transformers import RobertaTokenizerFast
import torch
from torch.utils.data import DataLoader

from miditok import REMI, TokenizerConfig, TokSequence
from pathlib import Path

from models import MelCAT_base
from dataset_utils import LiveMelCATDataset, MelCATCollator

from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

import os
import numpy as np
import csv

from tqdm import tqdm

import json

import symusic
import json2midi_utils as j2m
import pretty_midi as pm

MAX_LENGTH = 2048

models_folder = '/media/maindisk/maximos/data/pretrained_models/'
# models_folder = '/media/datadisk/data/pretrained_models/'

roberta_tokenizer_midi = RobertaTokenizerFast.from_pretrained(models_folder+'pop_midi_mlm_base/pop_wordlevel_tokenizer')
remi_tokenizer = REMI(params=Path(models_folder+'/pop_midi_mlm_base/pop_REMI_BPE_tokenizer.json'))

bart_config = BartConfig(
    vocab_size=roberta_tokenizer_midi.vocab_size,
    pad_token_id=roberta_tokenizer_midi.pad_token_id,
    bos_token_id=roberta_tokenizer_midi.bos_token_id,
    eos_token_id=roberta_tokenizer_midi.eos_token_id,
    decoder_start_token_id=roberta_tokenizer_midi.bos_token_id,
    forced_eos_token_id=roberta_tokenizer_midi.eos_token_id,
    max_position_embeddings=MAX_LENGTH,
    encoder_layers=8,
    encoder_attention_heads=16,
    encoder_ffn_dim=4096,
    decoder_layers=8,
    decoder_attention_heads=16,
    decoder_ffn_dim=4096,
    d_model=256,
    encoder_layerdrop=0.3,
    decoder_layerdrop=0.3,
    dropout=0.3
)

run_on_gpu = False

if run_on_gpu:
    dev = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model = MelCAT_base(bart_config, gpu=2).to(dev)
    checkpoint = torch.load('saved_models/bart_pop_embeds/bart_pop_embeds.pt', weights_only=True)
else:
    dev = torch.device("cpu")
    model = MelCAT_base(bart_config, gpu=None).to(dev)
    checkpoint = torch.load('saved_models/bart_pop_embeds/bart_pop_embeds.pt', map_location="cpu", weights_only=True)

model.load_state_dict(checkpoint)

model.eval()

DEFAULT_V_MEL = 70
DEFAULT_V_ACC = 50
DEFAULT_V_CHD = 50

temperature = 0.8

# TODO: bpm should come as an input from arguments
bpm = 120
# TODO: this should come as an input from arguments
json_inputs_folder = 'competition_test_inputs'
# get all json inputs
json_inputs = os.listdir( json_inputs_folder )

# prepare folders for competition
# prepare tempory midi folder for inputs
tmp_midis_folder = 'tmp_midis'
os.makedirs(tmp_midis_folder, exist_ok=True)
# make folder for json output
json_outputs_folder = 'competition_test_outputs'
os.makedirs(json_outputs_folder, exist_ok=True)
# make folder for midi total output
midi_outputs_folder = 'competition_test_midi_outputs'
os.makedirs(midi_outputs_folder, exist_ok=True)

dict_input = {}

# make midis from json inputs
for json_input in json_inputs:
    print(json_input)
    dict_input[json_input.split('.')[0]] = j2m.load_json( os.path.join(json_inputs_folder, json_input) )
    melody_notes = j2m.note_list_to_notes(dict_input[json_input.split('.')[0]]['melody'], default_v=DEFAULT_V_ACC, bpm=bpm)
    chord_notes = j2m.chord_list_to_notes(dict_input[json_input.split('.')[0]]['chords'], default_v=DEFAULT_V_CHD, bpm=bpm)
    midi = pm.PrettyMIDI(initial_tempo=bpm)
    midi.instruments = [pm.Instrument(65, is_drum=False, name='melody'), 
                        pm.Instrument(0, is_drum=False, name='chords')
                        ]
    midi.instruments[0].notes = melody_notes
    midi.instruments[1].notes = chord_notes
    midi.write(os.path.join(tmp_midis_folder, json_input.split('.')[0] +'.mid'))

dataset = LiveMelCATDataset(tmp_midis_folder, segment_size=40, resolution=4, max_seq_len=1024, only_beginning=True,\
                            ignore_short_pieces=False)
custom_collate_fn = MelCATCollator(max_seq_lens=dataset.max_seq_lengths, padding_values=dataset.padding_values)
# dataloader = DataLoader(dataset, batch_size=1, collate_fn=custom_collate_fn, drop_last=False)

def sample_with_temperature(logits, temperature=1.0):
    # Scale logits by temperature
    logits = logits / temperature
    # Apply softmax to get probabilities
    probs = F.softmax(logits, dim=-1)

    # Flatten the logits if necessary
    batch_size, seq_len, vocab_size = probs.shape
    probs = probs.view(-1, vocab_size)  # Merge batch_size and seq_len dimensions
    
    # Sample from the probability distribution
    sampled_tokens = torch.multinomial(probs, num_samples=1)
    
    # Reshape back to [batch_size, seq_len, 1]
    sampled_tokens = sampled_tokens.view(batch_size, seq_len, 1)

    # # Sample from the probability distribution
    # sampled_token = torch.multinomial(probs, num_samples=1)
    return sampled_tokens

def generate_bart_tokens(d, temperature=1.0, max_seq_len=4096, num_bars=1000):
    accomp_input = {
        'input_ids' : torch.LongTensor([[roberta_tokenizer_midi.bos_token_id]]),
        'attention_mask' : torch.LongTensor([[1]])
    }
    bars_count = 0
    logits = model(d['text'], d['melody'], d['chroma'], accomp_input)
    sampled_tokens = sample_with_temperature(logits, temperature)
    accomp_input['input_ids'] = torch.cat( (accomp_input['input_ids'].to(dev), sampled_tokens[0][-1:].to(dev)), -1)
    accomp_input['attention_mask'] = torch.cat( (accomp_input['attention_mask'].to(dev), torch.full(sampled_tokens[0][-1:].shape,1).to(dev)), -1)
    bars_count += sampled_tokens[0][-1][0] == 5
    while sampled_tokens[0][-1][0] != roberta_tokenizer_midi.eos_token_id and \
        accomp_input['input_ids'].shape[-1] < max_seq_len and\
        num_bars >= bars_count:
        print(accomp_input['input_ids'].shape[-1], 'bars_count:', bars_count, end='\r')
        logits = model(d['text'], d['melody'], d['chroma'], accomp_input)
        sampled_tokens = sample_with_temperature(logits, temperature)
        # print(sampled_tokens[0][-1])
        bars_count += sampled_tokens[0][-1][0] == 5
        if num_bars < bars_count:
            break
        accomp_input['input_ids'] = torch.cat( (accomp_input['input_ids'].to(dev), sampled_tokens[0][-1:].to(dev)), -1)
        accomp_input['attention_mask'] = torch.cat( (accomp_input['attention_mask'].to(dev), torch.full(sampled_tokens[0][-1:].shape,1).to(dev)), -1)
    return accomp_input

piece_idx = 0
for i in range(len(dataset)):
    d = custom_collate_fn( [dataset[i]] )
    piece_name = dataset.midis_list[i]
    print(piece_idx, '- piece:', piece_name)
    print(d['melody']['input_ids'])
    # toks = roberta_tokenizer_midi.convert_ids_to_tokens(d['accomp']['input_ids'][0])
    a = generate_bart_tokens(d, temperature=temperature, max_seq_len=1024, num_bars=9)
    toks = roberta_tokenizer_midi.convert_ids_to_tokens(a['input_ids'][0])
    toks_miditok = []
    for tok in toks:
        if '_' in tok:
            toks_miditok.append(tok.replace('x','.'))
    tok_seq = TokSequence(toks_miditok)
    m = remi_tokenizer.tokens_to_midi(tokens=[tok_seq])
    # make json from acc
    jmidi = {'acc': []}
    if len(m.tracks) == 0:
        print('ERROR: failed to generate for ' + piece_name)
    else:
        track = m.tracks[0]
        for note in track.notes:
            jmidi['acc'].append({
                'start': note.time / m.ticks_per_quarter*4,
                'pitch': note.pitch,
                'duration': note.duration / m.ticks_per_quarter*4
            })
        # incorporate in input
        dict_input[piece_name.split('.')[0]]['acc'] = jmidi['acc']
        # save json
        with open(os.path.join( json_outputs_folder, piece_name+'_out.json'), 'w') as json_file:
            json.dump(dict_input[piece_name.split('.')[0]], json_file)
        # make single midi with accompaniment
        for json_input in json_inputs:
            melody_notes = j2m.note_list_to_notes(dict_input[piece_name.split('.')[0]]['melody'], default_v=DEFAULT_V_ACC, bpm=bpm)
            acc_notes = j2m.note_list_to_notes(dict_input[piece_name.split('.')[0]]['acc'], default_v=DEFAULT_V_ACC, bpm=bpm)
            midi = pm.PrettyMIDI(initial_tempo=bpm)
            midi.instruments = [pm.Instrument(65, is_drum=False, name='melody'), 
                                pm.Instrument(0, is_drum=False, name='acc')
                                ]
            midi.instruments[0].notes = melody_notes
            midi.instruments[1].notes = acc_notes
            midi.write(os.path.join(midi_outputs_folder, piece_name+'_out.mid'))
    piece_idx += 1