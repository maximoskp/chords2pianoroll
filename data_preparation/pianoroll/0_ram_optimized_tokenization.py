import pypianoroll # change in outputs.py
from mido import MidiFile
import os
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from BinaryTokenizer import BinaryTokenizer
from tqdm import tqdm
import io
# https://copyprogramming.com/howto/convert-bytes-into-bufferedreader-object-in-python
import pickle

from BinaryTokenizer import BinaryTokenizer
from miditok import REMI, TokenizerConfig
from miditoolkit import MidiFile
# from pathlib import Path
import pandas as pd

# Our tokenizer's configuration
PITCH_RANGE = (21, 109)
BEAT_RES = {(0, 1): 8, (1, 2): 4, (2, 4): 2, (4, 8): 1}
NB_VELOCITIES = 24
SPECIAL_TOKENS = ["PAD", "MASK", "BOS", "EOS"]
USE_CHORDS = False
USE_RESTS = False
USE_TEMPOS = False
USE_TIME_SIGNATURE = False
USE_PROGRAMS = False
NB_TEMPOS = 32
TEMPO_RANGE = (50, 200)  # (min_tempo, max_tempo)
TOKENIZER_PARAMS = {
    "pitch_range": PITCH_RANGE,
    "beat_res": BEAT_RES,
    "num_velocities": NB_VELOCITIES,
    "special_tokens": SPECIAL_TOKENS,
    "use_chords": USE_CHORDS,
    "use_rests": USE_RESTS,
    "use_tempos": USE_TEMPOS,
    "use_time_signatures": USE_TIME_SIGNATURE,
    "use_programs": USE_PROGRAMS,
    "num_tempos": NB_TEMPOS,
    "tempo_range": TEMPO_RANGE,
}
config = TokenizerConfig(**TOKENIZER_PARAMS)
tokenizer = REMI(config)

# keep tokenizer info for later
tokenization_info = {}
tokenization_info['pad_id'] = tokenizer._tokens_to_ids(['PAD_None'])[0]
tokenization_info['bos_id'] = tokenizer._tokens_to_ids(['BOS_None'])[0]
tokenization_info['eos_id'] = tokenizer._tokens_to_ids(['EOS_None'])[0]
tokenization_info['mask_id'] = tokenizer._tokens_to_ids(['MASK_None'])[0]
tokenization_info['max_len'] = 0

binaryTokenizer = BinaryTokenizer(num_digits=12)

midifolder = '../data/giantmidi_small/'
midifiles = os.listdir(midifolder)

save_every = 500
file_idx = 0

# size in beats
segment_size = 64
piece_idx = 0

def make_segment(main_piece, tmp_pianoroll, start_idx, end_idx, piece_idx, transposition_idx, segment_idx):
    new_piece = deepcopy(main_piece)
    new_piece.tracks[0].pianoroll = deepcopy(tmp_pianoroll)
    # trim and binarize
    new_piece.trim(start_idx,end_idx)
    new_piece.binarize()
    # make chroma
    chroma = new_piece.tracks[0].pianoroll[:,:12]
    for i in range(12, 128-12, 12):
        chroma = np.logical_or(chroma, new_piece.tracks[0].pianoroll[:,i:(i+12)])
    chroma[:,-6:] = np.logical_or(chroma[:,-6:], new_piece.tracks[0].pianoroll[:,-6:])
    indexed_chroma = binaryTokenizer.transform( chroma )
    # make name of piece
    piece_name = f'p{piece_idx:05}_t{transposition_idx:02}_s{segment_idx:04}'
    # initialize bytes handle
    b_handle = io.BytesIO()
    # write midi data to bytes handle
    new_piece.write(b_handle)
    # start read pointer from the beginning
    b_handle.seek(0)
    # create a buffered reader to read the handle
    buffered_reader = io.BufferedReader(b_handle)
    # create a midi object from the "file", i.e., buffered reader
    midi_object = MidiFile(file=buffered_reader)
    # close the bytes handle
    b_handle.close()
    # print('midi_object: ', midi_object)
    tmp_tokens = tokenizer.midi_to_tokens( midi_object, apply_bpe_if_possible=False)
    return piece_name, tmp_tokens[0].ids, indexed_chroma

# initialize catalogue
with open('data/piece_per_idx.txt', 'w') as f:
    print('piece_idx, file_name', file=f)
# initialize problematic files catalogue
with open('data/problematic_pieces.txt', 'w') as f:
    print('piece_idx, file_name', file=f)

dicts = []

for midifile in tqdm(midifiles[piece_idx:]):
    try:
        main_piece = pypianoroll.read(midifolder + midifile)
    except:
        with open('data/problematic_pieces.txt', 'a') as f:
            print(f'{piece_idx:05}, {midifile}', file=f)
    else:
        with open('data/piece_per_idx.txt', 'a') as f:
            print(f'{piece_idx:05}, {midifile}', file=f)
        # keep size to know when to end
        main_piece_size = main_piece.downbeat.shape[0]
        transposition_idx = 0
        # roll
        for r in tqdm(range(-6,6,1), leave=False):
            # print(f'running for roll: {r}')
            start_idx = 0
            end_idx = segment_size*main_piece.resolution
            tmp_pianoroll = np.roll(main_piece.tracks[0].pianoroll, [0,r])
            segment_idx = 0
            tmp_d = {}
            while end_idx < main_piece_size:
                # print(f'running for start: {start_idx} and end: {end_idx}')
                piece_name, tmp_tokens, indexed_chroma = make_segment(main_piece, tmp_pianoroll, start_idx, end_idx, piece_idx, transposition_idx, segment_idx)
                start_idx = end_idx
                end_idx += segment_size*main_piece.resolution
                tmp_d['name'] = piece_name
                tmp_d['chroma'] = indexed_chroma
                tmp_d['surface'] = tmp_tokens
                tmp_d['starts'] = segment_idx == 0
                tmp_d['ends'] = False
                dicts.append(tmp_d)
                segment_idx += 1
                if tokenization_info['max_len'] < len(tmp_tokens):
                    tokenization_info['max_len'] = len(tmp_tokens)
            # end end_idx while
            end_idx = main_piece_size
            start_idx = end_idx - segment_size*main_piece.resolution
            tmp_d = {}
            if start_idx > 0:
                piece_name, tmp_tokens, indexed_chroma = make_segment(main_piece, tmp_pianoroll, start_idx, end_idx, piece_idx, transposition_idx, segment_idx)
                tmp_d['name'] = piece_name
                tmp_d['chroma'] = indexed_chroma
                tmp_d['surface'] = tmp_tokens
                tmp_d['starts'] = segment_idx == 0
                tmp_d['ends'] = True
                dicts.append(tmp_d)
                if tokenization_info['max_len'] < len(tmp_tokens):
                    tokenization_info['max_len'] = len(tmp_tokens)
            transposition_idx += 1
        # end transposition range for
        piece_idx += 1
        if piece_idx%save_every == 0:
            with open(f'data/dicts_{file_idx}.pickle', 'wb') as handle:
                pickle.dump(dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)
            file_idx += 1
            dicts = {}
# end midifile for

with open(f'data/tokenization_info.pickle', 'wb') as handle:
    pickle.dump(tokenization_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'data/dicts_{file_idx}.pickle', 'wb') as handle:
    pickle.dump(dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('filename.pickle', 'rb') as handle:
#     b = pickle.load(handle)