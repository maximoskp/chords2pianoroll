import torch
from torch.utils.data import Dataset

# from ..data_preparation.mel_chrom_accomp_text.chroma_subsystem.BinaryTokenizer import SimpleSerialChromaTokenizer
from BinaryTokenizer import SimpleSerialChromaTokenizer
from miditok import REMI, TokenizerConfig
from transformers import RobertaTokenizer, RobertaTokenizerFast, RobertaModel

# from ..data_preperation.mel_chrom_accomp_text import midi_pianoroll_utils as mpu
import midi_pianoroll_utils as mpu

from pathlib import Path
import os

import numpy as np
from copy import deepcopy

import pypianoroll

class LiveMelCATDataset(Dataset):
    def __init__(self, midis_folder, segment_size=64, resolution=24):
        self.midis_folder = midis_folder
        self.midis_list = os.listdir(midis_folder)
        self.segment_size = segment_size
        self.resolution = resolution
        self.binary_chroma_tokenizer = SimpleSerialChromaTokenizer()
        self.remi_tokenizer = REMI(params=Path('/media/datadisk/data/pretrained_models/midis_REMI_BPE_tokenizer.json'))
        self.roberta_tokenizer_chroma = RobertaTokenizerFast.from_pretrained('/media/datadisk/data/pretrained_models/chroma_mlm_tiny/chroma_wordlevel_tokenizer')
        self.roberta_tokenizer_midi = RobertaTokenizerFast.from_pretrained('/media/datadisk/data/pretrained_models/midi_mlm_tiny/midi_wordlevel_tokenizer')
        self.roberta_tokenizer_text = RobertaTokenizer.from_pretrained('roberta-base')
        self.padding_values = {
            'melody': self.roberta_tokenizer_midi.pad_token_id,
            'chroma': self.roberta_tokenizer_chroma.pad_token_id,
            'text': self.roberta_tokenizer_text.pad_token_id,
            'accomp': self.roberta_tokenizer_midi.pad_token_id
        }
        self.max_seq_lengths = {
            'melody': 1024,
            'chroma': 1024,
            'text': 1024,
            'accomp': 4096
        }
    # end init
    def __len__(self):
        return len(self.midis_list)
    # end len
    def __getitem__(self, idx):
        print('idx:', idx)
        print(self.midis_list[idx])
        # load a midi file in pianoroll
        main_piece = pypianoroll.read(self.midis_folder + os.sep + self.midis_list[idx], resolution=self.resolution)
        main_piece_size = main_piece.downbeat.shape[0]
        # make deepcopy
        new_piece = deepcopy(main_piece)
        # trim piece
        start_idx = np.random.randint( main_piece_size - self.segment_size*main_piece.resolution )
        end_idx = start_idx + self.segment_size*main_piece.resolution
        new_piece.trim(start_idx, end_idx)
        # split melody - accompaniment
        melody_piece, accomp_piece = mpu.split_melody_accompaniment_from_pianoroll( new_piece )
        # keep chroma from accompaniment
        chroma_zoomed_out = mpu.chroma_from_pianoroll(accomp_piece)
        # tokenize chroma to text tokens
        tokenized_chroma = self.binary_chroma_tokenizer(chroma_zoomed_out)
        chroma_string = ' '.join( tokenized_chroma['tokens'] )
        chroma_tokens = self.roberta_tokenizer_chroma( chroma_string )
        # make ghost files of melody and accomp pieces
        melody_file = mpu.pianoroll_to_midi_bytes(melody_piece)
        accomp_file = mpu.pianoroll_to_midi_bytes(accomp_piece)
        # tokenize melody and accompaniment midi to text
        remi_tokenized_melody = self.remi_tokenizer(melody_file)
        melody_string = ' '.join(remi_tokenized_melody[0].tokens)
        melody_tokens = self.roberta_tokenizer_midi(melody_string)
        remi_tokenized_accomp = self.remi_tokenizer(accomp_file)
        accomp_string = ' '.join(remi_tokenized_accomp[0].tokens)
        accomp_tokens = self.roberta_tokenizer_midi(accomp_string)
        # get text from title
        text_description = self.midis_list[idx]
        # tokenize text
        text_tokens = self.roberta_tokenizer_text(text_description)
        # return torch.LongTensor(melody_tokens['input_ids']),
        return {
            'melody': torch.LongTensor(melody_tokens['input_ids']),
            'chroma': torch.LongTensor(chroma_tokens['input_ids']),
            'text': torch.LongTensor(text_tokens['input_ids']),
            'accomp': torch.LongTensor(accomp_tokens['input_ids'])
        }
    # end getitem
# end class

class MelCATCollator:
    def __init__(self, max_seq_lens=None, padding_values=None):
        self.max_seq_lens = max_seq_lens
        self.padding_values = padding_values
    # end init

    def __call__(self, batch):
        # Create a dictionary to hold the batched data
        batch_dict = {}
        # Assume the batch is a list of dictionaries (each sample is a dict)
        for key in batch[0]:
            batch_dict[key] = {}
            values = [item[key] for item in batch]
            if isinstance(values[0], list):
                # If values are lists (sequences of variable lengths), pad them
                padded_values = pad_sequence([torch.tensor(v) for v in values], 
                                            batch_first=True, padding_value=self.padding_values[key])
            elif isinstance(values[0], torch.Tensor):
                # If values are tensors, stack them directly
                padded_values = pad_sequence(values, batch_first=True, padding_value=self.padding_values[key])
            # trim to max length
            if self.max_seq_lens is not None and padded_values.shape[1] > self.max_seq_lens[key]:
                padded_values = padded_values[:,:self.max_seq_lens[key]]
            batch_dict[key]['input_ids'] = padded_values
            attention_mask = torch.ones_like(padded_values, dtype=torch.long)
            attention_mask[padded_values == self.padding_values[key]] = 0
            batch_dict[key]['attention_mask'] = attention_mask
        return batch_dict
    # end call
# end class