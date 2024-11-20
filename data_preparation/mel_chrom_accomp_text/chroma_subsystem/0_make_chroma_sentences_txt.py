import pypianoroll
import os
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import torch
from torch.utils.data import Dataset
import io
import symusic
from tqdm import tqdm

from BinaryTokenizer import BinaryTokenizer, SimpleSerialChromaTokenizer, GCTSerialChromaTokenizer
from miditok import REMI, TokenizerConfig
from transformers import RobertaTokenizer, RobertaModel

import midi_pianoroll_utils as mpu

# datafolder = '/media/maindisk/maximos/data/GiantMIDI-PIano/midis_v1.2/aug/midis'
datafolder = '/media/maindisk/maximos/data/POP909/aug_folder'
datalist = os.listdir( datafolder )
print(len(datalist))

resolution = 4
# binary_tokenizer = BinaryTokenizer(num_digits=12)
# binary_tokenizer = SimpleSerialChromaTokenizer(max_num_segments=8)
binary_tokenizer = GCTSerialChromaTokenizer()

os.makedirs('../../data', exist_ok=True)

sentences_file_path = '../../data/gct_accompaniment_sentences.txt'
error_log_file_path = '../../data/pianoroll_error_pieces.txt'

# open the txt to write to
with open(sentences_file_path, 'w', encoding='utf-8') as the_file:
    the_file.write('')

# also keep a txt with pieces that are problematic
with open(error_log_file_path, 'w') as the_file:
    the_file.write('')

for i in tqdm(range(len( datalist ))):
    main_piece = pypianoroll.read(datafolder + os.sep + datalist[i], resolution=resolution)
    # make deepcopy
    new_piece = deepcopy(main_piece)
    # keep accompaniment
    _, accomp_piece = mpu.split_melody_accompaniment_from_pianoroll(new_piece)
    chroma_zoomed_out = mpu.chroma_from_pianoroll(accomp_piece, resolution=resolution)
    tokenized_chroma = binary_tokenizer(chroma_zoomed_out)
    with open(sentences_file_path, 'a', encoding='utf-8') as the_file:
        the_file.write(' '.join(tokenized_chroma['tokens']) + '\n')
    # try:
    #     main_piece = pypianoroll.read(datafolder + os.sep + datalist[i], resolution=resolution)
    #     # make deepcopy
    #     new_piece = deepcopy(main_piece)
    #     # keep accompaniment
    #     _, accomp_piece = mpu.split_melody_accompaniment_from_pianoroll(new_piece)
    #     chroma_zoomed_out = mpu.chroma_from_pianoroll(accomp_piece, resolution=resolution)
    #     tokenized_chroma = binary_tokenizer(chroma_zoomed_out)
    #     with open(sentences_file_path, 'a', encoding='utf-8') as the_file:
    #         the_file.write(' '.join(tokenized_chroma['tokens']) + '\n')
    # except:
    #     print('ERROR with ', datalist[i])
    #     with open(error_log_file_path, 'a') as the_file:
    #         the_file.write(datalist[i] + '\n')