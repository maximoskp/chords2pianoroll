import pypianoroll
import os
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from BinaryTokenizer import BinaryTokenizer
from tqdm import tqdm

# make folder structure
os.makedirs('data', exist_ok=True)
os.makedirs('data/pianorolls', exist_ok=True)
os.makedirs('data/midis', exist_ok=True)
os.makedirs('data/midi_tokens', exist_ok=True)

binaryTokenizer = BinaryTokenizer(num_digits=12)

midifolder = '../data/giantmidi/'
midifiles = os.listdir(midifolder)

# size in beats
segment_size = 64
piece_idx = 0

# initialize catalogue
with open('piece_per_idx.txt', 'w') as f:
    print('piece_idx, file_name', file=f)

for midifile in tqdm(midifiles[:1]):
    main_piece = pypianoroll.read(midifolder + midifile)
    # keep size to know when to end
    main_piece_size = main_piece.downbeat.shape[0]
    transposition_idx = 0
    with open('piece_per_idx.txt', 'a') as f:
        print(f'{piece_idx:05}, {midifile}', file=f)
    # roll
    for r in tqdm(range(-6,6,1), leave=False):
        # print(f'running for roll: {r}')
        start_idx = 0
        end_idx = segment_size*main_piece.resolution
        tmp_pianoroll = np.roll(main_piece.tracks[0].pianoroll, [0,r])
        segment_idx = 0
        while end_idx < main_piece_size:
            # print(f'running for start: {start_idx} and end: {end_idx}')
            # create deepcopy of main piece
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
            # save npz
            np.savez(f'data/pianorolls/' + piece_name + '.npz', indexed_chroma=indexed_chroma)
            # save midi
            new_piece.write(f'data/midis/' + piece_name + '.mid')
            start_idx = end_idx
            end_idx += segment_size*main_piece.resolution
            segment_idx += 1
        # end end_idx while
        transposition_idx += 1
    # end transposition range for
    piece_idx += 1
# end midifile for