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

from BinaryTokenizer import BinaryTokenizer, SimpleSerialChromaTokenizer
from miditok import REMI, TokenizerConfig
from transformers import RobertaTokenizer, RobertaModel

datafolder = '/media/maindisk/maximos/data/GiantMIDI-PIano/midis_v1.2/aug/midis'
datalist = os.listdir( datafolder )
print(len(datalist))

resolution = 24
# binary_tokenizer = BinaryTokenizer(num_digits=12)
binary_tokenizer = SimpleSerialChromaTokenizer(max_num_segments=8)

def split_melody_accompaniment(pypianoroll_structure):
    melody_piece = deepcopy( pypianoroll_structure )
    accomp_piece = deepcopy( pypianoroll_structure )

    mel_pr = melody_piece.tracks[0].pianoroll
    acc_pr = accomp_piece.tracks[0].pianoroll

    pr = np.array(melody_piece.tracks[0].pianoroll)
    running_melody = -1
    i = 0
    # for i in range( pr.shape[0] ):
    while i < pr.shape[0]:
        # check if any note
        if np.sum(pr[i,:]) > 0:
            # get running max
            running_max = np.max( np.nonzero( pr[i,:] ) )
            # check if there exists a running melody
            if running_melody > -1:
                # check if running melody is continued
                if running_melody == running_max:
                    # remove all lower pitches from melody
                    mel_pr[i, :running_max] = 0
                    # remove higher pitch from accomp
                    acc_pr[i, running_max] = 0
                else:
                    # running melody may need to change
                    # check if new highest pitch just started
                    if running_max > running_melody:
                        # a new higher note has started
                        # finish previous note that was highest until now
                        j = 0
                        while j+i < mel_pr.shape[0] and mel_pr[i+j, running_melody] > 0 and running_max > running_melody:
                            mel_pr[i+j, :running_melody] = 0
                            mel_pr[i+j, running_melody+1:running_max] = 0
                            acc_pr[i+j, running_melody] = 0
                            acc_pr[i+j, running_max] = 0
                            if np.sum( pr[i+j,:] ) > 0:
                                running_max = np.max( np.nonzero( pr[i+j,:] ) )
                            else:
                                running_melody = -1
                                break
                            j += 1
                        # start new running melody
                        i += j-1
                        running_melody = running_max
                    else:
                        # i should be > 0 since we have that running_melody > -1
                        # a lower note has come
                        # if has begun earlier, it should be ignored
                        if pr[i-1, running_max] > 0:
                            # its continuing an existing note - not part of melody
                            mel_pr[i, :] = 0
                            # running max should not be canceled, it remains as ghost max
                            # until a new higher max or a fresh lower max starts
                        else:
                            # a new fresh lower max starts that shouldn't be ignored
                            # start new running melody
                            running_melody = running_max
                            # remove all lower pitches from melody
                            mel_pr[i, :running_max] = 0
                            # remove higher pitch from accomp
                            acc_pr[i, running_max] = 0
            else:
                # no running melody, check max conditions
                # new note started - make it the running melody
                running_melody = running_max
                # remove all lower pitches from melody
                mel_pr[i, :running_max] = 0
                # remove higher pitch from accomp
                acc_pr[i, running_max] = 0
            # end if
        else:
            # there is a gap
            running_melody = -1
        # end if
        i += 1
    # end for
    return melody_piece, accomp_piece
# end split_melody_accompaniment

def chroma_from_pianoroll(main_piece, resolution=24):
    # first binarize a new deep copy
    binary_piece = deepcopy(main_piece)
    binary_piece.binarize()
    # make chroma
    chroma = binary_piece.tracks[0].pianoroll[:,:12]
    for i in range(12, 128-12, 12):
        chroma = np.logical_or(chroma, binary_piece.tracks[0].pianoroll[:,i:(i+12)])
    chroma[:,-6:] = np.logical_or(chroma[:,-6:], binary_piece.tracks[0].pianoroll[:,-6:])
    # quarter chroma resolution
    chroma_tmp = np.zeros( (1,12) )
    chroma_zoomed_out = None
    for i in range(chroma.shape[0]):
        chroma_tmp += chroma[i,:]
        if (i+1)%resolution == 0:
            if chroma_zoomed_out is None:
                chroma_zoomed_out = chroma_tmp >= np.mean( chroma_tmp )
            else:
                chroma_zoomed_out = np.vstack( (chroma_zoomed_out, chroma_tmp >= np.mean( chroma_tmp )) )
    if np.sum( chroma_tmp ) > 0:
        if chroma_zoomed_out is None:
            chroma_zoomed_out = chroma_tmp >= np.mean( chroma_tmp )
        else:
            chroma_zoomed_out = np.vstack( (chroma_zoomed_out, chroma_tmp >= np.mean( chroma_tmp )) )
    return chroma_zoomed_out
# end chroma_from_pianoroll

# open the txt to write to
with open('chroma_accompaniment_sentences.txt', 'w') as the_file:
    the_file.write('')

# also keep a txt with pieces that are problematic
with open('pianoroll_error_pieces.txt', 'w') as the_file:
    the_file.write('')

for i in tqdm(range(len( datalist ))):
    try:
        main_piece = pypianoroll.read(datafolder + os.sep + datalist[i], resolution=resolution)
        # make deepcopy
        new_piece = deepcopy(main_piece)
        # keep accompaniment
        _, accomp_piece = split_melody_accompaniment(new_piece)
        chroma_zoomed_out = chroma_from_pianoroll(accomp_piece, resolution=resolution)
        tokenized_chroma = binary_tokenizer(chroma_zoomed_out)
        with open('chroma_accompaniment_sentences.txt', 'a') as the_file:
            the_file.write(' '.join(tokenized_chroma['tokens']) + '\n')
    except:
        print('ERROR with ', datalist[i])
        with open('pianoroll_error_pieces.txt', 'a') as the_file:
            the_file.write(datalist[i] + '\n')