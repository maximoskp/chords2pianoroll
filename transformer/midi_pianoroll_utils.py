import numpy as np
from copy import deepcopy
import io
import symusic
import matplotlib.pyplot as plt

def remove_drum_tracks_in_place(main_piece):
    for t in main_piece.tracks:
        if t.is_drum:
            main_piece.tracks.remove(t)
# end remove_drum_tracks_in_place

def remove_drum_tracks(main_piece):
    new_piece = deepcopy(main_piece)
    for t in new_piece.tracks:
        if t.is_drum:
            new_piece.tracks.remove(t)
    return new_piece
# end remove_drum_tracks_in_place

def is_first_track_melody(main_piece):
    t0 = main_piece.tracks[0]
    b0 = t0.pianoroll > 0
    s0 = b0.sum(axis=1)
    m0 = s0[s0 > 0].mean()
    return m0 < 1.05 # allow 5% overlap
# end is_first_track_melody

def chroma_from_pianoroll(main_piece, resolution=24):
    # first binarize a new deep copy
    binary_piece = deepcopy(main_piece)
    binary_piece.binarize()
    # make chroma
    # chroma = binary_piece.tracks[0].pianoroll[:,:12]
    chroma = binary_piece.stack().max(axis=0)[:,:12]
    for i in range(12, 128-12, 12):
        # chroma = np.logical_or(chroma, binary_piece.tracks[0].pianoroll[:,i:(i+12)])
        chroma = np.logical_or(chroma, binary_piece.stack().max(axis=0)[:,i:(i+12)])
    # chroma[:,-6:] = np.logical_or(chroma[:,-6:], binary_piece.tracks[0].pianoroll[:,-6:])
    chroma[:,-6:] = np.logical_or(chroma[:,-6:], binary_piece.stack().max(axis=0)[:,-6:])
    # quarter chroma resolution
    chroma_tmp = np.zeros( (1,12) )
    chroma_zoomed_out = None
    for i in range(chroma.shape[0]):
        chroma_tmp += chroma[i,:]
        if (i+1)%resolution == 0:
            if chroma_zoomed_out is None:
                chroma_zoomed_out = np.logical_and( chroma_tmp >= np.mean( chroma_tmp ), chroma_tmp > 0 )
                chroma_tmp = np.zeros( (1,12) )
            else:
                chroma_zoomed_out = np.vstack( (chroma_zoomed_out, np.logical_and( chroma_tmp >= np.mean( chroma_tmp ), chroma_tmp > 0 )) )
                chroma_tmp = np.zeros( (1,12) )
    if np.sum( chroma_tmp ) > 0:
        if chroma_zoomed_out is None:
            chroma_zoomed_out = np.logical_and( chroma_tmp >= np.mean( chroma_tmp ), chroma_tmp > 0 )
        else:
            chroma_zoomed_out = np.vstack( (chroma_zoomed_out, np.logical_and( chroma_tmp >= np.mean( chroma_tmp ), chroma_tmp > 0 )) )
    # print('chroma.shape: ', chroma.shape)
    # print('chroma_zoomed_out.shape: ', chroma_zoomed_out.shape)
    # plt.imshow(chroma)
    # plt.show()
    # plt.imshow(chroma_zoomed_out)
    # plt.show()
    return chroma_zoomed_out
# end chroma_from_pianoroll

def split_melody_accompaniment_from_pianoroll(pypianoroll_structure, check_first_track=True):
    melody_piece = deepcopy( pypianoroll_structure )
    accomp_piece = deepcopy( pypianoroll_structure )
    if check_first_track and is_first_track_melody(pypianoroll_structure) and len(accomp_piece.tracks)>1 :
        mel_pr = melody_piece.tracks[0].pianoroll
        del( accomp_piece.tracks[0] )
        acc_pr = accomp_piece.stack().max(axis=0)
    else:
        # mel_pr = melody_piece.tracks[0].pianoroll
        # acc_pr = accomp_piece.tracks[0].pianoroll
        mel_pr = melody_piece.stack().max(axis=0)
        acc_pr = accomp_piece.stack().max(axis=0)

        # pr = np.array(melody_piece.tracks[0].pianoroll)
        pr = np.array(mel_pr)
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
    # end else is_first_track_melody
    # pass new pianorolls to track 0
    melody_piece.tracks[0].pianoroll = mel_pr
    accomp_piece.tracks[0].pianoroll = acc_pr
    # remove all other tracks
    while len(melody_piece.tracks) > 1:
        del( melody_piece.tracks[-1] )
    while len(accomp_piece.tracks) > 1:
        del( accomp_piece.tracks[-1] )
    return melody_piece, accomp_piece
# end split_melody_accompaniment

def pianoroll_to_midi_bytes(pianoroll_structure):
    # initialize bytes handle
    b_handle = io.BytesIO()
    # write midi data to bytes handle
    pianoroll_structure.write(b_handle)
    # start read pointer from the beginning
    b_handle.seek(0)
    # create a buffered reader to read the handle
    buffered_reader = io.BufferedReader(b_handle)
    # create a midi object from the "file", i.e., buffered reader
    midi_bytes = symusic.Score.from_midi(b_handle.getvalue())
    # close the bytes handle
    b_handle.close()
    return midi_bytes
# end 