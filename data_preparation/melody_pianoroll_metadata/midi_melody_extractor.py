import mido
import copy
import os
import shutil

## Module 1: extracting the upper line with mido:

def get_note(msg):
    dict = msg.dict()
    if 'note' not in dict.keys():
        return -999
    else:
        return dict['note']

def break_track_into_moments(track, ticks_per_beat=48):
    moments = []
    temp_list = []
    note_on_time_cummulative = 0 # in ticks
    for msg in track:
        note_on_time_cummulative = note_on_time_cummulative + msg.dict()['time']
        if msg.type == 'note_on' and msg.dict()['velocity'] > 0:
            if note_on_time_cummulative <= ticks_per_beat/32:
                temp_list.append(msg)
            else:
                moments.append(temp_list)
                temp_list = []
                temp_list.append(msg)
            note_on_time_cummulative = 0
    return moments

def single_track_upper_melody(track):
    temp_time_to_add = 0
    new_list = []

    for msg in track:
        dict = msg.dict()
        if 'channel' in dict.keys() and dict['channel'] != 0:
            time_to_add = dict['time']
            temp_time_to_add = temp_time_to_add + time_to_add
        else:
            msg.time = msg.time + temp_time_to_add
            temp_time_to_add = 0
            new_list.append(msg)

    moments = break_track_into_moments(new_list)
    watchlist = []
    result_list = []

    temp_time_to_add = 0
    for m in moments:
        tmp_block_list = []
        highest_note_value = max([get_note(x) for x in m])
        for msg in m:
            # not top voice begin:
            if msg.type == 'note_on' and msg.dict()['note'] < highest_note_value and msg.dict()['velocity'] > 0:
                note = msg.dict()['note']
                watchlist.append(note)
                temp_time_to_add = temp_time_to_add + msg.dict()['time']
            # not top voice end:
            elif msg.type == 'note_on' and  msg.dict()['velocity'] == 0 and msg.dict()['note'] in watchlist:
                note = msg.dict()['note']
                watchlist.remove(note)
                temp_time_to_add = temp_time_to_add + msg.dict()['time']
            # top voice:
            else:
                msg.time = msg.dict()['time'] + temp_time_to_add
                temp_time_to_add = 0
                tmp_block_list.append(msg)
        result_list.extend(tmp_block_list)

    result_track = copy.deepcopy(track)
    result_track.clear()
    # result_track = mido.MidiTrack()
    result_track.extend(result_list)
    return result_track

def single_track_melody_and_accompaniment(track,ticks_per_beat=48):
    temp_time_to_add = 0
    new_list = []

    for msg in track:
        dict = msg.dict()
        if 'channel' in dict.keys() and dict['channel'] != 0:
            time_to_add = dict['time']
            temp_time_to_add = temp_time_to_add + time_to_add
        else:
            msg.time = msg.time + temp_time_to_add
            temp_time_to_add = 0
            new_list.append(msg)

    moments = break_track_into_moments(new_list, ticks_per_beat)
    watchlist = []
    melody_list = []
    accomp_list = []

    temp_time_to_add_mel = 0
    temp_time_to_add_acc = 0
    for m in moments:
        tmp_melody_block_list = []
        tmp_accomp_block_list = []
        tmp_notes_list = [get_note(x) for x in m]
        if len( tmp_notes_list ) > 0:
            highest_note_value = max(tmp_notes_list)
            for msg in m:
                # not top voice begin:
                if msg.type == 'note_on' and msg.dict()['note'] < highest_note_value and msg.dict()['velocity'] > 0:
                    note = msg.dict()['note']
                    watchlist.append(note)
                    msg.time = msg.dict()['time'] + temp_time_to_add_acc
                    temp_time_to_add_acc = 0
                    tmp_accomp_block_list.append(msg)
                    temp_time_to_add_mel = temp_time_to_add_mel + msg.dict()['time']
                # not top voice end:
                elif ( msg.type == 'note_off' or (msg.type == 'note_on' and  msg.dict()['velocity'] == 0) ) and msg.dict()['note'] in watchlist:
                    note = msg.dict()['note']
                    watchlist.remove(note)
                    msg.time = msg.dict()['time'] + temp_time_to_add_acc
                    temp_time_to_add_acc = 0
                    tmp_accomp_block_list.append(msg)
                    temp_time_to_add_mel = temp_time_to_add_mel + msg.dict()['time']
                # top voice:
                else:
                    msg.time = msg.dict()['time'] + temp_time_to_add_mel
                    temp_time_to_add_acc = temp_time_to_add_acc + msg.dict()['time']
                    temp_time_to_add_mel = 0
                    tmp_melody_block_list.append(msg)
            melody_list.extend(tmp_melody_block_list)
            accomp_list.extend(tmp_accomp_block_list)
            print('tmp_melody_block_list:', len(tmp_melody_block_list))
            print('tmp_accomp_block_list:', len(tmp_accomp_block_list))

    melody_track = copy.deepcopy(track)
    accomp_track = copy.deepcopy(track)
    melody_track.clear()
    accomp_track.clear()
    # result_track = mido.MidiTrack()
    melody_track.extend(melody_list)
    accomp_track.extend(accomp_list)
    return melody_track, accomp_track

def midi_file_melody_only(midi_file_path):
    midi_file = mido.MidiFile(midi_file_path, clip=True)
    track = midi_file.tracks[0]
    resulted_track = single_track_upper_melody(track)

    midi_file.tracks[0] = resulted_track

    return midi_file

## Module 2: Starting the output:

### 2.1: MIDI file -> MIDI file (melody only):
def process(input_file_path,output_path):
    result = midi_file_melody_only(input_file_path)
    input_name = result.filename
    output_name = 'melody_only_'+ input_name
    result.save(output_path + output_name)

### 2.2: Folder of MIDI files -> Folder of MIDI files (melody only):
def find_name(string):
    index = string.rfind('/')
    return string[index + 1:]

def batch_process(input_folder_path, output_path):
    output_folder_name = 'melody_only_' + find_name(input_folder_path)
    output_folder_path = output_path + output_folder_name

    if os.path.exists(output_folder_path + '/'):
        shutil.rmtree(output_folder_path + '/')
    os.makedirs(output_folder_path + '/')

    for files in os.listdir(input_folder_path):
        input_file_path = input_folder_path + '/' + files
        input_file_name = find_name(input_file_path)
        result = midi_file_melody_only(input_file_path)
        result.filename = 'melody_only_' + input_file_name
        result.save(output_folder_path + '/' + result.filename)


## Module 3: Run this script

if __name__ == '__main__':
    input_folder_path = 'sample_MIDIs'
    output_path = './'
    batch_process(input_folder_path, output_path)
    print('Finished processing!')

