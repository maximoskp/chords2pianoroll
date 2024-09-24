import numpy as np
from BinaryTokenizer import BinaryTokenizer
from miditok import REMI, TokenizerConfig
from miditoolkit import MidiFile
from pathlib import Path

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

prefix = 'data'
path_to_dataset = f'{prefix}/midis/'
path_to_tokens = f'{prefix}/midi_tokens/'
path_to_tokenizer_config = f'{prefix}/midi_tokenizer.json'

midi_paths = list(Path(path_to_dataset).glob("**/*.mid"))
tokenizer.tokenize_midi_dataset(midi_paths, Path(path_to_tokens), data_augment_offsets=None)