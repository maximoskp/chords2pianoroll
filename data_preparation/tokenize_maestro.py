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
USE_TEMPOS = True
USE_TIME_SIGNATURE = False
USE_PROGRAMS = False
NB_TEMPOS = 32
TEMPO_RANGE = (50, 200)  # (min_tempo, max_tempo)
TOKENIZER_PARAMS = {
    "pitch_range": PITCH_RANGE,
    "beat_res": BEAT_RES,
    "nb_velocities": NB_VELOCITIES,
    "special_tokens": SPECIAL_TOKENS,
    "use_chords": USE_CHORDS,
    "use_rests": USE_RESTS,
    "use_tempos": USE_TEMPOS,
    "use_time_signatures": USE_TIME_SIGNATURE,
    "use_programs": USE_PROGRAMS,
    "nb_tempos": NB_TEMPOS,
    "tempo_range": TEMPO_RANGE,
}
config = TokenizerConfig(**TOKENIZER_PARAMS)
tokenizer = REMI(config)

### full dataset
## small prefix
# prefix = "/data/scratch/efthygeo/maestro/maestro-v3.0.0"
prefix = "/data/scratch/efthygeo/maestro/maestro-v3.0.0"


path_to_dataset = prefix
path_to_tokens = f'{prefix}_noBPE/'
path_to_tokens_bpe = f'{prefix}_BPE/'
path_to_tokenizer_config = f'{prefix}_tokenizer.json'

## smaller dataset
# path_to_dataset = 'data/maestro_small/'
# path_to_tokens = 'data/maestro_small_noBPE/'
# path_to_tokens_bpe = 'data/maestro_small_BPE/'
# path_to_tokenizer_config = 'data/maestro_small_tokenizer.json'

# Tokenize a whole dataset and save it at Json files
midi_paths = list(Path(path_to_dataset).glob("**/*.midi"))
# import pdb; pdb.set_trace()

data_augmentation_offsets = [1, 1, 1]  # data augmentation on 2 pitch octaves, 1 velocity and 1 duration values
tokenizer.tokenize_midi_dataset(midi_paths, Path(path_to_tokens),
                                data_augment_offsets=data_augmentation_offsets)

# Constructs the vocabulary with BPE, from the token files
tokenizer.learn_bpe(
    vocab_size=10000,
    tokens_paths=list(Path(path_to_tokens).glob("**/*.json")),
    start_from_empty_voc=False,
)

# Saving our tokenizer, to retrieve it back later with the load_params method
tokenizer.save_params(Path(path_to_tokenizer_config))

# Applies BPE to the previous tokens
tokenizer.apply_bpe_to_dataset(Path(path_to_tokens), Path(path_to_tokens_bpe))