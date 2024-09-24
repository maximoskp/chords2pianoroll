from pathlib import Path
from torch.utils.data import DataLoader
from torchtoolkit.data import create_subsets
from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetTok, DataCollator

saved_tokenizer_path = 'data/maestro_small_tokenizer.json'
tokenizer = REMI(params=Path(saved_tokenizer_path))

path_to_tokens_bpe = 'data/maestro_small_BPE/'
tokens_paths = list(Path(path_to_tokens_bpe).glob("**/*.json"))
dataset = DatasetTok(
    tokens_paths, max_seq_len=512, min_seq_len=384, one_token_stream=False,
)

subset_train, subset_valid = create_subsets(dataset, [0.3])
collator = DataCollator(tokenizer["PAD_None"], tokenizer["BOS_None"], tokenizer["EOS_None"], copy_inputs_as_labels=True)

dataloader_test = DataLoader(subset_valid, batch_size=16, collate_fn=collator)