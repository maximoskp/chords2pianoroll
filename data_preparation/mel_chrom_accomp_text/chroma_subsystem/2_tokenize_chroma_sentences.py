from transformers import RobertaConfig, RobertaForMaskedLM, Trainer, TrainingArguments, RobertaTokenizerFast, DataCollatorForLanguageModeling
import evaluate
from torch.cuda import is_available as cuda_available, is_bf16_supported
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
from datasets import Dataset
import pickle

MAX_LENGTH = 1024

# Load the trained tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained('../../data/chroma_tokenizer_1/' , max_len=MAX_LENGTH)

# Data collator for MLM (will handle random masking)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

sentences_file_path = '../../data/chroma_accompaniment_sentences.txt'

# Open and read the text file where each line is a new instance
with open(sentences_file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Remove trailing newline characters and strip extra spaces
sentences = [line.strip() for line in lines if line.strip()]

# Tokenize each sentence
tokenized_sentences = []
for sentence in tqdm(sentences[:10]):
    tokenized = tokenizer(
        sentence,
        max_length=MAX_LENGTH,
        truncation=True,
        return_overflowing_tokens=True,
        padding=False
    )
    
    # Collect all parts of the tokenized sentence
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized['attention_mask']
    for i_segment in range(len(input_ids)):
        tokenized_sentences.append( {
            'input_ids': input_ids[i_segment],
            'attention_mask': attention_mask[i_segment]
            } )

# Split the dataset (80% training, 20% testing)
train_texts, test_texts = train_test_split(tokenized_sentences, test_size=0.2, random_state=42)

# Extract input_ids and attention_mask for training and testing datasets
train_input_ids = [example['input_ids'] for example in train_texts]
train_attention_mask = [example['attention_mask'] for example in train_texts]

test_input_ids = [example['input_ids'] for example in test_texts]
test_attention_mask = [example['attention_mask'] for example in test_texts]

# Create a Hugging Face Dataset from tokenized inputs
def create_dataset(input_ids, attention_mask):
    dataset_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
    dataset = Dataset.from_dict(dataset_dict)
    return dataset

train_dataset = create_dataset(train_input_ids, train_attention_mask)
test_dataset = create_dataset(test_input_ids, test_attention_mask)

chroma_dataset = {
    'train_dataset': train_dataset,
    'test_dataset': test_dataset
}

with open('../../data/chroma_dataset.pickle', 'wb') as handle:
    pickle.dump(chroma_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)