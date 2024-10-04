from tokenizers import ByteLevelBPETokenizer
import os

# Open and read the text file
with open('chroma_accompaniment_sentences.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Remove trailing newline characters if any
corpus = [line.strip() for line in lines]
print('num sentences: ', len(corpus))

# Initialize a Byte-Level BPE tokenizer
tokenizer = ByteLevelBPETokenizer()

# Train the tokenizer on the corpus
tokenizer.train_from_iterator(corpus, vocab_size=1000, min_frequency=2, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])

# Save the tokenizer to a directory
os.makedirs('../data/chroma_tokenizer_1', exist_ok=True)
tokenizer.save_model('../data/chroma_tokenizer_1')
