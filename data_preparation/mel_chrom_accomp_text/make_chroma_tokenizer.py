from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

# Initialize the BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# Define a trainer with vocab size and other parameters
trainer = BpeTrainer(vocab_size=1000, min_frequency=2, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

# Open and read the text file
with open('chroma_accompaniment_sentences.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Remove trailing newline characters if any
corpus = [line.strip() for line in lines]
print('num sentences: ', len(corpus))


# Train the tokenizer on the pre-tokenized corpus
tokenizer.train_from_iterator(corpus, trainer=trainer)

# Save the tokenizer to a directory
tokenizer.save("./chroma_tokenizer")
