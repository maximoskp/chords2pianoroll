from transformers import RobertaConfig, RobertaForMaskedLM, Trainer, TrainingArguments, RobertaTokenizerFast, DataCollatorForLanguageModeling
import evaluate
from torch.cuda import is_available as cuda_available, is_bf16_supported
from tqdm import tqdm
import pickle

MAX_LENGTH = 1024

# Load the trained tokenizer
# tokenizer = RobertaTokenizerFast.from_pretrained('../data/chroma_tokenizer_1/' , max_len=MAX_LENGTH)
tokenizer = RobertaTokenizerFast.from_pretrained('../../data/chroma_wordlevel_tokenizer/' , max_len=MAX_LENGTH)

# Data collator for MLM (will handle random masking)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# Roberta-MED
model_config = RobertaConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=512,
    num_hidden_layers=8,
    num_attention_heads=8,
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    mask_token_id=tokenizer.mask_token_id,
    max_position_embeddings=MAX_LENGTH,
)

model = RobertaForMaskedLM(model_config)

with open('../../data/chroma_dataset.pickle', "rb") as input_file:
    chroma_dataset = pickle.load(input_file)

train_dataset = chroma_dataset['train_dataset']
test_dataset = chroma_dataset['test_dataset']

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        print(f"is indeed tuple")
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    # import pdb; pdb.set_trace()
    return logits.argmax(dim=-1)

metric = evaluate.load("accuracy")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]
    return metric.compute(predictions=preds, references=labels)

# Trainer
USE_CUDA = cuda_available()
if not cuda_available():
    print(f"NO 16-floats used")
    FP16 = FP16_EVAL = BF16 = BF16_EVAL = False
elif is_bf16_supported():
    BF16 = BF16_EVAL = True
    FP16 = FP16_EVAL = False
    print(f"BF16_float used")
else:
    BF16 = BF16_EVAL = False
    FP16 = FP16_EVAL = True
    (f"FP16_float used")
print(f"Using CUDA as {USE_CUDA}")

training_config = TrainingArguments(
    output_dir="../../data/chroma_mlm_medium_3e-4_100",
    overwrite_output_dir=False,
    do_train=True,
    do_eval=True,
    do_predict=False,
    evaluation_strategy="steps",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=8,  # for 1 device results in 512 bsz,
    eval_accumulation_steps=1,
    # gradient_accumulation_steps=8, # for 2 devices results in 1024 bsz,
    learning_rate=3e-4,
    weight_decay=0.01,
    max_grad_norm=0,
    warmup_steps=1300,
    log_level="debug",
    lr_scheduler_type="linear",
    logging_strategy="steps",
    max_steps=100*(673191//(64*8)),
    # num_train_epochs=10,
    logging_steps=128,
    logging_dir='./logs',
    save_strategy="steps",
    save_steps=256,
    save_total_limit=5,  # keeps 5 checkpoints only
    no_cuda=not USE_CUDA,
    seed=1993,
    fp16=FP16,
    fp16_full_eval=FP16_EVAL,
    bf16=BF16,
    bf16_full_eval=BF16_EVAL,
    load_best_model_at_end=True,
    report_to=["tensorboard"],
    gradient_checkpointing=False, # when True saves memory in the expense of slower backward
)

trainer = Trainer(
    model=model,
    args=training_config,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    callbacks=None
)