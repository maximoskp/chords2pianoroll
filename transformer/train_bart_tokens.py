from transformers import BartForConditionalGeneration, BartConfig
from transformers import RobertaTokenizerFast
import torch
from torch.utils.data import DataLoader

from models import MelCAT_base_tokens
from dataset_utils import LiveMelCATDataset, MelCATCollator

from torch.nn import CrossEntropyLoss

import os
import numpy as np
import csv

from tqdm import tqdm

load_saved = True

MAX_LENGTH = 1024

roberta_tokenizer_midi = RobertaTokenizerFast.from_pretrained('/media/datadisk/data/pretrained_models/midi_mlm_tiny/midi_wordlevel_tokenizer')

bart_config = BartConfig(
    vocab_size=roberta_tokenizer_midi.vocab_size,
    pad_token_id=roberta_tokenizer_midi.pad_token_id,
    bos_token_id=roberta_tokenizer_midi.bos_token_id,
    eos_token_id=roberta_tokenizer_midi.eos_token_id,
    decoder_start_token_id=roberta_tokenizer_midi.bos_token_id,
    forced_eos_token_id=roberta_tokenizer_midi.eos_token_id,
    max_position_embeddings=MAX_LENGTH,
    encoder_layers=8,
    encoder_attention_heads=8,
    encoder_ffn_dim=4096,
    decoder_layers=8,
    decoder_attention_heads=8,
    decoder_ffn_dim=4096,
    d_model=256,
    encoder_layerdrop=0.3,
    decoder_layerdrop=0.3,
    dropout=0.3
)


dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# dev = torch.device("cpu")
model = MelCAT_base_tokens(bart_config, gpu=0)

if load_saved:
    checkpoint = torch.load('saved_models/bart_base/bart_base.pt', weights_only=True)
    model.load_state_dict(checkpoint)

# # Freeze the parameters of pretrained models
# for param in model.text_encoder.parameters():
#     param.requires_grad = False

# for param in model.chroma_encoder.parameters():
#     param.requires_grad = False

# for param in model.midi_encoder.parameters():
#     param.requires_grad = False

# params = list(model.bart_model.parameters()) + list( model.text_lstm.parameters())
# optimizer = torch.optim.AdamW( params, lr=0.00001)
optimizer = torch.optim.AdamW( model.parameters(), lr=0.00001)

loss_fct = CrossEntropyLoss(ignore_index=roberta_tokenizer_midi.pad_token_id)

midifolder = '/media/datadisk/datasets/GiantMIDI-PIano/midis_v1.2/midis'
# midifolder = '/media/datadisk/data/Giant_PIano/'
dataset = LiveMelCATDataset(midifolder, segment_size=40, resolution=4, max_seq_len=1024, only_beginning=True)

custom_collate_fn = MelCATCollator(max_seq_lens=dataset.max_seq_lengths, padding_values=dataset.padding_values)

dataloader = DataLoader(dataset, batch_size=4, collate_fn=custom_collate_fn, drop_last=True)

save_name = 'bart_tokens'

# keep best validation loss for saving
best_val_loss = np.inf
save_dir = 'saved_models/' + save_name + '/'
transformer_path = save_dir + save_name + '.pt'
os.makedirs(save_dir, exist_ok=True)

# save results
os.makedirs('results', exist_ok=True)
results_path = 'results/' + save_name + '.csv'
result_fields = ['iteration', 'train_loss', 'tran_acc']
with open( results_path, 'w' ) as f:
    writer = csv.writer(f)
    writer.writerow( result_fields )

for epoch in range(1000):
    train_loss = 0
    running_loss = 0
    batch_num = 0
    running_accuracy = 0
    train_accuracy = 0
    with tqdm(dataloader, unit='batch') as tepoch:
        tepoch.set_description(f"Epoch {epoch} | trn")
        for b in tepoch:
            # shift accomp
            shifted_accomp = {
                'input_ids': b['accomp']['input_ids'].new_zeros(b['accomp']['input_ids'].shape),
                'attention_mask': b['accomp']['attention_mask'].new_zeros(b['accomp']['attention_mask'].shape)
            }
            shifted_accomp['input_ids'][:, 1:] = b['accomp']['input_ids'][:, :-1].clone()  # Shift by one
            shifted_accomp['attention_mask'][:, 1:] = b['accomp']['attention_mask'][:, :-1].clone()  # Shift by one
            shifted_accomp['input_ids'][:, 0] = roberta_tokenizer_midi.bos_token_id  # Add start token
            shifted_accomp['attention_mask'][:, 0] = 1  # Add attention at start

            optimizer.zero_grad()
            
            output = model( b['melody'], b['chroma'], shifted_accomp, b['accomp']['input_ids'])
            logits = output.logits
            target_ids = b['accomp']['input_ids'].to(dev).contiguous()  # Shifted target sequence
            # Flatten the logits and target for the loss calculation
            logits_flat = logits.view(-1, logits.size(-1))
            target_flat = target_ids.view(-1)
            # Compute the cross-entropy loss (ignoring padding tokens)
            loss = loss_fct(logits_flat, target_flat)
            # loss = output.loss

            loss.backward()
            optimizer.step()

            # update loss
            batch_num += 1
            running_loss += loss.item()
            train_loss = running_loss/batch_num
            # accuracy
            prediction = logits.argmax(dim=2, keepdim=True).squeeze()
            # print('prediction.shape:', prediction.shape)
            # print('target_ids.shape:', target_ids.shape)
            # print('shifted_accomp[attention_mask].shape:', shifted_accomp['attention_mask'].shape)
            if (target_ids != roberta_tokenizer_midi.pad_token_id).sum().item() > 0:
                # running_accuracy += (prediction[prediction != roberta_tokenizer_midi.pad_token_id] == target_ids[prediction != roberta_tokenizer_midi.pad_token_id]).sum().item()/(prediction != roberta_tokenizer_midi.pad_token_id).sum().item()
                running_accuracy += (prediction[target_ids != roberta_tokenizer_midi.pad_token_id] == target_ids[target_ids != roberta_tokenizer_midi.pad_token_id]).sum().item()/(target_ids != roberta_tokenizer_midi.pad_token_id).sum().item()
            else:
                running_accuracy += 0
            train_accuracy = running_accuracy/batch_num
            torch.set_printoptions(threshold=10_000)
            tepoch.set_postfix(loss=train_loss, accuracy=train_accuracy) # tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
            if batch_num % 100 == 0:
                if best_val_loss > train_loss:
                    print('saving!')
                    best_val_loss = train_loss
                    torch.save(model.state_dict(), transformer_path)
                with open( results_path, 'a' ) as f:
                    writer = csv.writer(f)
                    writer.writerow( [epoch, train_loss, train_accuracy] )
        if best_val_loss > train_loss:
            print('saving!')
            best_val_loss = train_loss
            torch.save(model.state_dict(), transformer_path)
        with open( results_path, 'a' ) as f:
            writer = csv.writer(f)
            writer.writerow( [epoch, train_loss, train_accuracy] )
