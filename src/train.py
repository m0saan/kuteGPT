from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_loaders_and_tokenizers
from .transformers.model import make_transformer
from .dataset import causal_mask

def greedy_decode(model, encoder_input, encoder_mask, max_len, src_tokenizer, target_tokenizer, device):
    sos_token = target_tokenizer.token_to_id('[SOS]')
    eos_token = target_tokenizer.token_to_id('[EOS]')
    encoder_output = model.encode(encoder_input, encoder_mask)
    decoder_input = torch.tensor([sos_token], dtype=torch.int64).to(device)
    
    while decoder_input.size(1) <= max_len:
        decoder_mask = causal_mask(decoder_input.size(-1)).as_type(decoder_input).to(device)
        decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
        prob = model.generate(decoder_output[:, -1])
        pred_next = prob.argmax(dim=-1)
        if pred_next.item() == eos_token:
            break
        decoder_input = torch.cat([decoder_input, pred_next], dim=-1)
    


def one_epoch(model, loss_fn, optimizer, train_dl, validation_dl, src_tokenizer, target_tokenizer, write, device):
    
    model.train()
    for batch in train_dl:
        encoder_inputs = batch['encoder_inputs'].to(device)
        decoder_inputs = batch['decoder_inputs'].to(device)
        encoder_mask = batch['encoder_mask'].to(device)
        decoder_mask = batch['decoder_mask'].to(device)
        label = batch['label'].to(device)

        encoder_outputs = model.encode(encoder_inputs, encoder_mask)
        decoder_outputs = model.decode(encoder_outputs, encoder_mask, decoder_inputs, decoder_mask)
        model_outputs = model.generate(decoder_outputs)

        
        loss = loss_fn(model_outputs.view(-1, target_tokenizer.get_vocab_size()), label.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_dl.set_postfix({'loss': f'{loss.item():6.3f}'})
        
    if validation_dl is not None:
        model.eval()
        for batch in validation_dl:
            encoder_inputs = batch['encoder_inputs'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            
            model_output = greedy_decode(model, encoder_inputs, encoder_mask, 100, src_tokenizer, target_tokenizer, device)
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_output_text = target_tokenizer.decode(model_output.detach().cpu().numpy())
            
            write(f'----->source: {source_text}')
            write(f'----->target: {target_text}')
            write(f'----->prediction: {model_output_text}')
            


def fit(train_config, model_config, tokenizer_config):

    device = 'mps' if torch.backends.mps.is_available() else \
        'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    dls_and_tokenizers_dict = get_loaders_and_tokenizers(tokenizer_config)
    src_tokenizer = dls_and_tokenizers_dict['src_tokenizer']
    tagret_tokenizer = dls_and_tokenizers_dict['target_tokenizer']
    
    model = make_transformer(model_config).to(device) # TODO: create the config for the model.
    
    optimizer = optim.AdamW(model.parameters(), lr=train_config['lr'])
    loss_fn = nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(train_config['epochs']):
        train_dl = tqdm(dls_and_tokenizers_dict['train_dl'], desc=f'Processing epoch: {epoch:02d}')
        one_epoch(model, loss_fn, optimizer, train_dl, src_tokenizer, tagret_tokenizer, lambda msg: train_dl.write(msg), device)
            