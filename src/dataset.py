import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets, DatasetDict
from tokenizers import Tokenizer

from typing import Dict, Union, Tuple

from tokenization.tokenizer_utils import extract_texts, get_tokenizers, TokenizerConfig

class EnArDataset(Dataset):

    def __init__(self,
                 dataset: DatasetDict,
                 src_tokenizer: Tokenizer,
                 target_tokenizer: Tokenizer,
                 src_lang: str,
                 target_lang: str,
                 model_max_length: int) -> None:
        super().__init__()
        self.dataset = dataset
        self.src_tokenizer = src_tokenizer
        self.target_tokenizer = target_tokenizer
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.model_max_length = model_max_length

        self.sos_token = torch.tensor([src_tokenizer.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([src_tokenizer.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([src_tokenizer.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: any) -> torch.Tensor:
        src_target = self.dataset[idx]
        src_text = src_target['translation'][self.src_lang]
        tgt_text = src_target['translation'][self.target_lang]

        encoder_input_tokens = self.src_tokenizer.encode(src_text).ids
        decoder_input_tokens = self.target_tokenizer.encode(tgt_text).ids

        num_pad_tokens_src = self.model_max_length - len(encoder_input_tokens) - 2
        num_pad_tokens_tgt = self.model_max_length - len(decoder_input_tokens) - 1

        assert num_pad_tokens_src < 0 and num_pad_tokens_tgt < 0, "Sentence too long!"

        encoder_inputs = torch.cat([self.sos_token,
                                    torch.tensor(encoder_input_tokens, dtype=torch.int64),
                                    self.eos_token,
                                    torch.tensor([self.pad_token] * num_pad_tokens_src)])

        decoder_inputs = torch.cat([self.sos_token,
                                    torch.tensor(decoder_input_tokens, dtype=torch.int64),
                                    torch.tensor([self.pad_token] * num_pad_tokens_src)])
        
        label = torch.cat([torch.tensor(decoder_input_tokens, dtype=torch.int64),
                           self.eos_token,
                           torch.tensor([self.pad_token] * num_pad_tokens_src)])

        assert encoder_inputs.size(0) == self.model_max_length
        assert decoder_inputs.size(0) == self.model_max_length
        assert label.size(0) == self.model_max_length

        return {'encoder_inputs': encoder_inputs,
                'decoder_inputs': decoder_inputs,
                'label': label,
                'encoder_mask': ((encoder_inputs != self.pad_token)[None:, ...][None:, ...]).int(),
                'decoder_mask': (decoder_inputs != self.pad_token[None:, ...][None:, ...]).int() & causal_mask(self.model_max_length),
                'src_text': src_text,
                'tgt_text': tgt_text
               }

def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0

def load_combined_dataset() -> DatasetDict:
    raw_dataset = load_dataset('opus100', name='ar-en')
    return raw_dataset, concatenate_datasets([raw_dataset['train'], raw_dataset['validation'], raw_dataset['test']])


def get_datasets(config, raw_dataset, src_tokenizer, target_tokenizer) -> Tuple[EnArDataset, EnArDataset, EnArDataset]:

    train_ds = EnArDataset(
        raw_dataset['train'],
        src_tokenizer,
        target_tokenizer,
        src_lang=config.source_lang, 
        target_lang=config.target_lang,
        model_max_length=config.model_max_length)

    validation_ds = EnArDataset(
        raw_dataset['validation'],
        src_tokenizer,
        target_tokenizer,
        src_lang=config.source_lang, 
        target_lang=config.target_lang,
        model_max_length=config.model_max_length)

    test_ds = EnArDataset(
        raw_dataset['test'],
        src_tokenizer,
        target_tokenizer,
        src_lang=config.source_lang, 
        target_lang=config.target_lang,
        model_max_length=config.model_max_length)
    return (train_ds, validation_ds, test_ds)

def create_data_loaders(config, train_ds, validation_ds, test_ds) -> Dict[str, DataLoader]:
    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)
    valid_dl = DataLoader(validation_ds, batch_size=1, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=8, shuffle=False)
    return {'train_dl': train_dl, 'valid_dl': valid_dl, 'test_dl': test_dl}

def get_loaders_and_tokenizers(config) -> Dict[str, Union[DataLoader, Tokenizer]]:
    raw_dataset, combined_dataset = load_combined_dataset()
    # texts = extract_texts(combined_dataset)  # Assuming this is required somewhere else
    src_tokenizer, target_tokenizer = get_tokenizers(config, combined_dataset)
    
    train_ds, validation_ds, test_ds = get_datasets(config, raw_dataset, src_tokenizer, target_tokenizer)
    data_loaders = create_data_loaders(config, train_ds, validation_ds, test_ds)
    
    return {
        **data_loaders,
        'src_tokenizer': src_tokenizer,
        'target_tokenizer': target_tokenizer
    }

if __name__ == '__main__':
    print("Testing datasets.py")
    config = TokenizerConfig()
    print(get_loaders_and_tokenizers(config))