from pathlib import Path

from typing import Dict, Union, Tuple
from torch.utils.data import DataLoader

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tokenizers import SentencePieceBPETokenizer



from dataclasses import dataclass
from typing import Dict

__all__ = ['TokenizerConfig', 'extract_texts', 'get_tokenizers', 'get_or_build_tokenizer']

@dataclass
class TokenizerConfig:
    tokenizer_file: str = 'tokenizer_config_{0}.json'
    eos_token: str = '[SOS]'
    model_max_length: int = 512
    pad_token: str = '[PAD]'
    return_tensors: str = 'pt'
    separate_vocabs: bool = False
    source_lang: str = 'en'
    target_lang: str = 'ar'
    unk_token: str = '[UNK]'

def extract_texts(dataset, lang="en"):
    return (item["translation"][lang] for item in dataset)

def build_tokenizer(dataset, lang):
    special_tokens = ['[SOS]', '[EOS]', '[PAD]', '[UNK]']
    tokenizer = SentencePieceBPETokenizer(unk_token='[UNK]')
    tokenizer.train_from_iterator(extract_texts(dataset, lang),
                                  special_tokens=special_tokens,
                                  min_frequency=2,
                                  show_progress=True,)    
    return tokenizer

def get_or_build_tokenizer(config, dataset, lang):
    tokenizer_path = Path(config.tokenizer_file.format(lang))
    
    if tokenizer_path.exists():
        return Tokenizer.from_file(str(tokenizer_path))
    
    tokenizer = build_tokenizer(dataset, lang)
    tokenizer.save(str(tokenizer_path))
    return tokenizer


def get_tokenizers(config, combined_dataset) -> Tuple[Tokenizer, Tokenizer]:
    src_tokenizer = get_or_build_tokenizer(config, combined_dataset, config.source_lang)
    target_tokenizer = get_or_build_tokenizer(config, combined_dataset, config.target_lang)
    return src_tokenizer, target_tokenizer
