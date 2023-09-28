import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from math import sqrt
from typing import Tuple, List, Optional

from src.config import Config
from layers import (
    InputEmbeddings,
    PositionalEncoding,
    Embeddings,
    Encoder,
    Decoder,
    Generator
)

__all__ = ['Transformer', 'make_transformer']

class Transformer(nn.Module):
    """
    Transformer model for sequence to sequence translation.
    """

    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 generator: Generator,
                 src_embeddings: Embeddings,
                 target_embeddings: Embeddings
                 ) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.src_embeddings = src_embeddings
        self.target_embeddings = target_embeddings

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        src_embed = self.src_embeddings(src)
        return self.encoder(src_embed, src_mask)
    
    def decode(self,
               encoder_output: torch.Tensor,
               src_mask: torch.Tensor,
               target: torch.tensor,
               target_mask: torch.Tensor) -> torch.Tensor:
        
        target_embed = self.target_embeddings(target)
        return self.decoder(target_embed, encoder_output, src_mask, target_mask)
    
    def generate(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.generator(hidden_state)

    def forward(self, src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pass
    
    
    
def make_transformer(config: Config) -> Transformer:
    # create the input embedding layer for src and target!
    src_input_embeddings = InputEmbeddings(config)
    tgt_input_embeddings = InputEmbeddings(config)

    # create the positional embedding layer for src and target!
    src_pos_embeddings = PositionalEncoding(config)
    tgt_pos_embeddings = PositionalEncoding(config)
    
    src_embedding = Embeddings(config, token_embeddings=src_input_embeddings, pos_embeddings=src_pos_embeddings)
    tgt_embedding = Embeddings(config, token_embeddings=tgt_input_embeddings, pos_embeddings=tgt_pos_embeddings)

    # create the encoder & decoder blocks!
    encoder = Encoder(config)
    decoder = Decoder(config)

    # create the projection layer
    generator = Generator(config)

    # initialize a transformer model!
    transformer = Transformer(encoder=encoder,
                             decoder=decoder,
                             generator=generator,
                             src_embeddings=src_embedding,
                             target_embeddings=tgt_embedding,
                             )
    
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.kaiming_uniform_(p)

    return transformer

if __name__ == '__main__':
    print(make_transformer(Config()))
