import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from math import sqrt


@dataclass
class Config:
    vocab_size: int = 32000 # Vocabulary size
    hidden_size: int = 512  # Dimensionality of the model
    max_position_embeddings: int = 512
    intermediate_size: int = 2048    # Dimensionality of the feedforward network
    num_attention_heads: int = 8  # Number of attention heads
    num_encoder_layers: int = 6   # Number of encoder layers
    num_decoder_layers: int = 6   # Number of decoder layers
    dropout: float = 0.1  # Dropout rate
    eps: float = 1e-6  # Epsilon value for layer normalization
    
    # For source sequence
    src_vocab_size: int = 32000  # Source vocabulary size
    src_seq_len: int = 512       # Maximum length of source sequence

    # For target sequence
    tgt_vocab_size: int = 32000  # Target vocabulary size
    tgt_seq_len: int = 512       # Maximum length of target sequence


class Encoder(nn.Module):
    pass


class Decoder(nn.Module):
    pass


class Generator(nn.Module):
    pass


class InputEmbeddings(nn.Module):
    
    def __init__(self, config: Config) -> None:
        super().__init__()
        
        self.config = config
        self.embeddings = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embeddings(x) * sqrt(self.config.hidden_size)


class PositionalEncoding(nn.Module):
    
    def __init__(self, config: Config) -> None:
        super().__init__()
        
        self.config = config
        position = torch.arange(0, config.max_position_embeddings, dtype=torch.float32).unsqueeze(-1) # (seq_len, 1)
        denominator = torch.pow(1000, torch.arange(0, config.hidden_size, 2, dtype=torch.float32) / config.hidden_size)
        odd_pe = torch.cos(position / denominator)
        even_pe = torch.sin(position / denominator)
        pe = torch.stack([odd_pe, even_pe], dim=-1).view(config.vocab_size, config.hidden_size).unsqueeze(0)
        pe.requires_grad_(False)
        self.register_buffer('pos_embeddings', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pos_embeddings[:, :x.size(1), :]
    
class Embeddings(nn.Module):
    
    def __init__(self,
                 config: Config,
                 token_embeddings: InputEmbeddings,
                 pos_embeddings: PositionalEncoding) -> None:
        super().__init__()
        
        self.token_embeddings = token_embeddings
        self.pos_embeddings = pos_embeddings
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.eps)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.pos_embeddings(input_ids)
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings



class Transformer(nn.Module):
    """
    Transformer model for sequence to sequence translation.
    """

    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 generator: Generator,
                 src_embeddings: InputEmbeddings,
                 target_embeddings: InputEmbeddings,
                 pos_encodings: PositionalEncoding) -> None:
        super().__init__(Transformer, self)

        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.src_embeddings = src_embeddings
        self.target_embeddings = target_embeddings
        self.pos_encodings = pos_encodings

    def encode(self,
               src: torch.Tensor,
               src_mask: torch.Tensor) -> torch.Tensor:

        src_embeddings = self.src_embeddings(src)
        pos_encodings = self.pos_encodings(src)
        x = src_embeddings + pos_encodings
        return self.encoder(x, src_mask)
    
    def decode(self,
               encoder_output: torch.Tensor,
               src_mask: torch.Tensor,
               target: torch.Tensor,
               target_mask: torch.Tensor) -> torch.Tensor:
        
        target_embeddings = self.target_embeddings(target)
        pos_encodings = self.pos_encodings(target)
        x = target_embeddings + pos_encodings
        return self.decoder(x, encoder_output, src_mask, target_mask)

    def forward(self, src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pass
