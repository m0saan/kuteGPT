import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from math import sqrt
from typing import Tuple, List, Optional
from src.config import Config


__all__ = ['InputEmbeddings', 'PositionalEncoding', 'Embeddings', 'FeedForward', 'AttentionHead',
           'MultiHeadAttention', 'RisidualConnection', 'EncoderBlock', 'Encoder', 'DecoderBlock',
           'Decoder', 'Generator']

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
        pe = torch.stack([odd_pe, even_pe], dim=-1).view(config.max_position_embeddings, config.hidden_size).unsqueeze(0)
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


class FeedForward(nn.Module):
    
    def __init__(self, config: Config) -> None:
        super().__init__()
        
        self.linear_1 = nn.Linear(in_features=config.hidden_size, out_features=config.intermediate_size)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(in_features=config.intermediate_size, out_features=config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_2(self.relu(self.linear_1(x)))
        x = self.dropout(x)
        return x


class AttentionHead(nn.Module):
    
    def __init__(self, config: Config) -> None:
        super().__init__()
        
        self.dim_k = config.hidden_size // config.num_attention_heads
        self.dim_v = self.dim_k
        self.linear_q = nn.Linear(in_features=config.hidden_size, out_features=self.dim_k)
        self.linear_k = nn.Linear(in_features=config.hidden_size, out_features=self.dim_k)
        self.linear_v = nn.Linear(in_features=config.hidden_size, out_features=self.dim_v)
        
    
    def forward(self,
                x_q: torch.Tensor,
                x_k: torch.Tensor,
                x_v: torch.Tensor, 
                mask: torch.Tensor = None) -> torch.Tensor:
        
        return AttentionHead._scaled_dot_product_attention(
            self.linear_q(x_q),
            self.linear_k(x_k),
            self.linear_v(x_v),
            mask=mask)
    
    @staticmethod
    def _scaled_dot_product_attention(query: torch.Tensor,
                                  key: torch.Tensor,
                                  value: torch.Tensor,
                                  mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        #(B, hidden_size, dim_k)x(B, hidden_size, dim_k)
        dim_k = query.shape[-1]
        attn_scores = torch.bmm(query, key.transpose(-2, -1)) / sqrt(dim_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill_(mask==0, float('-inf'))
        weights = F.softmax(attn_scores, dim=-1)
        return torch.bmm(weights, value), attn_scores
    

class MultiHeadAttention(nn.Module):
    
    def __init__(self, config: Config) -> None:
        super().__init__()
        
        assert config.hidden_size % config.num_attention_heads == 0, "d_model must be divisible by num_attention_heads"
        dim_k = config.hidden_size // config.num_attention_heads
        self.heads = nn.ModuleList([AttentionHead(config) for _ in range(config.num_attention_heads)])
        self.linear_out = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self,
                x_q: torch.Tensor,
                x_k: torch.Tensor,
                x_v: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        
        attention_scores = torch.cat([head(x_q, x_k, x_v)[0] for head in self.heads], dim=-1)
        return self.linear_out(self.dropout(attention_scores))

class RisidualConnection(nn.Module):
    
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=config.hidden_size, eps=config.eps)
        self.dropout = nn.Dropout(p=config.dropout)
        
    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        x = x + self.dropout(sublayer(self.layer_norm(x)))
        return x

class EncoderBlock(nn.Module):
    
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(config=config)
        self.feed_forward = FeedForward(config=config)
        self.risidual_1 = RisidualConnection(config=config)
        self.risidual_2 = RisidualConnection(config=config)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.risidual_1(x, lambda x: self.multi_head_attention(x, x, x, mask))
        x = self.risidual_2(x, self.feed_forward)
        return x
        
        

class Encoder(nn.Module):
    
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.encoder_blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.num_encoder_layers)])
        self.layer_norm = nn.LayerNorm(normalized_shape=config.hidden_size, eps=config.eps)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        for encoder in self.encoder_blocks:
            x = encoder(x, mask)
        return self.layer_norm(x)


class DecoderBlock(nn.Module):
    
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.cross_attention = MultiHeadAttention(config=config)
        self.multi_head_attention = MultiHeadAttention(config=config)
        self.feed_forward = FeedForward(config=config)
        self.risidual_1 = RisidualConnection(config=config)
        self.risidual_2 = RisidualConnection(config=config)
        self.risidual_3 = RisidualConnection(config=config)
        
        
    def forward(self,
                x: torch.Tensor,
                encoder_outpout: torch.Tensor,
                src_mask: torch.Tensor,
                target_mask: torch.Tensor) -> torch.Tensor:
        
        x = self.risidual_1(x, lambda x: self.multi_head_attention(x, x, x, target_mask))
        x = self.risidual_1(x, lambda: self.cross_attention(x, encoder_outpout, encoder_outpout, src_mask))
        x = self.risidual_3(x, self.feed_forward)
        return x


class Decoder(nn.Module):
    
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.decoder_blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_decoder_layers)])
        self.layer_norm = nn.LayerNorm(normalized_shape=config.hidden_size, eps=config.eps)
    
    def forward(self,
                x: torch.Tensor,
                encoder_output: torch.Tensor,
                src_mask: torch.Tensor,
                target_mask: torch.Tensor) -> torch.Tensor:
        
        for decoder in self.decoder_blocks:
            x = decoder(x, encoder_output, src_mask, target_mask)
        return self.layer_norm(x)


class Generator(nn.Module):
    
    def __init__(self, config: Config) -> None:
        super().__init__()
        
        self.linear = nn.Linear(in_features=config.hidden_size, out_features=config.vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.linear(x), dim=-1)
