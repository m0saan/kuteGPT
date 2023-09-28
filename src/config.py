from dataclasses import dataclass


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
