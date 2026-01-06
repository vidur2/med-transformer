import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding to input.
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class EventTransformerClassifier(nn.Module):
    """
    A transformer-based classifier for event categorization.
    
    Takes sequences of event tensors (from Event.to_tensor()) and classifies them
    into one of 800 categories using a transformer encoder followed by an MLP.
    """
    
    def __init__(
        self,
        input_dim: int = 387,
        num_classes: int = 800,
        hidden_dim: int = 512,
        num_transformer_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        auxiliary_data_dim: int = 0,
        procedure_dim: int = 0,
    ):
        """
        Initialize the EventTransformerClassifier.
        
        Args:
            input_dim: Dimension of each event vector (default: 387 from Event.to_tensor())
            num_classes: Number of output classes (default: 800)
            hidden_dim: Hidden dimension for transformer and MLP
            num_transformer_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            auxiliary_data_dim: Dimension of auxiliary data vector to concatenate with transformer output (default: 0)
            procedure_dim: Dimension of each procedure vector (default: 0, disabled if 0)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.auxiliary_data_dim = auxiliary_data_dim
        self.procedure_dim = procedure_dim
        
        # Projection layer to map input_dim to hidden_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding for events
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)
        
        # Transformer encoder for main events
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
        )
        
        # Procedure transformer (if procedure_dim > 0)
        if procedure_dim > 0:
            self.procedure_projection = nn.Linear(procedure_dim, hidden_dim)
            self.procedure_pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)
            procedure_encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.procedure_transformer = nn.TransformerEncoder(
                procedure_encoder_layer,
                num_layers=num_transformer_layers,
            )
        
        # MLP classifier head
        # Input dimension includes event transformer output + procedure transformer output + auxiliary data
        mlp_input_dim = hidden_dim + auxiliary_data_dim
        if procedure_dim > 0:
            mlp_input_dim += hidden_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    @classmethod
    def prepare_input(cls, event_tensor: torch.Tensor, auxiliary_data: torch.Tensor = None) -> torch.Tensor:
        """
        Prepare input for the forward method.
        
        Args:
            event_tensor: Output from Event.to_tensor() of shape (batch_size, input_dim)
                         or batch of events of shape (batch_size, seq_len, input_dim)
            auxiliary_data: Not used anymore - auxiliary data should be passed directly to forward()
        
        Returns:
            Event tensor of shape (batch_size, seq_len, input_dim)
        """
        # Ensure event_tensor is 3D (batch_size, seq_len, input_dim)
        if event_tensor.dim() == 2:
            # Single event per batch: (batch_size, input_dim) -> (batch_size, 1, input_dim)
            event_tensor = event_tensor.unsqueeze(1)
        
        return event_tensor
    
    def forward(self, x, procedure_x=None, auxiliary_data=None, src_key_padding_mask=None, procedure_padding_mask=None) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim) containing event vectors
            procedure_x: Optional procedure tensor of shape (batch_size, proc_seq_len, procedure_dim)
            auxiliary_data: Optional auxiliary data of shape (batch_size, auxiliary_data_dim)
            src_key_padding_mask: Optional mask of shape (batch_size, seq_len) where True indicates padding
            procedure_padding_mask: Optional mask of shape (batch_size, proc_seq_len) where True indicates padding
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Project input to hidden dimension
        # x shape: (batch_size, seq_len, input_dim) -> (batch_size, seq_len, hidden_dim)
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer encoder with padding mask
        # x shape: (batch_size, seq_len, hidden_dim)
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Pooling: use last non-padded token if mask provided, otherwise mean pooling
        if src_key_padding_mask is not None:
            # Get sequence lengths from mask (count non-padded tokens)
            seq_lengths = (~src_key_padding_mask).sum(dim=1)  # (batch_size,)
            # Get last non-padded token for each sequence
            batch_size = x.size(0)
            indices = (seq_lengths - 1).clamp(min=0)  # (batch_size,)
            x = x[torch.arange(batch_size), indices]  # (batch_size, hidden_dim)
        else:
            # Global average pooling over sequence dimension
            x = x.mean(dim=1)  # (batch_size, hidden_dim)
        
        # Process procedures through separate transformer if provided
        if procedure_x is not None and self.procedure_dim > 0:
            # procedure_x shape: (batch_size, proc_seq_len, procedure_dim)
            # Project to hidden dimension
            procedure_x = self.procedure_projection(procedure_x)
            # Add positional encoding
            procedure_x = self.procedure_pos_encoder(procedure_x)
            # Pass through procedure transformer with padding mask
            procedure_x = self.procedure_transformer(procedure_x, src_key_padding_mask=procedure_padding_mask)
            
            # Pooling for procedures
            if procedure_padding_mask is not None:
                proc_seq_lengths = (~procedure_padding_mask).sum(dim=1)
                proc_indices = (proc_seq_lengths - 1).clamp(min=0)
                procedure_x = procedure_x[torch.arange(batch_size), proc_indices]
            else:
                procedure_x = procedure_x.mean(dim=1)
            
            # Concatenate with main transformer output
            x = torch.cat([x, procedure_x], dim=1)
        
        # Concatenate auxiliary data if provided
        if auxiliary_data is not None:
            # Ensure auxiliary_data is 2D
            if auxiliary_data.dim() == 1:
                auxiliary_data = auxiliary_data.unsqueeze(0)
            # Broadcast if needed
            if auxiliary_data.size(0) == 1 and x.size(0) > 1:
                auxiliary_data = auxiliary_data.expand(x.size(0), -1)
            x = torch.cat([x, auxiliary_data], dim=1)
        
        # Pass through MLP classifier
        # x shape: (batch_size, hidden_dim + procedure_hidden + auxiliary_data_dim) -> (batch_size, num_classes)
        logits = self.mlp(x)
        
        return logits
