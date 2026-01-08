import torch
import torch.nn as nn
import math
from pathlib import Path


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
    A transformer-based classifier for per-event sequence labeling.
    
    Takes sequences of event tensors (from Event.to_tensor()) and produces
    classification predictions for each event in the sequence. The model uses
    a transformer encoder followed by an MLP applied to each timestep.
    
    Output shape: (batch_size, seq_len, num_classes)
    
    This enables sequence labeling tasks where each event receives its own
    multi-label classification, rather than a single prediction for the entire sequence.
    """
    
    @classmethod
    def from_shape_file(cls, shape_file: str, num_classes: int = 800, hidden_dim: int = 512, 
                       auxiliary_data_dim: int = 0, **kwargs):
        """
        Create model with dimensions automatically inferred from shape.json.
        
        Args:
            shape_file: Path to shape.json metadata file
            num_classes: Number of output classes per event (default: 800)
            hidden_dim: Hidden dimension for transformer (default: 512)
            auxiliary_data_dim: Dimension of auxiliary data (default: 0)
            **kwargs: Additional model parameters (num_heads, num_transformer_layers, dropout)
            
        Returns:
            EventTransformerClassifier instance configured from shape metadata
            
        Example:
            >>> model = EventTransformerClassifier.from_shape_file('shape.json', num_classes=800)
            >>> # Model automatically has input_dim=76, procedure_dim=384
            >>> # Produces per-event predictions: (batch_size, seq_len, 800)
        """
        try:
            from model.shape_utils import extract_dimensions
        except ImportError:
            # Try relative import if running from different location
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from shape_utils import extract_dimensions
        
        # Extract dimensions from shape file
        input_dim, procedure_dim = extract_dimensions(shape_file)
        
        # Create model with extracted dimensions
        return cls(
            input_dim=input_dim,
            procedure_dim=procedure_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            auxiliary_data_dim=auxiliary_data_dim,
            **kwargs
        )
    
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
            num_classes: Number of output classes per event (default: 800)
            hidden_dim: Hidden dimension for transformer and MLP
            num_transformer_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            auxiliary_data_dim: Dimension of auxiliary data vector to concatenate with transformer output (default: 0)
            procedure_dim: Dimension of each procedure vector (default: 0, disabled if 0)
            
        Note:
            The model produces per-event predictions with shape (batch_size, seq_len, num_classes).
            This differs from sequence classification which produces (batch_size, num_classes).
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.auxiliary_data_dim = auxiliary_data_dim
        self.procedure_dim = procedure_dim
        
        # Projection layer to map (input_dim * 2) to hidden_dim
        # Input is concatenated: [feature_values, availability_mask]
        # This allows the model to learn different behaviors for missing features
        self.input_projection = nn.Linear(input_dim * 2, hidden_dim)
        
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
    
    @staticmethod
    def make_feature_mask(x: torch.Tensor, missing_feat_idx: torch.Tensor = None, text_feature_ranges: dict = None) -> torch.Tensor:
        """
        Create feature availability mask for the last timestep.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            missing_feat_idx: Tensor of feature indices to mask in the last timestep.
                             If None, all features are available (mask = all 1s)
            text_feature_ranges: Optional dict mapping text feature names to their index ranges.
                                Format: {'feature_name': (start_idx, end_idx)}
                                When a text feature is masked, all indices in its range are masked.
                                Example: {'diagnosis': (50, 70)} means indices 50-69 are diagnosis text embedding
        
        Returns:
            Boolean mask of shape (batch_size, seq_len, input_dim) where:
            - 1.0 indicates feature is available
            - 0.0 indicates feature is missing
            
        Note: Only the last timestep (t = seq_len - 1) will have masked features.
              All other timesteps have all features available.
              
              For text features, the entire extended/repeated embedding range is masked
              when the feature index falls within any text feature range.
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Start with all features available (mask = 1)
        mask = torch.ones_like(x)
        
        if missing_feat_idx is not None:
            # Convert to tensor if list
            if isinstance(missing_feat_idx, list):
                missing_feat_idx = torch.tensor(missing_feat_idx, device=x.device)
            
            # For each missing feature index, determine what to mask
            for feat_idx in missing_feat_idx:
                feat_idx_val = feat_idx.item() if isinstance(feat_idx, torch.Tensor) else feat_idx
                
                # Check if this index belongs to a text feature range
                is_text_feature = False
                if text_feature_ranges is not None:
                    for feature_name, (start_idx, end_idx) in text_feature_ranges.items():
                        if start_idx <= feat_idx_val < end_idx:
                            # This is part of a text feature - mask the entire range
                            mask[:, -1, start_idx:end_idx] = 0.0
                            is_text_feature = True
                            break
                
                # If not a text feature, mask just this single index
                if not is_text_feature:
                    mask[:, -1, feat_idx_val] = 0.0
        
        return mask
    
    @staticmethod
    def apply_feature_missingness(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply feature missingness to input and augment with availability mask.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Availability mask of shape (batch_size, seq_len, input_dim)
                  where 1 = available, 0 = missing
        
        Returns:
            Augmented tensor of shape (batch_size, seq_len, input_dim * 2)
            where the first half is zero-filled features and second half is the mask.
            
        Implementation:
            x_filled = x * mask  # Zero out missing features
            x_aug = concat([x_filled, mask], dim=-1)  # Concatenate features and mask
            
        Why zero-fill:
            - Prevents information leakage from unavailable features
            - Model learns from the mask which features are present/absent
            - Training uses same logic to avoid train/test mismatch
        """
        # Zero out missing features to prevent information leakage
        x_filled = x * mask
        
        # Concatenate filled features with availability mask
        # Shape: (batch_size, seq_len, input_dim * 2)
        x_aug = torch.cat([x_filled, mask], dim=-1)
        
        return x_aug
    
    @staticmethod
    def create_padding_mask(sequences: torch.Tensor, pad_value: float = 0.0) -> torch.Tensor:
        """
        Create padding mask from sequences.
        
        Args:
            sequences: Input tensor of shape (batch_size, seq_len, *) or (batch_size, seq_len)
            pad_value: Value used for padding (default: 0.0)
        
        Returns:
            Boolean mask of shape (batch_size, seq_len) where True indicates padding
        """
        # Check if all features in a position are equal to pad_value
        if sequences.dim() == 3:
            # (batch_size, seq_len, feature_dim) -> check if all features are pad_value
            mask = (sequences == pad_value).all(dim=-1)
        else:
            # (batch_size, seq_len) -> direct comparison
            mask = (sequences == pad_value)
        return mask
    
    @staticmethod
    def create_padding_mask_from_lengths(seq_lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """
        Create padding mask from sequence lengths.
        
        Args:
            seq_lengths: Tensor of shape (batch_size,) containing actual sequence lengths
            max_len: Maximum sequence length (padded length)
        
        Returns:
            Boolean mask of shape (batch_size, max_len) where True indicates padding
        """
        batch_size = seq_lengths.size(0)
        # Create position indices: (1, max_len)
        positions = torch.arange(max_len, device=seq_lengths.device).unsqueeze(0)
        # Expand seq_lengths: (batch_size, 1)
        seq_lengths = seq_lengths.unsqueeze(1)
        # mask[i, j] = True if j >= seq_lengths[i]
        mask = positions >= seq_lengths
        return mask
    
    def forward(self, x, procedure_x=None, auxiliary_data=None, src_key_padding_mask=None, procedure_padding_mask=None, missing_feat_idx=None, text_feature_ranges=None, feature_mask=None) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim) containing event vectors
            procedure_x: Optional procedure tensor of shape (batch_size, proc_seq_len, procedure_dim)
            auxiliary_data: Optional auxiliary data of shape (batch_size, auxiliary_data_dim)
            src_key_padding_mask: Optional mask of shape (batch_size, seq_len) where True indicates padding
            procedure_padding_mask: Optional mask of shape (batch_size, proc_seq_len) where True indicates padding
            missing_feat_idx: Optional tensor/list of feature indices to mask in the last timestep.
                            If None, all features are assumed available.
            text_feature_ranges: Optional dict mapping text feature names to (start_idx, end_idx) tuples.
                               When a feature index in missing_feat_idx falls in a text range,
                               the entire range is masked.
            feature_mask: Optional pre-computed feature mask of shape (batch_size, seq_len, input_dim).
                         If provided, overrides missing_feat_idx and text_feature_ranges.
        
        Returns:
            Output logits of shape (batch_size, seq_len, num_classes)
            
            Each timestep receives its own classification prediction, enabling
            sequence labeling tasks. When computing loss, remember to mask out
            padded positions using src_key_padding_mask.
            
        Example:
            >>> model = EventTransformerClassifier(input_dim=76, num_classes=800)
            >>> x = torch.randn(4, 10, 76)  # batch_size=4, seq_len=10
            >>> 
            >>> # Define text feature ranges (e.g., diagnosis text embedded at indices 50-70)
            >>> text_ranges = {'diagnosis': (50, 70), 'procedure_desc': (70, 90)}
            >>> 
            >>> # Mask feature 55 (part of diagnosis text) - entire range 50-70 gets masked
            >>> output = model(x, missing_feat_idx=[55], text_feature_ranges=text_ranges)
        """
        # Handle feature-level missingness
        # Create availability mask and augment input with missingness indicators
        if feature_mask is None:
            feature_mask = self.make_feature_mask(x, missing_feat_idx, text_feature_ranges)
        x = self.apply_feature_missingness(x, feature_mask)
        
        # Project augmented input to hidden dimension
        # x shape: (batch_size, seq_len, input_dim * 2) -> (batch_size, seq_len, hidden_dim)
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer encoder with padding mask
        # x shape: (batch_size, seq_len, hidden_dim)
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Process procedures through separate transformer if provided
        if procedure_x is not None and self.procedure_dim > 0:
            # procedure_x shape: (batch_size, proc_seq_len, procedure_dim)
            # Project to hidden dimension
            procedure_x = self.procedure_projection(procedure_x)
            # Add positional encoding
            procedure_x = self.procedure_pos_encoder(procedure_x)
            # Pass through procedure transformer with padding mask
            procedure_x = self.procedure_transformer(procedure_x, src_key_padding_mask=procedure_padding_mask)
            
            # Pool procedures to single vector per sample
            if procedure_padding_mask is not None:
                proc_seq_lengths = (~procedure_padding_mask).sum(dim=1)
                proc_indices = (proc_seq_lengths - 1).clamp(min=0)
                batch_size = procedure_x.size(0)
                procedure_pooled = procedure_x[torch.arange(batch_size), proc_indices]  # (batch_size, hidden_dim)
            else:
                procedure_pooled = procedure_x.mean(dim=1)  # (batch_size, hidden_dim)
            
            # Expand procedure representation to match event sequence length
            # (batch_size, hidden_dim) -> (batch_size, seq_len, hidden_dim)
            seq_len = x.size(1)
            procedure_expanded = procedure_pooled.unsqueeze(1).expand(-1, seq_len, -1)
            
            # Concatenate with main transformer output at each timestep
            x = torch.cat([x, procedure_expanded], dim=-1)  # (batch_size, seq_len, hidden_dim*2)
        
        # Concatenate auxiliary data if provided
        if auxiliary_data is not None:
            # Ensure auxiliary_data is 2D
            if auxiliary_data.dim() == 1:
                auxiliary_data = auxiliary_data.unsqueeze(0)
            # Broadcast if needed
            if auxiliary_data.size(0) == 1 and x.size(0) > 1:
                auxiliary_data = auxiliary_data.expand(x.size(0), -1)
            # Expand to match sequence length
            seq_len = x.size(1)
            auxiliary_expanded = auxiliary_data.unsqueeze(1).expand(-1, seq_len, -1)
            x = torch.cat([x, auxiliary_expanded], dim=-1)
        
        # Pass through MLP classifier for each timestep
        # x shape: (batch_size, seq_len, hidden_dim + procedure_hidden + auxiliary_data_dim)
        # Reshape for MLP: (batch_size * seq_len, feature_dim)
        batch_size, seq_len, feature_dim = x.shape
        x_flat = x.reshape(batch_size * seq_len, feature_dim)
        
        # Apply MLP
        logits_flat = self.mlp(x_flat)  # (batch_size * seq_len, num_classes)
        
        # Reshape back to sequence: (batch_size, seq_len, num_classes)
        logits = logits_flat.reshape(batch_size, seq_len, self.num_classes)
        
        return logits
