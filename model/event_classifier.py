import torch
import torch.nn as nn


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
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.auxiliary_data_dim = auxiliary_data_dim
        
        # Projection layer to map input_dim to hidden_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Transformer encoder
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
        
        # MLP classifier head
        # Input dimension is hidden_dim + auxiliary_data_dim * input_dim if auxiliary data is provided
        mlp_input_dim = hidden_dim + (auxiliary_data_dim * input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
            nn.Softmax()
        )
    
    @classmethod
    def prepare_input(cls, event_tensor: torch.Tensor, auxiliary_data: torch.Tensor = None) -> torch.Tensor:
        """
        Prepare input for the forward method by concatenating event tensor and auxiliary data.
        
        Args:
            event_tensor: Output from Event.to_tensor() of shape (batch_size, input_dim)
                         or batch of events of shape (batch_size, seq_len, input_dim)
            auxiliary_data: Optional auxiliary data tensor of shape (batch_size, auxiliary_data_dim)
                                   or (1, auxiliary_data_dim) which will be broadcasted to batch_size
        
        Returns:
            Concatenated tensor suitable for forward() of shape (batch_size, seq_len + auxiliary_data_dim, input_dim)
            or (batch_size, 1 + auxiliary_data_dim, input_dim) if event_tensor is 2D
        """
        # Ensure event_tensor is 3D (batch_size, seq_len, input_dim)
        if event_tensor.dim() == 2:
            # Single event per batch: (batch_size, input_dim) -> (batch_size, 1, input_dim)
            event_tensor = event_tensor.unsqueeze(1)
        
        if auxiliary_data is None:
            return event_tensor
        
        batch_size, seq_len, input_dim = event_tensor.shape
        
        if auxiliary_data.dim() == 1:
            # Single auxiliary data: (auxiliary_data_dim,) -> (1, auxiliary_data_dim)
            auxiliary_data = auxiliary_data.unsqueeze(0)
        
        # If auxiliary_data has batch_size of 1, broadcast it to match event_tensor's batch_size
        if auxiliary_data.size(0) == 1 and batch_size > 1:
            auxiliary_data = auxiliary_data.expand(batch_size, -1)
        
        # auxiliary_data is now (batch_size, auxiliary_data_dim)
        # Reshape to (batch_size, auxiliary_data_dim, 1) and pad to input_dim
        auxiliary_data = auxiliary_data.unsqueeze(2)  # (batch_size, auxiliary_data_dim, 1)
        
        # Pad auxiliary data to match input_dim
        # Use padding to extend from 1 to input_dim
        auxiliary_data = torch.nn.functional.pad(auxiliary_data, (0, input_dim - 1))  # (batch_size, auxiliary_data_dim, input_dim)
        
        # Concatenate along sequence dimension
        # event_tensor: (batch_size, seq_len, input_dim)
        # auxiliary_data: (batch_size, auxiliary_data_dim, input_dim)
        return torch.cat([event_tensor, auxiliary_data], dim=1)
    
    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len + auxiliary_data_dim, input_dim) where:
               - x[:, :-auxiliary_data_dim, :] contains event vectors from Event.to_tensor()
               - x[:, -auxiliary_data_dim:, :] contains auxiliary data (if auxiliary_data_dim > 0)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Extract auxiliary data if it exists
        auxiliary_data = None
        if self.auxiliary_data_dim > 0:
            # auxiliary_data shape: (batch_size, auxiliary_data_dim, input_dim)
            auxiliary_data = x[:, -self.auxiliary_data_dim:, :]
            # Flatten auxiliary data: (batch_size, auxiliary_data_dim * input_dim)
            auxiliary_data = auxiliary_data.reshape(x.size(0), -1)
            # Remove auxiliary data from x
            x = x[:, :-self.auxiliary_data_dim, :]
        
        # Project input to hidden dimension
        # x shape: (batch_size, seq_len, input_dim) -> (batch_size, seq_len, hidden_dim)
        x = self.input_projection(x)
        
        # Pass through transformer encoder
        # x shape: (batch_size, seq_len, hidden_dim)
        x = self.transformer_encoder(x)
        
        # Global average pooling over sequence dimension
        # x shape: (batch_size, seq_len, hidden_dim) -> (batch_size, hidden_dim)
        x = x.mean(dim=1)
        
        # Concatenate auxiliary data if provided
        if auxiliary_data is not None:
            x = torch.cat([x, auxiliary_data], dim=1)
        
        # Pass through MLP classifier
        # x shape: (batch_size, hidden_dim + auxiliary_data_dim * input_dim) -> (batch_size, num_classes)
        logits = self.mlp(x)
        
        return logits
