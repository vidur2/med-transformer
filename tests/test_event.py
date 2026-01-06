import sys
import os
sys.path.insert(0, 'preprocess_event_data')
sys.path.insert(0, 'model')

import torch
from abstract import DataPoint, Category
from proc_json import Event
from event_classifier import EventTransformerClassifier


class TestDataPoint(DataPoint):
    def __init__(self, category: str, count: int, is_active: bool, description: str):
        super().__init__()
        self.category = Category(category)
        self.count = count
        self.is_active = is_active
        self.description = description
    
    def dump_contents(self):
        return [self.category, self.count, self.is_active, self.description]


def test_event_tensor_consistency():
    """Test that Event produces consistent tensor dimensions regardless of text length."""
    # Test with different text inputs to ensure consistent dimensions
    test_data_1 = TestDataPoint(
        category="diagnosis",
        count=42,
        is_active=True,
        description="Short text"
    )
    
    test_data_2 = TestDataPoint(
        category="treatment",
        count=100,
        is_active=False,
        description="This is a much longer description with many more words to test embedding consistency"
    )
    
    test_data_3 = TestDataPoint(
        category="medication",
        count=0,
        is_active=True,
        description="Another different length text input for testing purposes and verification"
    )
    
    # Create Event objects and convert to tensors
    event_1 = Event(test_data_1)
    event_2 = Event(test_data_2)
    event_3 = Event(test_data_3)
    
    tensor_1 = event_1.to_tensor()
    tensor_2 = event_2.to_tensor()
    tensor_3 = event_3.to_tensor()
    
    # Verify dimensions are consistent
    print(f"Tensor 1 shape: {tensor_1.shape}")
    print(f"Tensor 2 shape: {tensor_2.shape}")
    print(f"Tensor 3 shape: {tensor_3.shape}")
    
    assert tensor_1.shape == tensor_2.shape == tensor_3.shape, \
        "Error: Tensors have different dimensions despite same DataPoint structure!"
    
    # Expected dimension: 3 floats (category, count, is_active) + 384 (embedding dimension)
    expected_dim = 3 + 384
    assert tensor_1.shape[0] == expected_dim, \
        f"Error: Expected dimension {expected_dim}, but got {tensor_1.shape[0]}"
    
    print(f"\n✓ All tests passed!")
    print(f"✓ Consistent output dimension: {tensor_1.shape[0]}")
    print(f"✓ Breakdown: 3 numeric features + 384 embedding features")


def test_model_examples():
    """Test various usage patterns of EventTransformerClassifier."""
    # Example usage
    batch_size = 4
    seq_len = 10
    input_dim = 387
    auxiliary_data_dim = 64
    
    # Create model with auxiliary data dimension
    model = EventTransformerClassifier(auxiliary_data_dim=auxiliary_data_dim)
    
    # Example 1: Single events with auxiliary data
    print("Example 1: Single events with auxiliary data")
    event_tensor = torch.randn(batch_size, input_dim)  # Output from Event.to_tensor()
    auxiliary_data = torch.randn(batch_size, auxiliary_data_dim)
    
    # Prepare input using class method
    x = EventTransformerClassifier.prepare_input(event_tensor)
    print(f"  Event tensor shape: {event_tensor.shape}")
    print(f"  Auxiliary data shape: {auxiliary_data.shape}")
    print(f"  Prepared input shape: {x.shape}")
    
    output = model(x, auxiliary_data=auxiliary_data)
    print(f"  Output shape: {output.shape}\n")
    
    # Example 2: Sequences of events with auxiliary data
    print("Example 2: Sequences of events with auxiliary data")
    event_sequences = torch.randn(batch_size, seq_len, input_dim)
    auxiliary_data = torch.randn(batch_size, auxiliary_data_dim)
    
    x = EventTransformerClassifier.prepare_input(event_sequences)
    print(f"  Event sequences shape: {event_sequences.shape}")
    print(f"  Auxiliary data shape: {auxiliary_data.shape}")
    print(f"  Prepared input shape: {x.shape}")
    
    output = model(x, auxiliary_data=auxiliary_data)
    print(f"  Output shape: {output.shape}\n")
    
    # Example 3: Single auxiliary data broadcasted to batch
    print("Example 3: Single auxiliary data (1, N) broadcasted to batch")
    event_sequences = torch.randn(batch_size, seq_len, input_dim)
    auxiliary_data_single = torch.randn(1, auxiliary_data_dim)  # Shape (1, N)
    
    x = EventTransformerClassifier.prepare_input(event_sequences)
    print(f"  Event sequences shape: {event_sequences.shape}")
    print(f"  Auxiliary data shape: {auxiliary_data_single.shape}")
    print(f"  Prepared input shape: {x.shape}")
    
    output = model(x, auxiliary_data=auxiliary_data_single)
    print(f"  Output shape: {output.shape}")
    print(f"\n✓ Model created successfully!")


if __name__ == '__main__':
    test_event_tensor_consistency()
    print("\n" + "="*50 + "\n")
    test_model_examples()
