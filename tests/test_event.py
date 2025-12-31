import sys
import os
sys.path.insert(0, 'preprocess_event_data')

import torch
from abstract import DataPoint, Category
from proc_json import Event


class TestDataPoint(DataPoint):
    def __init__(self, category: int, count: int, is_active: bool, description: str):
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
        category=1,
        count=42,
        is_active=True,
        description="Short text"
    )
    
    test_data_2 = TestDataPoint(
        category=2,
        count=100,
        is_active=False,
        description="This is a much longer description with many more words to test embedding consistency"
    )
    
    test_data_3 = TestDataPoint(
        category=3,
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


if __name__ == '__main__':
    test_event_tensor_consistency()
