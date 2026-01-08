"""
Test the padding mask utility functions.
"""
import sys
import os
sys.path.insert(0, 'model')

import torch
from event_classifier import EventTransformerClassifier


def test_create_padding_mask_from_values():
    """Test creating padding mask from padded sequences."""
    print("Test 1: Create Padding Mask from Values")
    print("-" * 50)
    
    # Create batch with variable-length sequences (padded with 0s)
    batch = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]],  # Length 2
        [[5.0, 6.0], [0.0, 0.0], [0.0, 0.0]],  # Length 1
        [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]  # Length 3 (no padding)
    ])
    
    mask = EventTransformerClassifier.create_padding_mask(batch, pad_value=0.0)
    
    print(f"Batch shape: {batch.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Expected mask:\n{torch.tensor([[False, False, True], [False, True, True], [False, False, False]])}")
    print(f"Actual mask:\n{mask}")
    
    expected = torch.tensor([
        [False, False, True],
        [False, True, True],
        [False, False, False]
    ])
    
    assert mask.shape == (3, 3), f"Wrong mask shape: {mask.shape}"
    assert torch.equal(mask, expected), "Mask values don't match expected"
    print("✓ Padding mask from values test passed!\n")


def test_create_padding_mask_from_lengths():
    """Test creating padding mask from sequence lengths."""
    print("Test 2: Create Padding Mask from Lengths")
    print("-" * 50)
    
    # Sequence lengths for batch
    seq_lengths = torch.tensor([2, 1, 3])
    max_len = 3
    
    mask = EventTransformerClassifier.create_padding_mask_from_lengths(seq_lengths, max_len)
    
    print(f"Sequence lengths: {seq_lengths}")
    print(f"Max length: {max_len}")
    print(f"Mask shape: {mask.shape}")
    print(f"Expected mask:\n{torch.tensor([[False, False, True], [False, True, True], [False, False, False]])}")
    print(f"Actual mask:\n{mask}")
    
    expected = torch.tensor([
        [False, False, True],
        [False, True, True],
        [False, False, False]
    ])
    
    assert mask.shape == (3, 3), f"Wrong mask shape: {mask.shape}"
    assert torch.equal(mask, expected), "Mask values don't match expected"
    print("✓ Padding mask from lengths test passed!\n")


def test_model_with_padding_mask():
    """Test that model works with padding masks."""
    print("Test 3: Model Forward Pass with Padding Masks")
    print("-" * 50)
    
    batch_size = 4
    max_len = 5
    input_dim = 387
    
    # Create model
    model = EventTransformerClassifier()
    
    # Create batch with variable lengths
    x = torch.randn(batch_size, max_len, input_dim)
    seq_lengths = torch.tensor([5, 3, 4, 2])
    
    # Create padding mask
    mask = EventTransformerClassifier.create_padding_mask_from_lengths(seq_lengths, max_len)
    
    print(f"Input shape: {x.shape}")
    print(f"Sequence lengths: {seq_lengths}")
    print(f"Padding mask shape: {mask.shape}")
    
    # Forward pass with mask
    model.eval()
    with torch.no_grad():
        output = model(x, src_key_padding_mask=mask)
    
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, 800), f"Wrong output shape: {output.shape}"
    print("✓ Model with padding mask test passed!\n")


if __name__ == '__main__':
    print("=" * 50)
    print("PADDING MASK UTILITY TESTS")
    print("=" * 50)
    print()
    
    test_create_padding_mask_from_values()
    test_create_padding_mask_from_lengths()
    test_model_with_padding_mask()
    
    print("=" * 50)
    print("ALL PADDING MASK TESTS PASSED! ✓")
    print("=" * 50)
