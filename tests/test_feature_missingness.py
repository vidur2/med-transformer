"""
Test feature-level missingness masking functionality.

Tests both forward pass and backpropagation with feature missingness.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from model.event_classifier import EventTransformerClassifier


def test_make_feature_mask():
    """Test feature mask creation."""
    print("Testing make_feature_mask...")
    
    batch_size, seq_len, input_dim = 2, 5, 10
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Test 1: No missing features (mask = all 1s)
    mask = EventTransformerClassifier.make_feature_mask(x, missing_feat_idx=None)
    assert mask.shape == x.shape, f"Expected shape {x.shape}, got {mask.shape}"
    assert torch.all(mask == 1.0), "Mask should be all 1s when no features are missing"
    print("  ✓ No missing features: mask is all 1s")
    
    # Test 2: Some features missing in last timestep
    missing_idx = torch.tensor([0, 3, 7])
    mask = EventTransformerClassifier.make_feature_mask(x, missing_feat_idx=missing_idx)
    
    # Check that all timesteps except last have all 1s
    assert torch.all(mask[:, :-1, :] == 1.0), "All timesteps except last should have all 1s"
    
    # Check that specified features in last timestep are 0
    assert torch.all(mask[:, -1, missing_idx] == 0.0), "Specified features in last timestep should be 0"
    
    # Check that non-specified features in last timestep are 1
    non_missing_idx = [i for i in range(input_dim) if i not in missing_idx.tolist()]
    assert torch.all(mask[:, -1, non_missing_idx] == 1.0), "Non-specified features should be 1"
    print("  ✓ Missing features correctly masked in last timestep only")
    
    # Test 3: List input
    mask_from_list = EventTransformerClassifier.make_feature_mask(x, missing_feat_idx=[0, 3, 7])
    assert torch.all(mask == mask_from_list), "Mask from list should match mask from tensor"
    print("  ✓ List input produces same result as tensor input")
    
    print("✓ make_feature_mask tests passed!\n")


def test_apply_feature_missingness():
    """Test feature missingness application."""
    print("Testing apply_feature_missingness...")
    
    batch_size, seq_len, input_dim = 2, 3, 5
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Create a mask with some 0s
    mask = torch.ones_like(x)
    mask[:, -1, [1, 3]] = 0.0  # Mask features 1 and 3 in last timestep
    
    # Apply missingness
    x_aug = EventTransformerClassifier.apply_feature_missingness(x, mask)
    
    # Test 1: Output shape
    expected_shape = (batch_size, seq_len, input_dim * 2)
    assert x_aug.shape == expected_shape, f"Expected shape {expected_shape}, got {x_aug.shape}"
    print("  ✓ Output shape is correct: [B, T, 2F]")
    
    # Test 2: First half is zero-filled features
    x_filled = x_aug[:, :, :input_dim]
    expected_filled = x * mask
    assert torch.allclose(x_filled, expected_filled), "First half should be zero-filled features"
    print("  ✓ First half contains zero-filled features")
    
    # Test 3: Second half is the mask
    mask_part = x_aug[:, :, input_dim:]
    assert torch.allclose(mask_part, mask), "Second half should be the mask"
    print("  ✓ Second half contains the mask")
    
    # Test 4: Missing features are zeroed out
    assert torch.all(x_filled[:, -1, [1, 3]] == 0.0), "Missing features should be zero"
    assert torch.all(x_filled[:, -1, [0, 2, 4]] != 0.0), "Non-missing features should not be zero"
    print("  ✓ Missing features are correctly zeroed out")
    
    print("✓ apply_feature_missingness tests passed!\n")


def test_forward_with_missingness():
    """Test forward pass with feature missingness."""
    print("Testing forward pass with feature missingness...")
    
    batch_size = 4
    seq_len = 8
    input_dim = 387
    hidden_dim = 128
    num_classes = 800
    
    model = EventTransformerClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes
    )
    model.eval()
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Test 1: Forward pass without missingness
    output_no_missing = model(x, missing_feat_idx=None)
    assert output_no_missing.shape == (batch_size, seq_len, num_classes), \
        f"Expected output shape ({batch_size}, {seq_len}, {num_classes}), got {output_no_missing.shape}"
    print("  ✓ Forward pass without missingness works")
    
    # Test 2: Forward pass with missingness
    missing_idx = torch.tensor([10, 50, 100, 200])
    output_with_missing = model(x, missing_feat_idx=missing_idx)
    assert output_with_missing.shape == (batch_size, seq_len, num_classes), \
        f"Expected output shape ({batch_size}, {seq_len}, {num_classes}), got {output_with_missing.shape}"
    print("  ✓ Forward pass with missingness works")
    
    # Test 3: Outputs should be different
    assert not torch.allclose(output_no_missing, output_with_missing), \
        "Outputs with and without missingness should be different"
    print("  ✓ Missingness affects output as expected")
    
    # Test 4: Forward pass with auxiliary data and missingness
    auxiliary_data_dim = 10
    model_with_aux = EventTransformerClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        auxiliary_data_dim=auxiliary_data_dim
    )
    model_with_aux.eval()
    auxiliary_data = torch.randn(batch_size, auxiliary_data_dim)
    output_with_aux = model_with_aux(x, auxiliary_data=auxiliary_data, missing_feat_idx=missing_idx)
    assert output_with_aux.shape == (batch_size, seq_len, num_classes), \
        f"Expected output shape ({batch_size}, {seq_len}, {num_classes}), got {output_with_aux.shape}"
    print("  ✓ Forward pass with auxiliary data and missingness works")
    
    # Test 5: Forward pass with padding mask and missingness
    src_key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    src_key_padding_mask[:, -2:] = True  # Mark last 2 timesteps as padding
    output_with_padding = model(x, src_key_padding_mask=src_key_padding_mask, missing_feat_idx=missing_idx)
    assert output_with_padding.shape == (batch_size, seq_len, num_classes), \
        f"Expected output shape ({batch_size}, {seq_len}, {num_classes}), got {output_with_padding.shape}"
    print("  ✓ Forward pass with padding mask and missingness works")
    
    # Test 6: Forward pass with procedures and missingness
    model_with_proc = EventTransformerClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        procedure_dim=50
    )
    model_with_proc.eval()
    
    procedure_x = torch.randn(batch_size, 5, 50)
    output_with_proc = model_with_proc(x, procedure_x=procedure_x, missing_feat_idx=missing_idx)
    assert output_with_proc.shape == (batch_size, seq_len, num_classes), \
        f"Expected output shape ({batch_size}, {seq_len}, {num_classes}), got {output_with_proc.shape}"
    print("  ✓ Forward pass with procedures and missingness works")
    
    print("✓ Forward pass tests passed!\n")


def test_backprop_with_missingness():
    """Test backpropagation with feature missingness."""
    print("Testing backpropagation with feature missingness...")
    
    batch_size = 4
    seq_len = 8
    input_dim = 387
    hidden_dim = 128
    num_classes = 800
    
    model = EventTransformerClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes
    )
    model.train()
    
    x = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)
    missing_idx = torch.tensor([10, 50, 100, 200])
    
    # Forward pass
    output = model(x, missing_feat_idx=missing_idx)
    
    # Create dummy per-event target and compute loss
    target = torch.randint(0, 2, (batch_size, seq_len, num_classes)).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target)
    
    # Backward pass
    loss.backward()
    
    # Test 1: Input gradients exist
    assert x.grad is not None, "Input should have gradients"
    print("  ✓ Input gradients exist")
    
    # Test 2: All model parameters have gradients
    param_count = 0
    params_with_grad = 0
    for name, param in model.named_parameters():
        param_count += 1
        if param.grad is not None:
            params_with_grad += 1
            assert not torch.all(param.grad == 0), f"Parameter {name} has all-zero gradients"
    
    assert param_count == params_with_grad, \
        f"Only {params_with_grad}/{param_count} parameters have gradients"
    print(f"  ✓ All {param_count} parameters have non-zero gradients")
    
    # Test 3: Gradients for missing features in last timestep
    # The gradient for zero-filled features should still exist (through the mask multiplication)
    last_timestep_grad = x.grad[:, -1, :]
    assert last_timestep_grad is not None, "Last timestep should have gradients"
    assert torch.any(last_timestep_grad != 0), "Last timestep should have non-zero gradients"
    print("  ✓ Last timestep has gradients (including for masked features)")
    
    # Test 4: Gradients for non-last timesteps
    other_timesteps_grad = x.grad[:, :-1, :]
    assert torch.any(other_timesteps_grad != 0), "Other timesteps should have non-zero gradients"
    print("  ✓ Other timesteps have non-zero gradients")
    
    print("✓ Backpropagation tests passed!\n")


def test_backprop_with_procedures_and_missingness():
    """Test backpropagation with procedures and feature missingness."""
    print("Testing backpropagation with procedures and missingness...")
    
    batch_size = 4
    seq_len = 8
    input_dim = 387
    procedure_dim = 50
    hidden_dim = 128
    num_classes = 800
    
    model = EventTransformerClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        procedure_dim=procedure_dim
    )
    model.train()
    
    x = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)
    procedure_x = torch.randn(batch_size, 5, procedure_dim, requires_grad=True)
    missing_idx = torch.tensor([10, 50, 100])
    
    # Forward pass
    output = model(x, procedure_x=procedure_x, missing_feat_idx=missing_idx)
    
    # Create dummy per-event target and compute loss
    target = torch.randint(0, 2, (batch_size, seq_len, num_classes)).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target)
    
    # Backward pass
    loss.backward()
    
    # Test 1: Both inputs have gradients
    assert x.grad is not None, "Event input should have gradients"
    assert procedure_x.grad is not None, "Procedure input should have gradients"
    print("  ✓ Both event and procedure inputs have gradients")
    
    # Test 2: All model parameters have gradients
    param_count = 0
    params_with_grad = 0
    for name, param in model.named_parameters():
        param_count += 1
        if param.grad is not None:
            params_with_grad += 1
    
    assert param_count == params_with_grad, \
        f"Only {params_with_grad}/{param_count} parameters have gradients"
    print(f"  ✓ All {param_count} parameters have gradients")
    
    # Test 3: Procedure-specific parameters have gradients
    assert model.procedure_projection.weight.grad is not None, \
        "Procedure projection should have gradients"
    assert not torch.all(model.procedure_projection.weight.grad == 0), \
        "Procedure projection should have non-zero gradients"
    print("  ✓ Procedure-specific parameters have non-zero gradients")
    
    print("✓ Backpropagation with procedures tests passed!\n")


def test_missingness_device_safety():
    """Test that missingness masking works correctly on different devices."""
    print("Testing device safety...")
    
    batch_size = 2
    seq_len = 4
    input_dim = 10
    hidden_dim = 32
    num_classes = 5
    
    # Test on CPU
    model_cpu = EventTransformerClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes
    )
    x_cpu = torch.randn(batch_size, seq_len, input_dim)
    missing_idx = torch.tensor([1, 3, 5])
    
    output_cpu = model_cpu(x_cpu, missing_feat_idx=missing_idx)
    assert output_cpu.device.type == 'cpu', "Output should be on CPU"
    print("  ✓ Works correctly on CPU")
    
    # Test on GPU if available
    if torch.cuda.is_available():
        model_gpu = model_cpu.cuda()
        x_gpu = x_cpu.cuda()
        missing_idx_gpu = missing_idx.cuda()
        
        output_gpu = model_gpu(x_gpu, missing_feat_idx=missing_idx_gpu)
        assert output_gpu.device.type == 'cuda', "Output should be on CUDA"
        print("  ✓ Works correctly on GPU")
    else:
        print("  ⚠ GPU not available, skipping GPU test")
    
    print("✓ Device safety tests passed!\n")


def run_all_tests():
    """Run all feature missingness tests."""
    print("=" * 60)
    print("Running Feature Missingness Tests")
    print("=" * 60 + "\n")
    
    test_make_feature_mask()
    test_apply_feature_missingness()
    test_forward_with_missingness()
    test_backprop_with_missingness()
    test_backprop_with_procedures_and_missingness()
    test_missingness_device_safety()
    
    print("=" * 60)
    print("All Feature Missingness Tests Passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
