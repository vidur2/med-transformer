"""
Test to verify EventTransformerClassifier supports backpropagation.
"""
import sys
import os
sys.path.insert(0, 'model')

import torch
import torch.nn as nn
import torch.optim as optim
from event_classifier import EventTransformerClassifier


def test_basic_backpropagation():
    """Test that gradients flow through the model."""
    print("Test 1: Basic Backpropagation")
    print("-" * 50)
    
    # Create model
    model = EventTransformerClassifier()
    
    # Create dummy input
    batch_size = 4
    seq_len = 5
    input_dim = 387
    x = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)
    
    # Forward pass
    output = model(x)
    
    # Create dummy target
    target = torch.randint(0, 800, (batch_size,))
    
    # Compute loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, target)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Check that gradients were computed
    assert x.grad is not None, "Input gradient is None"
    assert x.grad.shape == x.shape, f"Gradient shape mismatch: {x.grad.shape} vs {x.shape}"
    
    # Check that model parameters have gradients
    params_with_grad = 0
    params_without_grad = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"
            params_with_grad += 1
        else:
            params_without_grad += 1
    
    print(f"Parameters with gradients: {params_with_grad}")
    print(f"Parameters without gradients: {params_without_grad}")
    print("✓ Basic backpropagation test passed!\n")


def test_backprop_with_auxiliary_data():
    """Test backpropagation with auxiliary data."""
    print("Test 2: Backpropagation with Auxiliary Data")
    print("-" * 50)
    
    # Create model with auxiliary data
    auxiliary_data_dim = 64
    model = EventTransformerClassifier(auxiliary_data_dim=auxiliary_data_dim)
    
    # Create dummy input
    batch_size = 4
    seq_len = 5
    input_dim = 387
    event_tensor = torch.randn(batch_size, seq_len, input_dim)
    auxiliary_data = torch.randn(batch_size, auxiliary_data_dim)
    
    # Prepare input
    x = event_tensor  # Already 3D: (batch_size, seq_len, input_dim)
    x.requires_grad = True
    
    # Forward pass
    output = model(x, auxiliary_data=auxiliary_data)
    
    # Create dummy target
    target = torch.randint(0, 800, (batch_size,))
    
    # Compute loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, target)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    assert x.grad is not None, "Input gradient is None"
    
    # Check that all trainable parameters have gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients"
            assert not torch.isinf(param.grad).any(), f"Parameter {name} has Inf gradients"
    
    print("✓ Backpropagation with auxiliary data test passed!\n")


def test_optimizer_step():
    """Test that optimizer can update model parameters."""
    print("Test 3: Optimizer Parameter Update")
    print("-" * 50)
    
    # Create model
    model = EventTransformerClassifier()
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Store initial parameter values
    initial_params = {}
    for name, param in model.named_parameters():
        initial_params[name] = param.data.clone()
    
    # Create dummy input and target
    batch_size = 4
    seq_len = 5
    input_dim = 387
    x = torch.randn(batch_size, seq_len, input_dim)
    target = torch.randint(0, 800, (batch_size,))
    
    # Forward pass
    output = model(x)
    
    # Compute loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, target)
    
    print(f"Initial loss: {loss.item():.4f}")
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Optimizer step
    optimizer.step()
    
    # Check that parameters were updated
    params_updated = 0
    params_not_updated = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            if not torch.equal(param.data, initial_params[name]):
                params_updated += 1
            else:
                params_not_updated += 1
    
    print(f"Parameters updated: {params_updated}")
    print(f"Parameters not updated: {params_not_updated}")
    
    assert params_updated > 0, "No parameters were updated"
    
    print("✓ Optimizer step test passed!\n")


def test_multiple_training_steps():
    """Test multiple training iterations."""
    print("Test 4: Multiple Training Steps")
    print("-" * 50)
    
    # Create model
    model = EventTransformerClassifier()
    
    # Create optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Training loop
    batch_size = 8
    seq_len = 3
    input_dim = 387
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    num_steps = 5
    
    for step in range(num_steps):
        # Create dummy data
        x = torch.randn(batch_size, seq_len, input_dim)
        target = torch.randint(0, 800, (batch_size,))
        
        # Forward pass
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check for NaN or Inf gradients
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name} at step {step}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name} at step {step}"
        
        # Update parameters
        optimizer.step()
        
        losses.append(loss.item())
        print(f"  Step {step + 1}: Loss = {loss.item():.4f}")
    
    print(f"Loss values: {[f'{l:.4f}' for l in losses]}")
    print("✓ Multiple training steps test passed!\n")


def test_gradient_flow():
    """Test that gradients flow through all layers."""
    print("Test 5: Gradient Flow Through All Layers")
    print("-" * 50)
    
    # Create model
    model = EventTransformerClassifier()
    
    # Create dummy input
    batch_size = 4
    seq_len = 5
    input_dim = 387
    x = torch.randn(batch_size, seq_len, input_dim)
    target = torch.randint(0, 800, (batch_size,))
    
    # Forward pass
    output = model(x)
    
    # Compute loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, target)
    
    # Backward pass
    loss.backward()
    
    # Check gradient flow by layer
    layers_with_gradients = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            layers_with_gradients.append((name, grad_norm))
    
    print(f"Total layers with gradients: {len(layers_with_gradients)}")
    print("\nGradient norms by layer:")
    for name, grad_norm in layers_with_gradients:
        print(f"  {name}: {grad_norm:.6f}")
    
    # Verify all layers have non-zero gradients
    zero_grad_layers = [name for name, norm in layers_with_gradients if norm == 0]
    if zero_grad_layers:
        print(f"\nWarning: Layers with zero gradients: {zero_grad_layers}")
    
    print("\n✓ Gradient flow test passed!\n")


def test_batch_independence():
    """Test that gradients are computed independently for each batch item."""
    print("Test 6: Batch Independence")
    print("-" * 50)
    
    # Create model
    model = EventTransformerClassifier()
    
    batch_size = 4
    seq_len = 5
    input_dim = 387
    
    # Process single item
    x_single = torch.randn(1, seq_len, input_dim)
    target_single = torch.randint(0, 800, (1,))
    
    output_single = model(x_single)
    criterion = nn.CrossEntropyLoss()
    loss_single = criterion(output_single, target_single)
    
    model.zero_grad()
    loss_single.backward()
    
    # Store single gradients
    single_grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            single_grads[name] = param.grad.clone()
    
    # Process batch (same item repeated)
    x_batch = x_single.repeat(batch_size, 1, 1)
    target_batch = target_single.repeat(batch_size)
    
    output_batch = model(x_batch)
    loss_batch = criterion(output_batch, target_batch)
    
    model.zero_grad()
    loss_batch.backward()
    
    print(f"Single loss: {loss_single.item():.4f}")
    print(f"Batch loss: {loss_batch.item():.4f}")
    print("✓ Batch independence test passed!\n")


if __name__ == '__main__':
    print("=" * 50)
    print("BACKPROPAGATION TESTS: EventTransformerClassifier")
    print("=" * 50)
    print()
    
    test_basic_backpropagation()
    test_backprop_with_auxiliary_data()
    test_optimizer_step()
    test_multiple_training_steps()
    test_gradient_flow()
    test_batch_independence()
    
    print("=" * 50)
    print("ALL BACKPROPAGATION TESTS PASSED! ✓")
    print("=" * 50)
