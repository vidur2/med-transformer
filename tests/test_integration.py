"""
Integration test for the complete pipeline:
DataPoint -> Event -> Tensor -> Model Inference
"""
import sys
import os
sys.path.insert(0, 'preprocess_event_data')
sys.path.insert(0, 'model')

import torch
from abstract import DataPoint, Category
from proc_json import Event
from event_classifier import EventTransformerClassifier


class TestDataPoint(DataPoint):
    """Test implementation of DataPoint for integration testing."""
    def __init__(self, category: str, count: int, is_active: bool, description: str):
        super().__init__()
        self.category = Category(category)
        self.count = count
        self.is_active = is_active
        self.description = description
    
    def dump_contents(self):
        return [self.category, self.count, self.is_active, self.description]


def test_single_datapoint_inference():
    """Test inference on a single DataPoint."""
    print("Test 1: Single DataPoint Inference")
    print("-" * 50)
    
    # Create a test data point
    data_point = TestDataPoint(
        category="diagnosis",
        count=42,
        is_active=True,
        description="Patient presents with acute respiratory symptoms"
    )
    
    # Convert to Event
    event = Event(data_point)
    
    # Convert to tensor
    event_tensor = event.to_tensor()
    print(f"Event tensor shape: {event_tensor.shape}")
    
    # Add batch dimension for model
    event_tensor = event_tensor.unsqueeze(0)  # (1, 387)
    
    # Create model
    model = EventTransformerClassifier()
    
    # Generate random auxiliary data
    auxiliary_data_dim = 64
    auxiliary_data = torch.randn(1, auxiliary_data_dim)
    
    # Prepare input
    x = EventTransformerClassifier.prepare_input(event_tensor)
    print(f"Model input shape: {x.shape}")
    
    # Run inference
    model.eval()
    with torch.no_grad():
        output = model(x, auxiliary_data=auxiliary_data)
    
    print(f"Model output shape: {output.shape}")
    print(f"Predicted class: {output.argmax(dim=1).item()}")
    print(f"Max probability: {output.max().item():.4f}")
    print("✓ Single DataPoint inference successful!\n")


def test_sequence_datapoint_inference():
    """Test inference on a sequence of DataPoints."""
    print("Test 2: Sequence of DataPoints Inference")
    print("-" * 50)
    
    # Create a sequence of test data points
    categories = ["diagnosis", "treatment", "medication", "lab_result", "vital_sign"]
    descriptions = [
        "Initial patient assessment and diagnosis",
        "Prescribed treatment plan initiated",
        "Medication administration recorded",
        "Laboratory test results received",
        "Vital signs monitoring data"
    ]
    
    data_points = []
    for i, (cat, desc) in enumerate(zip(categories, descriptions)):
        data_point = TestDataPoint(
            category=cat,
            count=i * 10,
            is_active=i % 2 == 0,
            description=desc
        )
        data_points.append(data_point)
    
    # Convert each DataPoint to Event, then to tensor
    event_tensors = []
    for dp in data_points:
        event = Event(dp)
        tensor = event.to_tensor()
        event_tensors.append(tensor)
    
    # Stack tensors into a sequence
    sequence_tensor = torch.stack(event_tensors)  # (seq_len, 387)
    print(f"Sequence tensor shape: {sequence_tensor.shape}")
    
    # Add batch dimension
    sequence_tensor = sequence_tensor.unsqueeze(0)  # (1, seq_len, 387)
    
    # Create model
    model = EventTransformerClassifier()
    
    # Generate random auxiliary data
    auxiliary_data_dim = 64
    auxiliary_data = torch.randn(1, auxiliary_data_dim)
    
    # Prepare input
    x = EventTransformerClassifier.prepare_input(sequence_tensor)
    print(f"Model input shape: {x.shape}")
    
    # Run inference
    model.eval()
    with torch.no_grad():
        output = model(x, auxiliary_data=auxiliary_data)
    
    print(f"Model output shape: {output.shape}")
    print(f"Predicted class: {output.argmax(dim=1).item()}")
    print(f"Max probability: {output.max().item():.4f}")
    print("✓ Sequence DataPoint inference successful!\n")


def test_batch_datapoint_inference():
    """Test inference on a batch of DataPoints with randomized auxiliary data."""
    print("Test 3: Batch of DataPoints with Random Auxiliary Data")
    print("-" * 50)
    
    batch_size = 8
    categories = ["diagnosis", "treatment", "medication", "lab_result", "vital_sign", "procedure"]
    descriptions = [
        "Routine checkup and examination",
        "Emergency room admission",
        "Surgical procedure completed",
        "Post-operative care administered",
        "Physical therapy session",
        "Radiology imaging performed"
    ]
    
    # Create a batch of events
    batch_tensors = []
    for i in range(batch_size):
        data_point = TestDataPoint(
            category=categories[i % len(categories)],
            count=torch.randint(0, 100, (1,)).item(),
            is_active=torch.rand(1).item() > 0.5,
            description=descriptions[i % len(descriptions)]
        )
        event = Event(data_point)
        tensor = event.to_tensor()
        batch_tensors.append(tensor)
    
    # Stack into batch
    batch_tensor = torch.stack(batch_tensors)  # (batch_size, 387)
    print(f"Batch tensor shape: {batch_tensor.shape}")
    
    # Create model with auxiliary data
    auxiliary_data_dim = 128
    model = EventTransformerClassifier(auxiliary_data_dim=auxiliary_data_dim)
    
    # Generate random auxiliary data for each sample
    auxiliary_data = torch.randn(batch_size, auxiliary_data_dim)
    print(f"Auxiliary data shape: {auxiliary_data.shape}")
    
    # Prepare input
    x = EventTransformerClassifier.prepare_input(batch_tensor)
    print(f"Model input shape: {x.shape}")
    
    # Run inference
    model.eval()
    with torch.no_grad():
        output = model(x, auxiliary_data=auxiliary_data)
    
    print(f"Model output shape: {output.shape}")
    print(f"Predicted classes: {output.argmax(dim=1).tolist()}")
    print(f"Max probabilities per sample: {output.max(dim=1).values.tolist()}")
    print("✓ Batch DataPoint inference successful!\n")


def test_batch_sequence_inference():
    """Test inference on a batch of sequences with varying auxiliary data."""
    print("Test 4: Batch of Sequences with Varying Auxiliary Data")
    print("-" * 50)
    
    batch_size = 4
    seq_len = 6
    categories = ["diagnosis", "treatment", "medication", "lab_result"]
    
    # Create batch of sequences
    batch_sequences = []
    for b in range(batch_size):
        sequence = []
        for s in range(seq_len):
            data_point = TestDataPoint(
                category=categories[torch.randint(0, len(categories), (1,)).item()],
                count=torch.randint(0, 200, (1,)).item(),
                is_active=torch.rand(1).item() > 0.3,
                description=f"Batch {b} Event {s}: Medical record entry"
            )
            event = Event(data_point)
            tensor = event.to_tensor()
            sequence.append(tensor)
        batch_sequences.append(torch.stack(sequence))
    
    # Stack into batch
    batch_tensor = torch.stack(batch_sequences)  # (batch_size, seq_len, 387)
    print(f"Batch sequence tensor shape: {batch_tensor.shape}")
    
    # Create model
    auxiliary_data_dim = 32
    model = EventTransformerClassifier(auxiliary_data_dim=auxiliary_data_dim)
    
    # Generate different auxiliary data for each batch item
    auxiliary_data = torch.randn(batch_size, auxiliary_data_dim)
    print(f"Auxiliary data shape: {auxiliary_data.shape}")
    
    # Prepare input
    x = EventTransformerClassifier.prepare_input(batch_tensor)
    print(f"Model input shape: {x.shape}")
    
    # Run inference
    model.eval()
    with torch.no_grad():
        output = model(x, auxiliary_data=auxiliary_data)
    
    print(f"Model output shape: {output.shape}")
    print(f"Predicted classes: {output.argmax(dim=1).tolist()}")
    print("✓ Batch sequence inference successful!\n")


if __name__ == '__main__':
    print("=" * 50)
    print("INTEGRATION TESTS: DataPoint -> Event -> Model")
    print("=" * 50)
    print()
    
    test_single_datapoint_inference()
    test_sequence_datapoint_inference()
    test_batch_datapoint_inference()
    test_batch_sequence_inference()
    
    print("=" * 50)
    print("ALL INTEGRATION TESTS PASSED! ✓")
    print("=" * 50)
