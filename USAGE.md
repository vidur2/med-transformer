# Medical Event Transformer - Usage Guide

## Overview

This project provides a transformer-based model for **per-event sequence labeling** on medical event data. Each event in a patient's timeline receives its own multi-label classification prediction.

## Key Features

- **Per-event predictions**: Each timestep gets independent classification (batch_size, seq_len, num_classes)
- **Variable-length sequences**: Automatic padding and masking for efficient batching
- **Feature-level missingness**: Handle missing features in the most recent event
- **Dual-stream processing**: Separate transformers for events and procedures
- **Shape-aware initialization**: Model dimensions auto-configured from data

## Quick Start

### 1. Data Preparation

Each patient's data should have:
- **events**: List of event tensors (one per timestep)
- **procedures**: List of lists of procedure tensors (per-event procedures)
- **targets**: List of target labels (one per event)

**Important**: Procedures are now structured per-event. Each event has its own list of associated procedures. During training, for timestep `i`, the model uses all procedures from events `0` to `i` (cumulative context).

```python
from preprocess_event_data.proc_json import Event
from preprocess_event_data.datamodel import EventDataPoint, fieldMap

# Process your raw data
tensor_data = {}
for patient_id, patient_raw_data in your_data.items():
    # Convert each event to tensor
    events = []
    for event_raw in patient_raw_data['events']:
        event_dp = EventDataPoint(event_raw, fieldMap)
        event_obj = Event(event_dp)
        events.append(event_obj.to_tensor())  # torch.Tensor (event_dim,)
    
    # Per-event procedures: each event has its own list
    procedures = []
    for event_raw in patient_raw_data['events']:
        event_procedures = []
        for proc_raw in event_raw.get('procedures', []):
            proc_dp = EventDataPoint(proc_raw, fieldMapProc)
            proc_obj = Event(proc_dp)
            event_procedures.append(proc_obj.to_tensor())
        procedures.append(event_procedures)  # List[List[Tensor]]
    
    # Per-event targets (multi-label)
    targets = [
        torch.tensor(event['target']) for event in patient_raw_data['events']
    ]  # Each target is shape (num_classes,)
    
    tensor_data[patient_id] = {
        'events': events,              # List[Tensor]
        'procedures': procedures,      # List[List[Tensor]] - per-event!
        'targets': targets             # List[Tensor]
    }
```

### 2. Create DataLoader

```python
from model.data_loader import create_dataloader

dataloader = create_dataloader(
    tensor_data=tensor_data,
    shape_file='preprocess_event_data/shape.json',
    batch_size=16,
    shuffle=True
)
```

### 3. Initialize Model

```python
from model.event_classifier import EventTransformerClassifier

# Auto-configure from shape.json
model = EventTransformerClassifier.from_shape_file(
    'preprocess_event_data/shape.json',
    num_classes=800,
    hidden_dim=512,
    num_transformer_layers=3,
    num_heads=8
)
```

### 4. Training Loop with Masked Loss

**Important**: You must mask out padded positions when computing loss!

**Note on Procedures**: The DataLoader provides cumulative procedures for each timestep. For timestep `i`, `procedures_per_event[i]` contains all procedures from events 0 to i. This enables the model to use historical procedure context.

```python
import torch.nn.functional as F

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for batch in dataloader:
    events = batch['events']  # (batch_size, seq_len, event_dim)
    targets = batch['targets']  # (batch_size, seq_len, num_classes)
    event_mask = batch['event_padding_mask']  # (batch_size, seq_len), True = padding
    
    # Get cumulative procedures for the full sequence (last timestep has all)
    last_timestep = events.shape[1] - 1
    procedures_full = batch['procedures_per_event'][last_timestep]  # (batch_size, cumulative_procs, proc_dim)
    proc_mask_full = batch['procedure_masks_per_event'][last_timestep]  # (batch_size, cumulative_procs)
    
    # Forward pass
    output = model(
        events,
        procedure_x=procedures_full,
        src_key_padding_mask=event_mask,
        procedure_padding_mask=proc_mask_full
    )  # (batch_size, seq_len, num_classes)
    
    # Compute masked loss (CRITICAL - ignore padding!)
    valid_mask = ~event_mask  # True = valid, False = padding
    
    # Loss per timestep per class
    loss_per_timestep = F.binary_cross_entropy_with_logits(
        output, targets, reduction='none'
    )  # (batch_size, seq_len, num_classes)
    
    # Average over classes, then mask padding
    loss_per_timestep = loss_per_timestep.mean(dim=-1)  # (batch_size, seq_len)
    loss_per_timestep = loss_per_timestep * valid_mask.float()
    
    # Average over valid timesteps only
    num_valid = valid_mask.sum()
    loss = loss_per_timestep.sum() / num_valid if num_valid > 0 else loss_per_timestep.sum()
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 5. Inference with Feature Missingness

Handle missing lab results or diagnostic features in the most recent event:

```python
model.eval()

with torch.no_grad():
    # Specify which features are unavailable in the last timestep
    # Example: lab results at indices [10, 25, 30] haven't come back yet
    missing_features = torch.tensor([10, 25, 30, 45, 60])
    
    # Get cumulative procedures up to last timestep
    last_timestep = batch['events'].shape[1] - 1
    procedures_full = batch['procedures_per_event'][last_timestep]
    proc_mask_full = batch['procedure_masks_per_event'][last_timestep]
    
    output = model(
        batch['events'],
        procedure_x=procedures_full,
        src_key_padding_mask=batch['event_padding_mask'],
        procedure_padding_mask=proc_mask_full,
        missing_feat_idx=missing_features  # <<< Feature missingness masking
    )
    
    # Get probabilities for each event
    probabilities = torch.sigmoid(output)  # (batch_size, seq_len, num_classes)
    
    # Binary predictions
    predictions = (probabilities > 0.5).float()
```

## Data Augmentation: Random Feature Missingness

Train the model to be robust to incomplete data:

```python
import random

for batch in dataloader:
    # Get cumulative procedures for full sequence
    last_timestep = batch['events'].shape[1] - 1
    procedures_full = batch['procedures_per_event'][last_timestep]
    proc_mask_full = batch['procedure_masks_per_event'][last_timestep]
    
    # Randomly apply missingness in 50% of batches
    if random.random() > 0.5:
        num_features = 76  # From shape.json
        num_missing = random.randint(5, 15)
        missing_indices = torch.randperm(num_features)[:num_missing]
        
        output = model(
            batch['events'],
            procedure_x=procedures_full,
            src_key_padding_mask=batch['event_padding_mask'],
            procedure_padding_mask=proc_mask_full,
            missing_feat_idx=missing_indices
        )
    else:
        # No missingness
        output = model(
            batch['events'],
            procedure_x=procedures_full,
            src_key_padding_mask=batch['event_padding_mask'],
            procedure_padding_mask=proc_mask_full
        )
    
    # Compute masked loss as before
    # ...
```

## Architecture Details

### Per-Event Procedures with Cumulative Context

**Key Innovation**: Each event has its own associated procedures, and when making predictions for event `i`, the model has access to all procedures from events `0` to `i`.

```
Example timeline:
Event 0: [Lab A, Lab B]           → Timestep 0 uses: [Lab A, Lab B]
Event 1: [X-ray]                  → Timestep 1 uses: [Lab A, Lab B, X-ray]
Event 2: [Lab C, MRI, CT]         → Timestep 2 uses: [Lab A, Lab B, X-ray, Lab C, MRI, CT]
```

This cumulative approach ensures:
- **Temporal consistency**: Later events always have access to earlier procedures
- **Realistic modeling**: Mimics how clinicians review all previous diagnostics
- **Efficient training**: Single forward pass processes all timesteps

The DataLoader automatically creates these cumulative sequences:
- `procedures_per_event[0]`: Procedures from event 0
- `procedures_per_event[1]`: Procedures from events 0-1
- `procedures_per_event[2]`: Procedures from events 0-2
- etc.

### Input Processing
1. **Feature Augmentation**: Input is augmented to `(input_dim * 2)` by concatenating `[features, availability_mask]`
2. **Projection**: Linear layer maps to `hidden_dim`
3. **Positional Encoding**: Added to preserve temporal ordering

### Transformer Processing
- **Event Transformer**: Processes main event sequence
- **Procedure Transformer** (optional): Processes procedures separately
- **Cumulative Procedure Context**: For each timestep i, uses all procedures from events 0 to i
- Both use multi-head self-attention with padding masks

### Output Generation
- Procedure representations are pooled then expanded to all timesteps
- MLP classifier applied **independently** to each timestep
- Output: `(batch_size, seq_len, num_classes)`

### Feature Missingness (Last Timestep Only)
- Only affects the most recent event (`timestep = -1`)
- Specified features are zero-filled
- Binary mask indicates availability: `[zero_filled_features, mask]`
- Model learns to make predictions even with missing features

## Common Pitfalls

❌ **Forgetting to mask padding in loss**
```python
# WRONG - includes padding in loss
loss = F.binary_cross_entropy_with_logits(output, targets)
```

✅ **Correct - mask out padding**
```python
# CORRECT
valid_mask = ~batch['event_padding_mask']
loss_per_timestep = F.binary_cross_entropy_with_logits(output, targets, reduction='none').mean(dim=-1)
loss = (loss_per_timestep * valid_mask.float()).sum() / valid_mask.sum()
```

❌ **Wrong target shape**
```python
# WRONG - single target per sequence
targets = torch.randint(0, 2, (batch_size, num_classes))
```

✅ **Correct - per-event targets**
```python
# CORRECT - one target per event
targets = torch.randint(0, 2, (batch_size, seq_len, num_classes))
```

## Testing

Run the test suite:
```bash
# Feature missingness tests
python tests/test_feature_missingness.py

# DataLoader integration tests
python model/data_loader.py
```

## Model Configuration

From `shape.json`:
- **Event dimension**: 76 features
- **Procedure dimension**: 385 features
- **Sequence lengths**: Variable (min: 1, max: varies by patient)

Recommended hyperparameters:
- `hidden_dim`: 512
- `num_transformer_layers`: 3
- `num_heads`: 8
- `dropout`: 0.1
- `batch_size`: 16-32
- `learning_rate`: 1e-4

## Output Interpretation

```python
output = model(batch['events'])  # (4, 10, 800)
# batch_size=4, seq_len=10, num_classes=800

# Predictions for patient 0, event 5
event_5_logits = output[0, 5, :]  # (800,)
event_5_probs = torch.sigmoid(event_5_logits)  # (800,)

# Top 5 predicted classes for this event
top5_classes = torch.topk(event_5_probs, k=5)
print(f"Top 5 classes: {top5_classes.indices}")
print(f"Probabilities: {top5_classes.values}")
```

## Performance Tips

1. **Gradient Accumulation**: For large models, accumulate gradients over multiple batches
2. **Mixed Precision**: Use `torch.cuda.amp` for faster training
3. **Batch Size**: Larger batches (32-64) improve training stability
4. **Sequence Length**: Consider truncating very long sequences (>20 events)
5. **Data Augmentation**: Random feature missingness improves robustness

## Citation

If you use this code, please ensure proper attribution and follow the project's license.
