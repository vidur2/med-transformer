# Medical Event Transformer

A deep learning system for medical event classification using transformer architectures with text-aware feature masking and cumulative procedure context.

## Overview

This project provides an end-to-end pipeline for:
- Preprocessing medical event data with text embeddings
- Training transformer models with feature missingness handling
- Making predictions on raw medical event data
- Multi-label classification of 800+ target categories

## Key Features

- **Text-Aware Feature Masking**: Automatically detects and masks entire text embedding ranges (not individual positions)
- **Cumulative Procedure Context**: Each event prediction uses procedures from all previous events (not current event)
- **Dual-Stream Architecture**: Separate transformers for events and procedures with cross-attention
- **Per-Event Predictions**: Sequence-to-sequence predictions with targets for each event
- **Automatic Dimension Detection**: Detects event/procedure dimensions from data
- **Class Imbalance Handling**: Supports weighted loss and focal loss for sparse labels

## Project Structure

```
med-transformer/
├── preprocess_event_data/
│   ├── abstract.py              # Base classes (DataPoint, Category)
│   ├── datamodel.py             # Data processing and tensor conversion
│   ├── proc_json.py             # Event tensor conversion with text tracking
│   └── generated_categories/    # Auto-generated category mappings
├── model/
│   ├── event_classifier.py      # Transformer model with text-aware masking
│   └── data_loader.py          # DataLoader with cumulative procedures
├── data/
│   ├── processed_structured.json    # Raw input data
│   ├── tensor_data.json            # Preprocessed tensor data
│   ├── text_ranges.json            # Auto-detected text feature ranges
│   └── target_categories.json      # Category name mappings
├── checkpoints/
│   └── best_model.pt           # Trained model checkpoint
├── train.py                    # Training script
├── inference.py                # Inference on raw data
├── detect_dimensions.py        # Auto-detect tensor dimensions
└── tests/                      # Unit tests
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/vidur2/med-transformer.git
cd med-transformer
```

2. Install dependencies:
```bash
pip install torch sentence-transformers numpy tqdm scikit-learn
```

## Training a Model

### Step 1: Preprocess Data

Process raw medical events and convert to tensors:

```bash
python preprocess_event_data/datamodel.py
```

This will:
- Read from `data/processed_structured.json`
- Convert events and procedures to tensors using text embeddings
- Detect and save text feature ranges for each schema type
- Output `tensor_data.json` and `text_ranges.json`
- Export target category mappings to `target_categories.json`

### Step 2: Detect Dimensions

Automatically detect tensor dimensions from the processed data:

```bash
python detect_dimensions.py
```

This updates `text_ranges.json` with:
- `event_dim`: Event feature dimension
- `procedure_dim`: Procedure feature dimension  
- `num_classes`: Number of target classes

### Step 3: Train the Model

Train with auto-detected dimensions:

```bash
# Basic training
python train.py --data_path data/tensor_data.json --epochs 50 --batch_size 32

# With feature missingness augmentation
python train.py --data_path data/tensor_data.json --epochs 50 --batch_size 32 --use_missingness

# Custom hyperparameters
python train.py \
    --data_path data/tensor_data.json \
    --epochs 100 \
    --batch_size 16 \
    --hidden_dim 512 \
    --num_heads 8 \
    --num_transformer_layers 6 \
    --dropout 0.1 \
    --lr 1e-4 \
    --use_missingness
```

The model will:
- Automatically load dimensions from `text_ranges.json`
- Load text feature ranges for text-aware masking
- Train with cumulative procedure context (procedures from previous events only)
- Save best model to `checkpoints/best_model.pt`
- Display training and validation loss per epoch

### Training Options

```
--data_path         Path to tensor_data.json (required)
--epochs            Number of training epochs (default: 50)
--batch_size        Batch size (default: 32)
--hidden_dim        Hidden dimension size (default: 512)
--num_heads         Number of attention heads (default: 8)
--num_transformer_layers  Number of transformer layers (default: 4)
--dropout           Dropout rate (default: 0.1)
--lr                Learning rate (default: 1e-4)
--use_missingness   Enable random feature missingness augmentation
--save_dir          Checkpoint directory (default: checkpoints)
--seed              Random seed (default: 42)
```

## Running Inference

### Interactive Mode

Test predictions interactively:

```bash
python inference.py --model_path checkpoints/best_model.pt --interactive
```

Then enter event JSON:
```json
{"PAITENT_ID": "12345", "AGE": 65, "SEX": "Female", "PRIMARY_DIAGNOSIS": "I50.9", "PRIMARY_DIAGNOSIS_DESCRIPTION": "Heart failure, unspecified", "procedures": [{"name": "Echocardiogram", "code": "93306"}]}
```

Or load from file:
```
> file:data/example.json
```

### Batch Prediction

Predict on a single event:

```bash
python inference.py \
    --model_path checkpoints/best_model.pt \
    --input_file data/example.json \
    --top_k 10 \
    --threshold 0.5
```

Process multiple events:

```bash
python inference.py \
    --model_path checkpoints/best_model.pt \
    --input_file data/processed_structured.json \
    --output_file predictions.json \
    --top_k 20 \
    --threshold 0.3
```

### Inference Options

```
--model_path      Path to trained model checkpoint (required)
--input_file      Path to input JSON file with event data
--output_file     Path to save predictions (JSON)
--interactive     Run in interactive mode
--top_k           Number of top predictions to return (default: 10)
--threshold       Probability threshold for predictions (default: 0.5)
--device          Device to run on: cpu or cuda (default: cpu)
```

### Input Format

Inference accepts raw event data in the same format as `processed_structured.json`:

```json
{
  "PAITENT_ID": "12345",
  "ENCOUNTER_ID": 7429665188981,
  "ADMISSION_AGE_YEARS": 65,
  "SEX": "Female",
  "PRIMARY_DIAGNOSIS_CODE": "I50.9",
  "PRIMARY_DIAGNOSIS_DESCRIPTION": "Heart failure, unspecified",
  "VIZ_MSDRG_DESCRIPTION": "Heart failure and shock with MCC",
  "procedures": [
    {
      "code": "30233N1",
      "name": "Transfusion of blood cells",
      "date": "2025-10-18"
    }
  ]
}
```

**Note**: Metadata fields like `variable_description` and `DIAGNOSIS_RISK_VARIABLE_DESCRIPTION` are automatically filtered out through the `fieldMap` and will not affect predictions.

### Output Format

Predictions include:

```json
{
  "top_k": [
    {"category": "Respiratory Support", "probability": 0.87, "index": 42},
    {"category": "Cardiac Monitoring", "probability": 0.76, "index": 15}
  ],
  "above_threshold": [
    {"category": "Respiratory Support", "probability": 0.87, "index": 42}
  ],
  "num_procedures": 2,
  "max_probability": 0.87,
  "mean_probability": 0.12
}
```

## Model Architecture

- **Input Processing**: Text fields embedded with `all-MiniLM-L6-v2` (384 dims)
- **Event Encoder**: Transformer with text-aware feature masking
- **Procedure Encoder**: Separate transformer for cumulative procedures  
- **Fusion**: Cross-attention between event and procedure streams
- **Output**: Per-event multi-label predictions (800 classes)

### Text-Aware Masking

When training with `--use_missingness`, the model randomly masks features:
- Text embeddings are masked as complete 384-dim units (not individual positions)
- Ranges detected automatically from `text_ranges.json`
- Helps model handle missing clinical data

### Cumulative Procedures

For each event at timestep `t`:
- Uses procedures from events `0` to `t-1` (previous events only)
- First event has no procedure context (realistic constraint)
- Reflects real-world scenario: procedures in current encounter are unknown at prediction time

## Running Tests

```bash
# Test event tensor conversion
python tests/test_event.py

# Test data loader
python model/data_loader.py

# Test model forward pass
python tests/test_integration.py
```

## Requirements

- Python 3.7+
- PyTorch 1.x
- sentence-transformers
- numpy
- tqdm
- scikit-learn
