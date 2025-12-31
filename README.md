# Event Data Preprocessing Project

This project provides tools for preprocessing event data by converting structured data into vector embeddings suitable for machine learning models.

## Overview

The project contains a core module that:
- Parses structured event data inheriting from `DataPoint`
- Converts categorical, numeric, and text data into unified vector representations
- Uses sentence transformers to generate embeddings for text fields
- Outputs results as PyTorch tensors for deep learning applications

## Project Structure

```
mother/
├── preprocess_event_data/
│   ├── abstract.py          # Base classes (DataPoint, Category)
│   ├── proc_json.py         # Event processing class
│   └── __pycache__/
├── tests/
│   └── test_event.py        # Unit tests
├── README.md
└── .gitignore
```

## Installation

1. Clone or navigate to the project directory
2. Install required dependencies:

```bash
pip install torch sentence-transformers
```

## Running Tests

From the root directory of the project, run:

```bash
python ./tests/test_event.py
```

This will execute the test suite which verifies that:
- Event objects produce consistent tensor dimensions
- Text inputs of varying lengths produce embeddings of the same size
- Output tensors are properly formatted for PyTorch

## Usage

### Basic Example

```python
from preprocess_event_data.abstract import DataPoint, Category
from preprocess_event_data.proc_json import Event

# Create a custom DataPoint implementation
class MyData(DataPoint):
    def __init__(self, category, count, is_active, description):
        self.category = Category(category)
        self.count = count
        self.is_active = is_active
        self.description = description
    
    def dump_contents(self):
        return [self.category, self.count, self.is_active, self.description]

# Create an event and convert to tensor
data = MyData(1, 42, True, "Sample text")
event = Event(data)
tensor = event.to_tensor()
```

## Features

- **Shared Model Instance**: The sentence transformer model is initialized once and shared across all Event instances for efficiency
- **Flexible Data Types**: Supports categories, integers, booleans, and text strings
- **Consistent Output**: Regardless of text input length, all outputs maintain consistent tensor dimensions (387 features: 3 numeric + 384 embedding dimensions)
- **PyTorch Integration**: Direct tensor output for use in deep learning pipelines

## Requirements

- Python 3.7+
- torch
- sentence-transformers

## Model

The project uses the `all-MiniLM-L6-v2` sentence transformer model, which provides:
- 384-dimensional embeddings
- Lightweight and efficient inference
- Suitable for real-time applications
