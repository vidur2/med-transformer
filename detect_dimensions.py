"""
Detect dimensions from tensor_data.json and update text_ranges.json with dimension metadata.

This script reads the tensor data and automatically detects:
- Event dimension
- Procedure dimension  
- Number of classes

Then updates text_ranges.json to include this metadata for use in training.

Usage:
    python detect_dimensions.py
"""

import json
from pathlib import Path


def detect_dimensions(data_path: str, text_ranges_path: str):
    """
    Detect dimensions from tensor_data.json and update text_ranges.json.
    
    Args:
        data_path: Path to tensor_data.json
        text_ranges_path: Path to text_ranges.json to update
    """
    print("=" * 60)
    print("Detecting dimensions from tensor data")
    print("=" * 60)
    print()
    
    # Load tensor data
    print(f"Loading data from {data_path}...")
    with open(data_path, 'r') as f:
        tensor_data = json.load(f)
    
    if not tensor_data:
        raise ValueError("No data found in tensor_data.json")
    
    # Get first patient's data
    sample_patient_id = next(iter(tensor_data.keys()))
    sample_data = tensor_data[sample_patient_id]
    
    # Detect event dimension
    if not sample_data['events']:
        raise ValueError("No events found in sample patient data")
    event_dim = len(sample_data['events'][0])
    
    # Detect procedure dimension
    procedure_dim = None
    for patient_id, patient_data in tensor_data.items():
        for event_procs in patient_data['procedures']:
            if len(event_procs) > 0:
                procedure_dim = len(event_procs[0])
                break
        if procedure_dim is not None:
            break
    
    if procedure_dim is None:
        print("Warning: No procedures found in data. Using default procedure_dim=384")
        procedure_dim = 384
    
    # Detect number of classes
    num_classes = len(sample_data['targets'][0])
    
    print(f"✓ Detected dimensions:")
    print(f"  Event dimension: {event_dim}")
    print(f"  Procedure dimension: {procedure_dim}")
    print(f"  Number of classes: {num_classes}")
    print()
    
    # Load existing text_ranges.json
    if Path(text_ranges_path).exists():
        print(f"Loading existing text_ranges.json from {text_ranges_path}...")
        with open(text_ranges_path, 'r') as f:
            ranges_data = json.load(f)
    else:
        print(f"text_ranges.json not found. Creating new file...")
        ranges_data = {
            'text_feature_ranges_by_schema': {},
            'tensor_dims_by_schema': {}
        }
    
    # Add dimension metadata
    ranges_data['dimensions'] = {
        'event_dim': event_dim,
        'procedure_dim': procedure_dim,
        'num_classes': num_classes
    }
    
    # Save updated text_ranges.json
    with open(text_ranges_path, 'w') as f:
        json.dump(ranges_data, f, indent=2)
    
    print(f"✓ Updated {text_ranges_path} with dimension metadata")
    print()
    
    return {
        'event_dim': event_dim,
        'procedure_dim': procedure_dim,
        'num_classes': num_classes
    }


def main():
    # Default paths
    data_dir = Path(__file__).parent / 'data'
    data_path = data_dir / 'tensor_data.json'
    text_ranges_path = data_dir / 'text_ranges.json'
    
    if not data_path.exists():
        print(f"Error: {data_path} not found")
        print("Please run preprocess_event_data/datamodel.py first to generate tensor_data.json")
        return
    
    dimensions = detect_dimensions(str(data_path), str(text_ranges_path))
    
    print("✓ Done!")
    print()
    print("You can now run training with auto-detected dimensions:")
    print(f"  python train.py --data_path {data_path} --epochs 50")


if __name__ == '__main__':
    main()
