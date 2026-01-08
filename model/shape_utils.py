"""
Utilities for reading and processing shape metadata for model configuration.

This module provides functions to:
1. Extract input dimensions from shape.json
2. Calculate sequence length statistics for batching
3. Configure model parameters automatically
"""

import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import torch


def load_shape_metadata(shape_file: str) -> Dict:
    """
    Load shape metadata from JSON file.
    
    Args:
        shape_file: Path to shape.json file
        
    Returns:
        Dictionary with patient IDs as keys and shape info as values
    """
    with open(shape_file, 'r') as f:
        return json.load(f)


def extract_dimensions(shape_file: str) -> Tuple[int, int]:
    """
    Extract fixed feature dimensions from shape metadata.
    
    Args:
        shape_file: Path to shape.json file
        
    Returns:
        Tuple of (event_feature_dim, procedure_feature_dim)
        
    Example:
        >>> event_dim, proc_dim = extract_dimensions('shape.json')
        >>> print(f"Event features: {event_dim}, Procedure features: {proc_dim}")
        Event features: 76, Procedure features: 384
    """
    metadata = load_shape_metadata(shape_file)
    
    # Get first patient's shape to extract dimensions
    first_patient = next(iter(metadata.values()))
    
    event_dim = first_patient['x_shape'][1]
    procedure_dim = first_patient['procedure_shape'][1]
    
    # Verify dimensions are consistent across all patients
    for patient_id, shapes in metadata.items():
        assert shapes['x_shape'][1] == event_dim, \
            f"Inconsistent event dimension for patient {patient_id}"
        # Note: Some procedure shapes have dim 1, so we take the most common non-1 value
        if shapes['procedure_shape'][1] != 1:
            assert shapes['procedure_shape'][1] == procedure_dim, \
                f"Inconsistent procedure dimension for patient {patient_id}"
    
    return event_dim, procedure_dim


def get_sequence_statistics(shape_file: str) -> Dict[str, Dict[str, int]]:
    """
    Calculate sequence length statistics for efficient batching.
    
    Args:
        shape_file: Path to shape.json file
        
    Returns:
        Dictionary with statistics:
        {
            'events': {
                'min': min_event_seq_len,
                'max': max_event_seq_len,
                'mean': avg_event_seq_len,
                'median': median_event_seq_len
            },
            'procedures': {
                'min': min_proc_seq_len,
                'max': max_proc_seq_len,
                'mean': avg_proc_seq_len,
                'median': median_proc_seq_len
            }
        }
    """
    metadata = load_shape_metadata(shape_file)
    
    event_lengths = []
    procedure_lengths = []
    
    for shapes in metadata.values():
        event_lengths.append(shapes['x_shape'][0])
        procedure_lengths.append(shapes['procedure_shape'][0])
    
    def calc_stats(lengths):
        lengths_sorted = sorted(lengths)
        n = len(lengths_sorted)
        return {
            'min': min(lengths),
            'max': max(lengths),
            'mean': sum(lengths) // n,
            'median': lengths_sorted[n // 2],
            'p95': lengths_sorted[int(0.95 * n)],
            'p99': lengths_sorted[int(0.99 * n)]
        }
    
    return {
        'events': calc_stats(event_lengths),
        'procedures': calc_stats(procedure_lengths),
        'total_patients': len(metadata)
    }


def get_model_config_from_shape(
    shape_file: str,
    hidden_dim: int = 512,
    num_classes: int = 800,
    auxiliary_data_dim: int = 0,
    **kwargs
) -> Dict:
    """
    Generate complete model configuration from shape metadata.
    
    Args:
        shape_file: Path to shape.json file
        hidden_dim: Hidden dimension for transformer (default: 512)
        num_classes: Number of output classes (default: 800)
        auxiliary_data_dim: Dimension of auxiliary data (default: 0)
        **kwargs: Additional model parameters
        
    Returns:
        Dictionary with model configuration parameters
        
    Example:
        >>> config = get_model_config_from_shape('shape.json', hidden_dim=256)
        >>> model = EventTransformerClassifier(**config)
    """
    event_dim, procedure_dim = extract_dimensions(shape_file)
    stats = get_sequence_statistics(shape_file)
    
    config = {
        'input_dim': event_dim,
        'procedure_dim': procedure_dim,
        'hidden_dim': hidden_dim,
        'num_classes': num_classes,
        'auxiliary_data_dim': auxiliary_data_dim,
        # Add sequence statistics for reference
        '_sequence_stats': stats,
        # Add any additional parameters
        **kwargs
    }
    
    return config


def print_shape_summary(shape_file: str):
    """
    Print a summary of shape metadata for quick inspection.
    
    Args:
        shape_file: Path to shape.json file
    """
    event_dim, procedure_dim = extract_dimensions(shape_file)
    stats = get_sequence_statistics(shape_file)
    
    print("=" * 60)
    print("Shape Metadata Summary")
    print("=" * 60)
    print()
    print(f"Total Patients: {stats['total_patients']}")
    print()
    print("Feature Dimensions:")
    print(f"  Event features: {event_dim}")
    print(f"  Procedure features: {procedure_dim}")
    print()
    print("Event Sequence Lengths:")
    print(f"  Min: {stats['events']['min']}")
    print(f"  Max: {stats['events']['max']}")
    print(f"  Mean: {stats['events']['mean']}")
    print(f"  Median: {stats['events']['median']}")
    print(f"  95th percentile: {stats['events']['p95']}")
    print(f"  99th percentile: {stats['events']['p99']}")
    print()
    print("Procedure Sequence Lengths:")
    print(f"  Min: {stats['procedures']['min']}")
    print(f"  Max: {stats['procedures']['max']}")
    print(f"  Mean: {stats['procedures']['mean']}")
    print(f"  Median: {stats['procedures']['median']}")
    print(f"  95th percentile: {stats['procedures']['p95']}")
    print(f"  99th percentile: {stats['procedures']['p99']}")
    print()
    print("=" * 60)
    print()
    print("Recommended Model Configuration:")
    print("=" * 60)
    print(f"  input_dim={event_dim}")
    print(f"  procedure_dim={procedure_dim}")
    print(f"  hidden_dim=512  # Configurable")
    print(f"  num_classes=800  # Based on your task")
    print()
    print("Batching Recommendations:")
    print(f"  Consider padding to max_len={stats['events']['p99']} for events")
    print(f"  Consider padding to max_len={stats['procedures']['p99']} for procedures")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    
    # Default to shape.json in preprocess_event_data folder
    if len(sys.argv) < 2:
        shape_file = Path(__file__).parent.parent / "preprocess_event_data" / "shape.json"
    else:
        shape_file = sys.argv[1]
    
    if not Path(shape_file).exists():
        print(f"Error: {shape_file} not found")
        sys.exit(1)
    
    print_shape_summary(str(shape_file))
    
    # Example usage
    print("\nExample Usage:")
    print("=" * 60)
    print("from model.shape_utils import get_model_config_from_shape")
    print("from model.event_classifier import EventTransformerClassifier")
    print()
    print("config = get_model_config_from_shape('shape.json')")
    print("model = EventTransformerClassifier(**config)")
    print("=" * 60)
