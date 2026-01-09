"""
Inference script for medical event classification.

Takes raw event data (as seen in processed_structured.json) and predicts target categories.

Usage:
    python inference.py --model_path checkpoints/best_model.pt --input_file example.json
    python inference.py --model_path checkpoints/best_model.pt --interactive
"""

import argparse
import json
import torch
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'preprocess_event_data'))

from model.event_classifier import EventTransformerClassifier
from preprocess_event_data.datamodel import EventDataPoint, TargetCategory, fieldMap
from preprocess_event_data.proc_json import Event


class MedicalEventPredictor:
    """Inference wrapper for medical event classification."""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        
        # Load checkpoint
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model arguments
        args = checkpoint.get('args', None)
        if args is None:
            raise ValueError("Checkpoint missing 'args'. Cannot reconstruct model.")
        print(args)
        # Initialize model
        self.model = EventTransformerClassifier(
            input_dim=args['event_dim'],
            procedure_dim=args['procedure_dim'],
            num_classes=args['num_classes'],
            hidden_dim=args['hidden_dim'],
            num_heads=args['num_heads'],
            num_transformer_layers=args['num_layers'],
            dropout=args['dropout']
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✓ Model loaded successfully")
        print(f"  Event dim: {args['event_dim']}")
        print(f"  Procedure dim: {args['procedure_dim']}")
        print(f"  Num classes: {args['num_classes']}")
        
        self.num_classes = args['num_classes']
        
        # Load target category mappings
        self.load_target_categories()
    
    def load_target_categories(self):
        """Load target category mappings."""
        target_cat_path = Path(__file__).parent / 'data' / 'target_categories.json'
        
        if target_cat_path.exists():
            with open(target_cat_path, 'r') as f:
                data = json.load(f)
            self.category_names = data['categories']
            self.category_to_idx = data['category_to_index']
            self.idx_to_category = {idx: cat for cat, idx in self.category_to_idx.items()}
            print(f"✓ Loaded {len(self.category_names)} target categories")
        else:
            print(f"Warning: {target_cat_path} not found. Using indices only.")
            self.category_names = [f"class_{i}" for i in range(self.num_classes)]
            self.idx_to_category = {i: f"class_{i}" for i in range(self.num_classes)}
    
    def preprocess_event(self, raw_event: Dict) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Preprocess a raw event into tensors.
        
        Args:
            raw_event: Raw event dictionary (as in processed_structured.json)
        
        Returns:
            Tuple of (event_tensor, list of procedure tensors)
        """
        # Create EventDataPoint
        event_dp = EventDataPoint(raw_event, fieldMap)
        
        # Convert event to tensor
        event_tensor = Event(event_dp, field_names=event_dp.field_names, schema_type='event').to_tensor()
        
        # Convert procedures to tensors
        procedure_tensors = []
        for proc in event_dp.procs:
            proc_tensor = Event(proc, field_names=getattr(proc, 'field_names', None), schema_type='procedure').to_tensor()
            procedure_tensors.append(proc_tensor)
        
        # Add placeholder procedure if none exist (to avoid crashes)
        if len(procedure_tensors) == 0:
            proc_dim = 385  # Default
            procedure_tensors.append(torch.zeros(proc_dim))
        
        return event_tensor, procedure_tensors
    
    def predict(self, raw_event: Dict, top_k: int = 10, threshold: float = 0.5) -> Dict:
        """
        Make predictions for a single event.
        
        Args:
            raw_event: Raw event dictionary
            top_k: Number of top predictions to return
            threshold: Probability threshold for binary predictions
        
        Returns:
            Dictionary with predictions
        """
        # Preprocess
        event_tensor, procedure_tensors = self.preprocess_event(raw_event)
        
        # Add batch dimension and move to device
        event_batch = event_tensor.unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, event_dim)
        
        # Stack procedures
        if len(procedure_tensors) > 0:
            procedure_batch = torch.stack(procedure_tensors).unsqueeze(0).to(self.device)  # (1, num_procs, proc_dim)
        else:
            procedure_batch = torch.zeros(1, 1, 385).to(self.device)
        
        # Create masks (no padding for single event)
        event_mask = torch.zeros(1, 1, dtype=torch.bool).to(self.device)
        proc_mask = torch.zeros(1, procedure_batch.shape[1], dtype=torch.bool).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            output = self.model(
                event_batch,
                procedure_x=procedure_batch,
                src_key_padding_mask=event_mask,
                procedure_padding_mask=proc_mask
            )  # (1, 1, num_classes)
        
        # Get probabilities
        probs = torch.sigmoid(output[0, 0]).cpu().numpy()  # (num_classes,)
        
        # Get top-k predictions
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_probs = probs[top_indices]
        
        top_predictions = []
        for idx, prob in zip(top_indices, top_probs):
            category = self.idx_to_category.get(idx, f"class_{idx}")
            top_predictions.append({
                'category': category,
                'probability': float(prob),
                'index': int(idx)
            })
        
        # Get all predictions above threshold
        above_threshold_indices = np.where(probs > threshold)[0]
        threshold_predictions = []
        for idx in above_threshold_indices:
            category = self.idx_to_category.get(idx, f"class_{idx}")
            threshold_predictions.append({
                'category': category,
                'probability': float(probs[idx]),
                'index': int(idx)
            })
        
        # Sort threshold predictions by probability
        threshold_predictions = sorted(threshold_predictions, key=lambda x: x['probability'], reverse=True)
        
        return {
            'top_k': top_predictions,
            'above_threshold': threshold_predictions,
            'num_procedures': len(procedure_tensors),
            'max_probability': float(probs.max()),
            'mean_probability': float(probs.mean()),
        }
    
    def predict_sequence(self, raw_events: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        Make predictions for a sequence of events (patient history).
        
        Args:
            raw_events: List of raw event dictionaries
            top_k: Number of top predictions per event
        
        Returns:
            List of prediction dictionaries, one per event
        """
        predictions = []
        
        for i, raw_event in enumerate(raw_events):
            print(f"\nProcessing event {i+1}/{len(raw_events)}...")
            pred = self.predict(raw_event, top_k=top_k)
            predictions.append(pred)
        
        return predictions


def load_example_from_file(file_path: str) -> Dict:
    """Load example event from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # If it's a list, take the first item
    if isinstance(data, list):
        return data[0]
    return data


def interactive_mode(predictor: MedicalEventPredictor):
    """Interactive mode for entering event data."""
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("\nEnter event data as JSON (or 'quit' to exit)")
    print("Example format:")
    print(json.dumps({
        "PAITENT_ID": "12345",
        "AGE": 65,
        "SEX": "M",
        "PRIMARY_DIAGNOSIS": "Heart Failure",
        "PRIMARY_DIAGNOSIS_DESCRIPTION": "Congestive heart failure, unspecified",
        "procedures": [
            {"name": "Echocardiogram", "variable_description": "Cardiac imaging"}
        ]
    }, indent=2))
    print("\n" + "-" * 60 + "\n")
    
    while True:
        try:
            print("Enter event JSON (single line or type 'file:<path>' to load from file):")
            user_input = input("> ").strip()
            
            if user_input.lower() == 'quit':
                print("Exiting...")
                break
            
            if user_input.startswith('file:'):
                file_path = user_input[5:].strip()
                raw_event = load_example_from_file(file_path)
            else:
                raw_event = json.loads(user_input)
            
            # Make prediction
            print("\nMaking prediction...")
            results = predictor.predict(raw_event, top_k=10, threshold=0.3)
            
            # Display results
            print("\n" + "=" * 60)
            print("PREDICTION RESULTS")
            print("=" * 60)
            print(f"\nNumber of procedures: {results['num_procedures']}")
            print(f"Max probability: {results['max_probability']:.4f}")
            print(f"Mean probability: {results['mean_probability']:.4f}")
            
            print(f"\nTop 10 predictions:")
            print("-" * 60)
            for i, pred in enumerate(results['top_k'], 1):
                print(f"{i}. {pred['category']}")
                print(f"   Probability: {pred['probability']:.4f}")
            
            if results['above_threshold']:
                print(f"\nPredictions above threshold (0.3):")
                print("-" * 60)
                for pred in results['above_threshold']:
                    print(f"- {pred['category']}: {pred['probability']:.4f}")
            else:
                print("\nNo predictions above threshold (0.3)")
            
            print("\n" + "=" * 60 + "\n")
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Medical Event Classification Inference')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--input_file', type=str, default=None,
                       help='Path to input JSON file with event data')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Path to save predictions (JSON)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--top_k', type=int, default=10,
                       help='Number of top predictions to return')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Probability threshold for predictions')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to run on (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = MedicalEventPredictor(args.model_path, device=args.device)
    
    if args.interactive:
        # Interactive mode
        interactive_mode(predictor)
    
    elif args.input_file:
        # Batch mode
        print(f"\nLoading input from {args.input_file}...")
        
        with open(args.input_file, 'r') as f:
            data = json.load(f)
        
        # Check if it's a single event or list of events
        if isinstance(data, list):
            print(f"Processing {len(data)} events...")
            predictions = predictor.predict_sequence(data, top_k=args.top_k)
        else:
            print("Processing single event...")
            predictions = [predictor.predict(data, top_k=args.top_k, threshold=args.threshold)]
        
        # Display results
        for i, pred in enumerate(predictions, 1):
            print(f"\n{'='*60}")
            print(f"Event {i} - Prediction Results")
            print('='*60)
            print(f"Number of procedures: {pred['num_procedures']}")
            print(f"Max probability: {pred['max_probability']:.4f}")
            
            print(f"\nTop {args.top_k} predictions:")
            for j, p in enumerate(pred['top_k'], 1):
                print(f"  {j}. {p['category']}: {p['probability']:.4f}")
            
            if pred['above_threshold']:
                print(f"\nPredictions above threshold ({args.threshold}):")
                for p in pred['above_threshold']:
                    print(f"  - {p['category']}: {p['probability']:.4f}")
        
        # Save to file if specified
        if args.output_file:
            output_data = {
                'predictions': predictions,
                'model_path': args.model_path,
                'input_file': args.input_file,
                'top_k': args.top_k,
                'threshold': args.threshold
            }
            
            with open(args.output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"\n✓ Saved predictions to {args.output_file}")
    
    else:
        print("Error: Must specify either --input_file or --interactive")
        parser.print_help()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
