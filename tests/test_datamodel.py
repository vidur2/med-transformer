"""
Tests for EventDataPoint from datamodel module.
"""
import sys
import os
sys.path.insert(0, 'preprocess_event_data')

import torch
from datamodel import EventDataPoint, fieldMap
from proc_json import Event


def test_simple_json_parsing():
    """Test that EventDataPoint correctly parses simple JSON."""
    print("Test 1: Simple JSON parsing")
    print("-" * 50)
    
    json_data = '{"example": 42}'
    event = EventDataPoint(json_data)
    contents = event.dump_contents()
    
    print(f"Input JSON: {json_data}")
    print(f"Parsed contents: {contents}")
    
    assert len(contents) == 1, f"Expected 1 field, got {len(contents)}"
    assert contents[0] == 42, f"Expected 42, got {contents[0]}"
    assert isinstance(contents[0], int), f"Expected int, got {type(contents[0])}"
    
    print("✓ Simple JSON parsing test passed!\n")


def test_alphabetical_sorting():
    """Test that fields are sorted alphabetically by key."""
    print("Test 2: Alphabetical sorting of fields")
    print("-" * 50)
    
    json_data = '{"zebra": 100, "apple": 25, "example": 50}'
    event = EventDataPoint(json_data)
    contents = event.dump_contents()
    
    print(f"Input JSON: {json_data}")
    print(f"Parsed contents: {contents}")
    
    # Should be sorted: apple (str), example (int), zebra (str)
    assert len(contents) == 3, f"Expected 3 fields, got {len(contents)}"
    assert contents[0] == '25', f"Expected '25' for apple, got {contents[0]}"
    assert contents[1] == 50, f"Expected 50 for example, got {contents[1]}"
    assert contents[2] == '100', f"Expected '100' for zebra, got {contents[2]}"
    
    print("✓ Alphabetical sorting test passed!\n")


def test_field_mapping():
    """Test that fieldMap correctly maps types."""
    print("Test 3: Field type mapping")
    print("-" * 50)
    
    json_data = '{"example": 789, "description": "patient diagnosis", "status": "active"}'
    event = EventDataPoint(json_data)
    contents = event.dump_contents()
    
    print(f"Input JSON: {json_data}")
    print(f"Parsed contents: {contents}")
    
    # Should be sorted: description (str), example (int), status (str)
    assert len(contents) == 3, f"Expected 3 fields, got {len(contents)}"
    assert isinstance(contents[0], str), f"Expected str for description, got {type(contents[0])}"
    assert isinstance(contents[1], int), f"Expected int for example, got {type(contents[1])}"
    assert isinstance(contents[2], str), f"Expected str for status, got {type(contents[2])}"
    assert contents[1] == 789, f"Expected 789 for example, got {contents[1]}"
    
    print("✓ Field mapping test passed!\n")


def test_empty_json():
    """Test that EventDataPoint handles empty JSON objects."""
    print("Test 4: Empty JSON handling")
    print("-" * 50)
    
    json_data = '{}'
    event = EventDataPoint(json_data)
    contents = event.dump_contents()
    
    print(f"Input JSON: {json_data}")
    print(f"Parsed contents: {contents}")
    
    assert len(contents) == 0, f"Expected 0 fields, got {len(contents)}"
    
    print("✓ Empty JSON test passed!\n")


def test_eventdatapoint_to_tensor():
    """Test converting EventDataPoint through Event to tensor."""
    print("Test 5: EventDataPoint to tensor conversion")
    print("-" * 50)
    
    json_data = '{"category": "diagnosis", "example": 123, "patient_id": "12345"}'
    event_dp = EventDataPoint(json_data)
    event = Event(event_dp)
    tensor = event.to_tensor()
    
    print(f"Input JSON: {json_data}")
    print(f"EventDataPoint contents: {event_dp.dump_contents()}")
    print(f"Tensor shape: {tensor.shape}")
    
    assert tensor.dim() == 1, f"Expected 1D tensor, got {tensor.dim()}D"
    assert tensor.shape[0] > 0, f"Expected non-empty tensor, got shape {tensor.shape}"
    
    print("✓ Tensor conversion test passed!\n")


def test_numeric_values():
    """Test that numeric values are properly handled."""
    print("Test 6: Numeric value handling")
    print("-" * 50)
    
    json_data = '{"example": 999, "count": 42, "ratio": 3.14}'
    event = EventDataPoint(json_data)
    contents = event.dump_contents()
    
    print(f"Input JSON: {json_data}")
    print(f"Parsed contents: {contents}")
    
    # Should be sorted: count (str), example (int), ratio (str)
    assert len(contents) == 3, f"Expected 3 fields, got {len(contents)}"
    assert contents[0] == '42', f"Expected '42' for count, got {contents[0]}"
    assert contents[1] == 999, f"Expected 999 for example, got {contents[1]}"
    assert isinstance(contents[1], int), f"Expected int for example, got {type(contents[1])}"
    
    print("✓ Numeric value handling test passed!\n")


def test_consistency_across_instances():
    """Test that multiple instances with same structure produce consistent results."""
    print("Test 7: Consistency across instances")
    print("-" * 50)
    
    json_data_1 = '{"example": 100, "name": "test1"}'
    json_data_2 = '{"example": 200, "name": "test2"}'
    
    event_1 = EventDataPoint(json_data_1)
    event_2 = EventDataPoint(json_data_2)
    
    contents_1 = event_1.dump_contents()
    contents_2 = event_2.dump_contents()
    
    print(f"Instance 1 JSON: {json_data_1}")
    print(f"Instance 1 contents: {contents_1}")
    print(f"Instance 2 JSON: {json_data_2}")
    print(f"Instance 2 contents: {contents_2}")
    
    assert len(contents_1) == len(contents_2), "Instances have different field counts"
    assert type(contents_1[0]) == type(contents_2[0]), "Field types don't match"
    assert type(contents_1[1]) == type(contents_2[1]), "Field types don't match"
    
    print("✓ Consistency test passed!\n")


if __name__ == '__main__':
    print("=" * 50)
    print("DATAMODEL TESTS: EventDataPoint")
    print("=" * 50)
    print()
    
    test_simple_json_parsing()
    test_alphabetical_sorting()
    test_field_mapping()
    test_empty_json()
    test_eventdatapoint_to_tensor()
    test_numeric_values()
    test_consistency_across_instances()
    
    print("=" * 50)
    print("ALL DATAMODEL TESTS PASSED! ✓")
    print("=" * 50)
