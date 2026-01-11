from abstract import DataPoint, Category
from json import loads, dump
from typing import List
from generated_categories.fieldmap_generated import fieldMap as fieldMapExt
from proc_json import Event
import torch

DATE = 'DATE'


class TargetCategory(Category):
    pass

TargetCategory._categories = []


class VariableCategory(Category):
    pass

VariableCategory._categories = []

fieldMap = {
    'procedures': 'EventDataPoint',
    'event': 'pl'
}

fieldMapExt.update(fieldMap)

fieldMap = fieldMapExt


fieldMapProc = {
    'name': str,
    'variable_description': TargetCategory
}

class EventDataPoint(DataPoint):
    def __init__(self, raw_data: dict, fieldMap: dict = fieldMap):
        self.ord = []
        self.tgt = []
        self.procs = []
        self.patid = -1
        self.field_names = []  # Track field names in order
        for k, v in raw_data.items():
            if (k == 'variable_description'):
                self.tgt.append(TargetCategory(v))
                continue
            if (k == 'PAITENT_ID'):
                self.patid = v
                continue
            if (k not in fieldMap):
                continue
            if (type(v) == list):
                if (fieldMap[k] == 'EventDataPoint'):
                    for i in v:
                        dp = EventDataPoint(i, fieldMapProc)
                        self.procs.append(dp)
                        self.tgt.extend(dp.tgt)
                else:
                    self.tgt = [TargetCategory(i) for i in v]
            else:
                if (fieldMap.get(k, str) in [float, int] and v is None):
                    self.ord.append(fieldMap.get(k, str)(-1))
                    self.field_names.append(k)
                else:
                    self.ord.append(fieldMap.get(k, str)(v))
                    self.field_names.append(k)

    def dump_contents(self):
        return self.ord

if (__name__ == '__main__'):
    blob = ''
    with open('data/processed_structured.json') as f:
        blob = f.read()
    assoc_patid = dict()
    for dp in loads(blob):
        e = EventDataPoint(dp, fieldMap)
        tgt = torch.zeros(800)
        for i in e.tgt:
            tgt[i.get_cat()] = 1.0  # Standard multi-label format (don't normalize)
        
        # Ensure each event has at least one procedure (add empty placeholder if needed)
        # This prevents crashes when collating batches with all-empty procedure sequences
        procedures_list = []
        for p in e.procs:
            procedures_list.append(Event(p, field_names=getattr(p, 'field_names', None), schema_type='procedure').to_tensor())
        
        # If no procedures, add a zero-filled placeholder procedure
        if len(procedures_list) == 0:
            # Get procedure dimension from existing procedures or use default
            proc_dim = 384  # Default procedure dimension
            procedures_list.append(torch.zeros(proc_dim))
        
        if (e.patid not in assoc_patid):
            assoc_patid[e.patid] = {
                'events': [Event(e, field_names=e.field_names, schema_type='event').to_tensor()],
                'procedures': [procedures_list],
                'targets': [tgt]
            }
        else:
            assoc_patid[e.patid]['events'].append(Event(e, field_names=e.field_names, schema_type='event').to_tensor())
            assoc_patid[e.patid]['procedures'].append(procedures_list)
            assoc_patid[e.patid]['targets'].append(tgt)
    
    # Convert tensors to lists for JSON serialization
    output_data = {}
    for patient_id, patient_data in assoc_patid.items():
        output_data[patient_id] = {
            'events': [event.tolist() for event in patient_data['events']],
            'procedures': [[proc.tolist() for proc in event_procs] for event_procs in patient_data['procedures']],
            'targets': [target.tolist() for target in patient_data['targets']]
        }
    
    # Save to JSON
    output_path = 'data/tensor_data.json'
    with open(output_path, 'w') as f:
        dump(output_data, f, indent=2)
    
    print(f"Saved tensor data for {len(output_data)} patients to {output_path}")
    
    # Export text feature ranges for use in training
    text_ranges_path = 'data/text_ranges.json'
    Event.export_text_ranges(text_ranges_path)
    
    # Display detected text feature ranges for each schema
    print("\nDetected text feature ranges:")
    for schema_type in ['event', 'procedure']:
        ranges = Event.get_text_feature_ranges(schema_type)
        if ranges:
            print(f"  {schema_type}:")
            for field_name, (start, end) in ranges.items():
                print(f"    {field_name}: [{start}, {end})")
        else:
            print(f"  {schema_type}: (no text features)")
    
    # Serialize TargetCategory mappings
    target_categories_path = 'data/target_categories.json'
    target_category_data = {
        'num_categories': len(TargetCategory._categories),
        'categories': TargetCategory._categories,
        'category_to_index': {cat: idx for idx, cat in enumerate(TargetCategory._categories)}
    }
    with open(target_categories_path, 'w') as f:
        dump(target_category_data, f, indent=2)
    
    print(f"\n✓ Saved {len(TargetCategory._categories)} target categories to {target_categories_path}")

