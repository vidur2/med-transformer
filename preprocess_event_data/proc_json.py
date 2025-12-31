import json
import torch
from abstract import DataPoint, Category
from sentence_transformers import SentenceTransformer


class Event:
    # Class-level shared model instance
    _model = None
    
    @classmethod
    def _get_model(cls):
        """Lazy initialization of the shared sentence transformer model."""
        if cls._model is None:
            cls._model = SentenceTransformer('all-MiniLM-L6-v2')
        return cls._model
    
    def __init__(self, data: DataPoint, use_llm: bool = True):
        # Get the shared model instance
        model = self._get_model()
        
        self.parsed_vec = []
        for info in data.dump_contents():
            if (isinstance(info, Category)):
                self.parsed_vec.append(float(info.get_cat()))
            elif (type(info) == int or type(info) == bool):
                self.parsed_vec.append(float(info))
            elif (type(info) == str):
                # Convert string to embedding and extend the list
                embedding = model.encode(info)
                self.parsed_vec.extend(embedding.tolist())
            else:
                raise Exception("Cannot parse unknown data")
    
    def to_tensor(self) -> torch.Tensor:
        """Convert parsed_vec to a PyTorch tensor."""
        return torch.tensor(self.parsed_vec, dtype=torch.float32)

