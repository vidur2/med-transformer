
from abc import ABC, abstractmethod
from typing import Union, List

class Category:
    # Shared class-level list to track all categories
    _categories: List[str] = []
    
    def __init__(self, category_name: str):
        # Check if category already exists in the shared list
        if category_name in self._categories:
            self.cat_ord = self._categories.index(category_name)
        else:
            # Add new category to the list
            self._categories.append(category_name)
            self.cat_ord = len(self._categories) - 1
    
    def get_cat(self) -> int:
        return self.cat_ord

class DataPoint(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def dump_contents(self) -> List[Union[Category, int, float, bool]]:
        pass