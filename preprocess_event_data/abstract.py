
from abc import ABC, abstractmethod
from typing import Union, List

class Category:
    def __init__(self, cat_ord: int):
        self.cat_ord = cat_ord
    
    def get_cat(self) -> int:
        return self.cat_ord

class DataPoint(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def dump_contents(self) -> List[Union[Category, int, float, bool]]:
        pass