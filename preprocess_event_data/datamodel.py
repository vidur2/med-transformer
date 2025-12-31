from abstract import DataPoint, Category
from json import loads
from typing import List



fieldMap = {
    'example': int
}

class EventDataPoint(DataPoint):
    def __init__(self, raw_data: str):
        parsed = list(loads(raw_data).items())
        parsed.sort(key = lambda x: x[0])
        self.ord = []
        for k, v in parsed:
            self.ord.append(fieldMap.get(k, str)(v))

    def dump_contents(self):
        return self.ord