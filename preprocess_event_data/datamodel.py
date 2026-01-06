from abstract import DataPoint, Category
from json import loads
from typing import List

DATE = 'DATE'


class TargetCategory(Category):
    pass

TargetCategory._categories = []


class VariableCategory(Category):
    pass

VariableCategory._categories = []

fieldMap = {
    'example': int,
    'procedures': 'EventDataPoint',
    "DIAGNOSIS_RISK_VARIABLE_DESCRIPTION": 'pl'
}

fieldMapProc = {
    'name': str,
    'variable_description': VariableCategory
}

class EventDataPoint(DataPoint):
    def __init__(self, raw_data: dict, fieldMap: dict = fieldMap):
        self.ord = []
        self.tgt = []
        for k, v in raw_data.items():
            if (k not in fieldMap):
                continue
            if (fieldMap[k] == DATE):
                self.ord.append(datetime.datetime.strptime(v, "%Y-%m-%d")).timestamp()
            elif (type(v) == list):
                tmp = []
                if (fieldMap[k] == 'EventDataPoint'):
                    for i in v:
                        tmp.append(EventDataPoint(i, fieldMapProc))
                    self.ord.append(tmp)
                else:
                    self.tgt = [TargetCategory(i) for i in v]
            else:
                self.ord.append(fieldMap.get(k, str)(v))

    def dump_contents(self):
        return self.ord

if (__name__ == '__main__'):
    blob = ''
    with open('/Users/vidurmodgil/Desktop/Data/Programming Projects/mother/data/processed_structured.json') as f:
        blob = f.read()
    # print(loads(blob)[0]['procedures'])
    e = EventDataPoint(loads(blob)[0], fieldMap)
    print(e.tgt[0].get_cat())
    print(e.ord[0][0].ord[1].get_cat())