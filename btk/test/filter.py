from __future__ import generators
from types import *

class FilterFactory:
    def __init__(self):
        self._filters = {}

    def register(self, filter):
        self._filters[filter.type] = filter

    def produce(self, type):
        return self._filters[type]

    def __str__(self):
        retval = "Filters:"
        for f in self._filters.keys():
             retval = retval + "\n" + f
        return retval
    
class Params:
    def __init__(self):
        self._dict = {}
        self._params = {}

    def setDict(self, dict):
        self._dict = dict
        
    def add(self, name, default, type):
        self._params[name] = (default, type)

    def __getitem__(self, name):
        (default, type) = self._params[name]
        try:
            retval = self._dict[name]
        except KeyError:
            retval = default
        if type is IntType:
            retval = int(retval)
        elif type is FloatType:
            retval = float(retval)
        return retval

    def __str__(self):
        retval = ""
        for (n,p) in self._params.iteritems():
            retval += "Name: %s\tDefault: %s\tType: %s\n" % (
                n, p[0], p[1])
        return retval
    
class Filter:
    type = "Filter"

    def __init__(self, parents=None):
        if parents == None:
            parents = []
        self._parents = parents
        self._params = Params()
        self.initParams()

    def __str__(self):
        retval = "Filter: %s\nParameter:\n" % self.type
        retval += self._params.__str__()
        return retval

    def initParams(self):
        self.type = "Filter"

    def setParams(self, dict):
        pass

    def filter(self, data):
        pass

    def __iter__(self):
        iterators = []
        for i in self._parents:
            iterators.append(i.__iter__())
            
        while 1:
            data = []
            for i in iterators:
                data.append(i.next())
            yield self.filter(data)

class Source(Filter):
    type = "Source"

    def initParams(self):
        self.type = "Source"
        self._params.add("count", 10, IntType)
        self._params.add("delay", 0.1, FloatType)

    def setParams(self, dict):
        self._params.setDict(dict)
        self._count = self._params["count"]
        self._delay = self._params["delay"]

    def __iter__(self):
        import time
        for i in range(self._count):
            yield(i)
            time.sleep(self._delay)

class Adder(Filter):
    type = "Adder"
    
    def initParams(self):
        self.type = "Adder"
        self._params.add("x", 0, IntType)

    def setParams(self, dict):
        self._params.setDict(dict)
        self._x = self._params["x"]
        
    def filter(self, data):
        data = data[0]
        return  data + self._x

class Decorator(Filter):
    type = "Decorator"

    def initParams(self):
        self.type = "Decorator"
        self._params.add("char", "_", StringType)

    def setParams(self, dict):
        self._params.setDict(dict)
        self._char = self._params["char"]
        
    def filter(self, data):
        return "%s %s %s" % (self._char, data[0], self._char)

class Merger(Filter):
    type = "Merger"

    def initParams(self):
        self.type = "Merger"

    def filter(self, data):
        retval = ""
        for i in data:
            retval += "%s, " % i
        return retval[:-2]

ff = FilterFactory()
ff.register(Filter)
ff.register(Source)
ff.register(Adder)
ff.register(Decorator)
ff.register(Merger)

# this is how you use it manually

#f = ff.produce("Decorator")(ff.produce("Adder")(ff.produce("Source")()))
#f = Adder(Source())
#for i in f:
#    print i
