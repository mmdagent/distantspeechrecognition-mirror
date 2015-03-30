from __future__ import generators
from filter import *
from btk.sound import *
from btk.plot import *

class MicArray(Filter):
    type = "SoundMicArray"

    def initParams(self):
        self.type = "SoundMicArray"
        self._params.add("buffersize", 1024, IntType)
        self._params.add("filelist", None, ListType)
        
    def setParams(self, dict):
        self._params.setDict(dict)
        buffersize = self._params["buffersize"]
        files = self._params["filelist"]
        self._soundMic = SoundMicArray(buffersize, files)

    def __iter__(self):
        for data in self._soundMic:
            yield data

class OneChannel(Filter):
    type = "onechannel"

    def initParams(self):
        self.type = "onechannel"
        self._params.add("channel", 0, IntType)

    def setParams(self, dict):
        self._params.setDict(dict)
        self._channel = self._params["channel"]

    def filter(self, data):
        channel = data[self._channel]
        return channel

class Fft(Filter):
    type = "fft"

    def initParams(self):
        self.type = "fft"
        
    def filter(self, data):
        import Numeric
        import FFT
        if type(data) == type(Numeric.array([])):
            fft = Numeric.absolute(FFT.fft(data))
        return fft

class Plot(Filter):
    type = "plot"

    def initParams(self):
        self.type = "plot"
        
    def filter(self, data):
        print type(data)
        plotData(data)
        return data

ff.register(MicArray)
ff.register(OneChannel)
ff.register(Fft)
ff.register(Plot)
        
