import btk.common
from btk.modulated import *
from numpy import *
import pickle
import math

from btk.stream import *
from btk.feature import *

class SineWaveFeature:
    def __init__(self, windowLen, freq):
        self._windowLen = windowLen
        self._freq      = freq

    def __iter__(self):
        print 'Iterating ... '
        count = 0
        v = zeros(windowLen, Int16)
        for j in range(40):
            for i in range(self._windowLen):
                val = 1000.0 * cos(2.0 * pi * self._freq *
                                   (float(count) / self._windowLen))
                v[i] = int(val)
                count += 1


            print 'Original:'
            print v
            yield v
        raise StopIteration

    def reset(self):
        print 'Passing ... '
        pass

    def size(self):
        return self._windowLen

M         = 8
m         = 4
d         = 1
windowLen = 8

prototypeFileName = './prototype_M'+str(M)+'_m'+str(m)

fp = open(prototypeFileName, 'r')
prototype = pickle.load(fp)
fp.close()

freq_c = 0.05

sampleFeature = SineWaveFeature(windowLen = windowLen, freq = freq_c)

# Analysis chain for FFT Filter Bank
analysisFB    = FFTAnalysisBankShortPtr(PyVectorShortFeatureStreamPtr(sampleFeature), prototype = prototype, M = M, m = m, d = d)
synthesisFB   = FFTSynthesisBankPtr(analysisFB, prototype = prototype, M = M, m = m, d = d)

for outSamp in synthesisFB:
    print 'Reconstruction:'
    print outSamp
