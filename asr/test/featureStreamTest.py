from __future__ import generators

from asr.feature import *
from btk.subbandBeamforming import *

from Numeric import *

class IteratorTest:
    def __init__(self, len = 256):
        self._len = len

    def __iter__(self):
        cnt = 0
        while 1:
            f = zeros(self._len, Int16)
            for i in range(self._len):
                f[i] = i % self._len
            # print 'Yielding ', f
            yield array(f, Int16)
            if cnt >= 50:
                raise StopIteration
            cnt += 1
            print cnt

    def size(self):
        print 'Len = ', self._len
        return self._len

windowLen = 256
powN      = windowLen / 2 + 1
filterN   = 129
melN      = 30

i = IteratorTest(windowLen)
s = PyVectorShortFeatureStreamPtr(i)
h = HammingFeaturePtr(s)
f = FFTFeaturePtr(h, fftLen = windowLen)
m = PowerFeaturePtr(f, powN = powN)
e = MelFeaturePtr(m, powN = powN, filterN = melN)
l = LogFeaturePtr(e)
c = CepstralFeaturePtr(l)
a = AdjacentFeaturePtr(c)

fs = FeatureSetPtr()
fs.add(m)
fs.add(e)
fs.add(l)
fs.add(c)
fs.add(a)
ft = fs['Adjacent']

for feat in a:
    print feat
