#!/usr/bin/python

from __future__ import generators

from Numeric import *
import pickle
import math
from select import select

from btk.modulated import *
from btk.plot import plotData
from btk.multiplot import *
from btk.signalGen import *
import sfe.common
from sfe.stream import *
from sfe.feature import *
from utils import *



M         = 512
m         = 4
windowLen = 10000

amplitude = 200

prototype = pickleLoadPrototype(M,m)

bb        = FilterbankBlackboardPtr(prototype, M, m, windowLen)


# signal = RawFileFeature("/home/merkosh/Projekte/Filterbank/signals/segment1-chan00-pcm.raw",
#                         windowLen=windowLen, windowShift=10)


# signal = FunctionFeature(windowLen,
#                          a=lambda _: amplitude,
#                          f=lambda t: sin(2*pi*t /windowLen),
#                          b=lambda a: 0.3*a,
#                          x=Sequence(),
#                          c=lambda a: 0,
#                          typecode=Float64)

# signal         = WaveFeature(windowLen, frequency=2.5, amplitude=amplitude, typecode=Float64)
# signal         = SineWaveFeature(windowLen, 2.05)



signal_tmp     = SampleFeaturePtr(blockLen=windowLen, shiftLen=windowLen, padZeros=1)
signal_tmp.read("/home/merkosh/Projekte/Filterbank/signals/segment1-chan00-pcm.sph",
                format=SF_FORMAT_WAV |SF_FORMAT_PCM_16)
signal         = FeatureAdapter(signal_tmp, typecode=Float64)


analysisFB     = SimpleAnalysisbankPtr(PyVectorFeatureStreamPtr(signal), bb)
synthesisFB    = SimpleSynthesisbankPtr(analysisFB, bb)

# signalIter     = iter(signal)
synthesisIter  = iter(synthesisFB)

signal_plot    = FeaturePlot(None, with="lines 3", title="signal")
output_plot    = FeaturePlot(None, with="lines 2", title="output")
error_plot     = FeaturePlot(None, with="lines 1", title="error [%]")

G = Multiplot((2,1), persist=1)
G('set grid')

G[(0,0)] = signal_plot
G[(1,0)] = output_plot
G[(2,0)] = error_plot

while (1):
    a = synthesisIter.next()
    b = signal.getFeature()
    print ".",
    sys.stdout.flush()
#     c = 100*abs(a-b)/max(abs(b))

#     output_plot.update(a)
#     signal_plot.update(b)
#     error_plot.update(c)

#     G.plot(0.1)
