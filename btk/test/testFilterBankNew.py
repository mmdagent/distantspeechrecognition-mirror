import string
import os
import wave

from btk.feature import *
from btk.modulated import *

def process():
    protoPath    = '/home/jmcd/experiments/1000/prototypes'
    samplePath   = '/home/jmcd/docs/2006/jobSearch/talks/audio/original/Headset/17_18/AMI_WSJ_OLAP_17_18-Headset2_T3c020w_T2c0207.wav'

    M    =   128
    m    =   2
    r    =   1

    R    = 2**r
    D    = M / R
    L    = 2 * M * m
    tau  = L/2
    v    = 0.01

    waveFileName = './wav/M=%d-r=%d-m=%d-tau=%d-v=%f-oversampled.wav' %(M, r, m, tau, v)

    sampleRate = 16000

    # Read analysis prototype 'h'
    protoFile = '%s/h-M=%d-m=%d-r=%d.txt' %(protoPath, M, m, r)
    print 'Loading analysis prototype from \'%s\'' %protoFile
    fp = open(protoFile, 'r')
    h = pickle.load(fp)
    fp.close()
    # print h

    # Read synthesis prototype 'g'
    protoFile = '%s/g-M=%d-m=%d-r=%d.txt' %(protoPath, M, m, r)
    print 'Loading synthesis prototype from \'%s\'' %protoFile
    fp = open(protoFile, 'r')
    g = pickle.load(fp)
    fp.close()
    # print g

    sampleFeature = SampleFeaturePtr(blockLen = D, shiftLen = D, padZeros = True)
    analysisBank  = FFTAnalysisBankFloatPtr(sampleFeature, prototype = h, M = M, m = m, r = r, halfBandShift = False)
    synthesisBank = FFTSynthesisBankPtr(analysisBank,      prototype = g, M = M, m = m, r = r, halfBandShift = False)

    if not os.path.exists(os.path.dirname(waveFileName)):
        os.makedirs(os.path.dirname(waveFileName))
    wavefile = wave.open(waveFileName, 'w')
    wavefile.setnchannels(1)
    wavefile.setsampwidth(2)
    wavefile.setframerate(int(sampleRate))

    print 'Processing data ...'
    sampleFeature.read(samplePath)
    for b in synthesisBank:
        # print b[:10]
        s = b * 1.0E+03
        # print s[:10]
        wavefile.writeframes(asarray(s, Float).astype('s').tostring())

    wavefile.close()

process()
