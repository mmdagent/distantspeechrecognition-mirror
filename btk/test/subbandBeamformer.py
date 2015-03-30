#!/usr/bin/python

from btk import dbase
import sys
import os
import os.path

from btk.common import *
from btk.stream import *
from btk.feature import *
from btk.utils import *

from btk.modulated import *
from btk.subbandBeamforming import *

from utils import *
from utils import log as writeLog
from btk.signalGen import *


# Microphone array geometry
arrgeom = array([[   0. , -307.5,    0. ],
                 [   0. , -266.5,    0. ]])
#                  [   0. , -225.5,    0. ],
#                  [   0. , -184.5,    0. ],
#                  [   0. , -143.5,    0. ],
#                  [   0. , -102.5,    0. ],
#                  [   0. ,  -61.5,    0. ],
#                  [   0. ,  -20.5,    0. ]])



def procUtts():
    # Filter bank parameters
    nChan       = len(arrgeom)
    fftLen      = 512
    nBlocks     = 4
    subSampRate = 2
    sampleRate  = 16000.0
    sampleRate  = 44100.0

    # Source location
    x_s = 2000000.0
    y_s = 0.0
    z_s = 0.0

    # Filter bank parameters
    M                 = 512
    m                 = 4
    windowLen         = M
    prototype         = pickleLoadPrototype(M, m)

    blackboard = FilterbankBlackboardPtr(prototype, M, m, windowLen)

    # Paths for array data and database
    inDataDir   = "/home/merkosh/data/Chirp_and_Impuls/"
    outDataDir  = './wav'
    outfile     = "test.wav"

    # Build the analysis chain and MFCC generation
    sampleFeats = []
    beamformer = SubbandDSPtr(fftLen=2*M, halfBandShift=1)
    for _ in range(nChan):
#        sampleFeature = SampleFeaturePtr(blockLen=windowLen, shiftLen=windowLen, padZeros=1)
        sampleFeature = WaveFeature(windowLen, amplitude=10000, typecode=Float32)
        sampleFeats.append(sampleFeature)
        analysisFB    = SimpleAnalysisbankPtr(PyVectorFloatFeatureStreamPtr(sampleFeature), blackboard)
        beamformer.setChannel(analysisFB)

    delays = calcDelays(x_s, y_s, z_s, arrgeom)
    print "delays: ",delays
    beamformer.calcArrayManifoldVectors(sampleRate, delays)

    synthesisFB = SimpleSynthesisbankPtr(beamformer, blackboard)
    
#     # Load the waveform for each channel
#     for i in range(nChan):
#         infile = "%s/Chirp_and_Impuls%d.wav"%(inDataDir,i)
#         sampleFeats[i].read(infile, format = SF_FORMAT_WAV |SF_FORMAT_PCM_16)
#         sampN = sampleFeats[i].samplesN()
        
    # get reconstructed speech from beamformer
    wavebuffer = []
    i = 0
    for b in synthesisFB:
        wavebuffer.extend(b)
        print ".",; sys.stdout.flush()
        i += 1
        if (i == 50): break
        
    # Write WAV file to disk
    filename = "%s/%s"%(outDataDir, outfile)
    print "writing data to: %s"%filename
    
    saveWav(wavebuffer, filename)

    sys.exit()
    print "\n\nMFCCs all written\n\n"



if (__name__ == "__main__"):
    procUtts()
