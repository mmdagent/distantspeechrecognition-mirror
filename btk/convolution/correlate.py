#
#                           Beamforming Toolkit
#                                  (btk)
#
#   Module:  correlate.py
#   Purpose: Correlate the beamformed chirp signal with the original
#            waveform to extract the impulse response of the room + beamformer.
#   Date:    December 2, 2005
#   Author:  John McDonough

from btk.stream import *
from btk.common import *
from btk.feature import *
from btk.convolution import *

import wave

def corr():
    M = 512

    chirpFile    = './chirp_stetch7000.wav'

    sampleFile   = './beamformer-output.wav'
    outDataFile  = './impulse-response.wav'

    chirpFeature = SampleFeaturePtr(blockLen=M, shiftLen=M, padZeros=True)
    chirpFeature.read(chirpFile)
    chirpSignal  = chirpFeature.dataDouble() / 4.0E+11
    chirpSignal  = chirpSignal[::-1]

    sampleFeature = SampleFeaturePtr(blockLen=M, shiftLen=M, padZeros=1)
    overlapAdd    = OverlapAddAnalysisPtr(sampleFeature, chirpSignal)
    print '\n data \n'
    print overlapAdd.getNVar()
    print overlapAdd.getSampleLength()

    overlapSyn    = OverlapAddSynthesisPtr(overlapAdd.get(), overlapAdd.getIRLength(), overlapAdd.getSampleLength(), overlapAdd.getNVar())

    sampleFeature.read(sampleFile)
    sampN = sampleFeature.samplesN()

    # wavebuffer = []
    i = 0
    for b in overlapAdd:
        if i == 100:
            break

        print b[:10]
        # wavebuffer.extend(b)
        i += 1
    print overlapAdd.get()
#     # Write WAV file to disk
#     wavefile = wave.open(outDataFile, 'w')
#     wavefile.setnchannels(1)
#     wavefile.setsampwidth(2)
#     wavefile.setframerate(16000)

#     wavebuffer = asarray(wavebuffer[:sampN])
#     wavebuffer = asarray(wavebuffer)
#     # wavebuffer /= 4
#     wavefile.setnframes(sampN)
#     wavefile.writeframes(wavebuffer.astype('s').tostring())
#     wavefile.close()

corr()
