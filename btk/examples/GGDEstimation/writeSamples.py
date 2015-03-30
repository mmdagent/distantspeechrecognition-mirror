#!/usr/bin/python
#
#   Module:  writeSamples.py 
#   Purpose: write subband components into files
#   Date:    Feb. 11, 2010
#   Author:  
#   Required Module: myutils, sfe, btk

import string  
import os
import os.path
import shutil
import gzip
import pickle
from types import FloatType
import getopt, sys
from copy import *
import wave 
from myutils import *

from btk.common import *
from btk.stream import *
from btk.feature import *
from btk.matrix import *
from btk.utils import *

from btk.subbandBeamforming import *
from btk.sad import *
from btk.modulated import *
import btk.GGDEst
from btk.GGDEst import *

trace = 1

def accumObservations( spectralSource, sFrame, eFrame, R=1 ):
    """@brief accumulate observed subband components for adaptation """
    """@param sFrame: the start frame"""
    """@param eFrame: the end frame"""
    """@param R : R = 2**r, where r is a decimation factor"""
    """@return XL[frame][fftLen][chanN] : input subband snapshots"""

    fftLen = spectralSource.fftLen()
    observations = []
    # zero mean at this time... , mean = Numeric.zeros(chanN).astype(Numeric.Complex)
    snapShotArray = SnapShotArrayPtr( fftLen, 1 )
    if (trace > 0):
        print 'From %d to %d' %( sFrame, eFrame)

    #for sX in range(sFrame,eFrame):
    for sX in range(eFrame):
        sbSample = numpy.array(spectralSource.next())
        snapShotArray.newSample( sbSample, 0 )
        snapShotArray.update()
        if sX >= sFrame and sX < eFrame :
            X_t = [] # X_t[fftLen][chanN]
            if sX % R == 0:
                for fbinX in range(fftLen):
                    X_t.append( copy.deepcopy( snapShotArray.getSnapShot(fbinX) ) )
                observations.append( X_t )

    del snapShotArray
    return observations


def simpleVAD( wavefile, D, sampleRate, marginFrame = 3 ):
    sampleFeature  = SampleFeaturePtr(blockLen = D, shiftLen = D, padZeros = True )
    preemphFeature = PreemphasisFeaturePtr( sampleFeature, mu = 0.95 )
    energyFeature  = EnergyVADMetricPtr( preemphFeature, initialEnergy = 1.0e+07, threshold = 0.31, headN = 4, tailN = 10, energiesN = 800 )
    sampleFeature.read(wavefile)

    vadseq = []
    startFrame = -1
    frameX = 0    
    for vadflag in energyFeature:
        vadflag = int(vadflag)
        vadseq.append( vadflag )
        if vadflag > 0 and startFrame < 0 :
            startFrame = frameX - marginFrame
        frameX += 1

    endFrame = -1
    for frameX in range(len(vadseq)-1,-1,-1):
        if vadseq[frameX] > 0 and endFrame < 0 :
            endFrame = frameX + marginFrame
            break

    if startFrame < 0 :
        startFrame = 0

    if endFrame < 0 :
        endFrame = len(vadseq)-1

    if endFrame >= len(vadseq):
        endFrame = len(vadseq)-1
        
    return [startFrame,endFrame]
        
    
def writeSamples( segListFile, M, m, r, h_fb, IDPos ):
    
    fftLen = M
    halfBandShift = False
    R    = 2**r
    D    = M / R # frame shift
    sampleRate = 16000
    
    # Specify list of segments
    fileLst            = FGet(segListFile)

    for wavefile in fileLst:
        if not os.path.exists(wavefile):
            print 'Could not find file %s' %wavefile
            continue

        [startFrame,endFrame] = simpleVAD( wavefile, D, sampleRate )
        sampleFeature = SampleFeaturePtr(blockLen = D, shiftLen = D, padZeros = True )
        preemphFeature = PreemphasisFeaturePtr( sampleFeature, mu = 0.95 )
        analysisFB    = OverSampledDFTAnalysisBankPtr( preemphFeature, prototype = h_fb, M = M, m = m, r = r, delayCompensationType = 2 )
        sampleFeature.read(wavefile, samplerate = sampleRate, cfrom = startFrame * D, to = endFrame * D )
        if (trace > 0):
            print 'Load %s from %0.3f to %0.3f' %(wavefile, startFrame*D/float(sampleRate), endFrame*D/float(sampleRate) )

        # get number of samples of any file
        sampleN = sampleFeature.samplesN()
        sframe = 0
        eframe = sampleN/D - 1 #( sampleN/D-1 ) * 3 / 4
        samples = accumObservations( analysisFB, int(sframe), int(eframe), 1 )
        outputPrefix = getOutPrefix( wavefile, IDPos )
        print outputPrefix

        # write subband components
        buf1 = numpy.zeros( len(samples), numpy.complex )
        for fbinX in range(1,fftLen/2+1):
            for frX in range(len(samples)):
                buf1[frX] = samples[frX][fbinX][0]
            filename = '%s/Param_M=%d-m=%d-r=%d/%s/f%04d.sc.gz' % (outDataDir, M, m, r, outputPrefix, fbinX )
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            if not os.path.exists(filename):                    
                fp = gzip.open(filename, 'wb',1)
                pickle.dump(buf1, fp)
                fp.close()
            else:
                print '%s already exists' %(filename)


#####################
try:
    opts, args = getopt.getopt(sys.argv[1:], "hi:M:m:r:", ["help", "input=", "M=", "m=", "r=" ])
except getopt.GetoptError:
    # print help information and exit:
    print "Invalid options"
    sys.exit(2)

#######Default values
M  = 512 # fftLen / number of subbands
m  = 2   # m * M = length of window?
r  = 3   # decimation factor
IDPos = 3 # indicates the file path name which is used for the name of the output file.

# Data directories
exptDir    = '.' 
segListFile = '%s/trainL1' %exptDir
protoPath  = '../FB0000/prototype.ny' 
protoFile  = '%s/h-M=%d-m=%d-r=%d.txt' %(protoPath, M, m, r)
outDataDir = '%s/out' %exptDir

for o, a in opts:
    if o in ("-h", "--help"):
        sys.exit()
    elif o in ("-i", "--input"):
        segListFile = a
    elif o in ("-M", "--M"):
        M  = int(a)
    elif o in ("-m", "--m"):
        m  = int(a)
    elif o in ("-r", "--r"):
        r  = int(a)

# Read analysis prototype 'h'
protoFile = '%s/h-M=%d-m=%d-r=%d.txt' %(protoPath, M, m, r)
print 'Loading analysis prototype from \'%s\'' %protoFile
fp = open(protoFile, 'r')
h_fb = pickle.load(fp)
fp.close()

print 'SegList %s : M=%d m=%d r=%d' %(segListFile, M, m, r)
writeSamples( segListFile, M, m, r, h_fb, IDPos )

