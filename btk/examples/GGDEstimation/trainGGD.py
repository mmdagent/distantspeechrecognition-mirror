#!/usr/bin/python
#
# 			          Millennium
#                     Automatic Speech Recognition System
#                                   (asr)
#
#   Module:  mnBeamformingBatch.py 
#   Purpose: Compute the optimized beamforming weights
#   Date:    March 7, 2008
#   Author:  CWBF

import sys
import string  
import os
import os.path
import shutil
import gzip
import pickle
import re
from types import FloatType
import getopt, sys
from myutils import *

from btk.common import *
from btk.stream import *
from btk.feature import *
from btk.matrix import *
from btk.utils import *

import numpy
#from pygsl import *
import pygsl
from pygsl import sf
from pygsl import minimize, errno
from pygsl import _numobj as numx

#from btk import dbase
from btk.modulated import *
#from btk.subbandBeamforming import *
from btk.beamformer import *
import btk.GGDcEst
from btk.GGDcEst import *

trace = 1
IDPos = 3

# @brief calculate the variance and set the deviation to the scale parameter
def calcVariance( cggdEstimator, fbinX, dataDir, segListFile, M, m, r  ):
    # Specify list of segments
    fileLst = open( segListFile, 'r' )
    nSamples = 0 # the number of training samples
    sumOfY2  = 0 # the sum of the variance

    for nextFile in fileLst:
        outputPrefix = getOutPrefix( nextFile, IDPos )

        # load subband components
        scfilename = '%s/Param_M=%d-m=%d-r=%d/%s/f%04d.sc.gz' %(dataDir, M, m, r, outputPrefix, fbinX )
        if not os.path.exists(scfilename):
            print 'Could not find file %s' %scfilename
            continue
        fp = gzip.open(scfilename, 'rb',1)
        samples_f = pickle.load(fp)
        fp.close()

        frameN = len(samples_f)
        for frX in range(frameN):
            absY = abs( samples_f[frX] )
            Y2   = absY * absY
            sumOfY2  += Y2
            nSamples += 1
                    
    fileLst.close()
    var = sumOfY2 / float( nSamples )
    cggdEstimator.setScalingPar( var )
    cggdEstimator.fixConst()

    if trace > 0:
        print 'The variance %f / %d = %f' %(sumOfY2,nSamples,var)

    return cggdEstimator


# @brief solve the scale parameter based on the maximum likelihood criterion
def MLEstimateScaleParameter( cggdEstimator, fbinX, dataDir, segListFile, M, m, r ):
    # Specify list of segments
    fileLst = open( segListFile, 'r' )
    nSamples = 0

    for nextFile in fileLst:
        outputPrefix = getOutPrefix( nextFile, IDPos )

        # load subband components
        scfilename = '%s/Param_M=%d-m=%d-r=%d/%s/f%04d.sc.gz' %(dataDir, M, m, r, outputPrefix, fbinX )
        if not os.path.exists(scfilename):
            print 'Could not find file %s' %scfilename
            continue
        fp = gzip.open(scfilename, 'rb',1)
        samples_f = pickle.load(fp)
        fp.close()

        frameN = len(samples_f)
        for frX in range(frameN):
            Y = samples_f[frX]
            cggdEstimator.acc( Y )
            nSamples +=1
                    
    fileLst.close()
    cggdEstimator.update( '', True )
    cggdEstimator.fixConst()

    return cggdEstimator


def calcNegLogLikelihood( p, (cggdEstimator, fbinX, dataDir, segListFile, M, m, r ) ):
    
    # Specify list of segments
    # compute the log gamma in order to avoid the overflow 
    lg1 = sf.lngamma( 1.0 / p )
    lg2 = sf.lngamma( 2.0 / p )
    lB  = lg1[0] - lg2[0]
    sa  = cggdEstimator.getScalingPar()
    nSamples = 0
    sumNrmYp = 0.0

    # Specify list of segments
    fileLst = open( segListFile, 'r' )
    for nextFile in fileLst:
        outputPrefix = getOutPrefix( nextFile, IDPos )

        # load subband components
        scfilename = '%s/Param_M=%d-m=%d-r=%d/%s/f%04d.sc.gz' % (dataDir, M, m, r, outputPrefix, fbinX )
        if not os.path.exists(scfilename):
            print 'Could not find file %s' %scfilename
            continue
        fp = gzip.open(scfilename, 'rb',1)
        samples_f = pickle.load(fp)
        fp.close()

        frameN = len(samples_f)
        for frX in range(frameN):
            absY   = abs( samples_f[frX] )
            Y2     = absY * absY
            lnrmYp = p * ( numpy.log(Y2) - lB - numpy.log(sa) )
            nrmYp  = numpy.exp( lnrmYp )
            sumNrmYp += nrmYp
            nSamples +=1

    fileLst.close()

    constT =  numpy.log( p ) - numpy.log( numpy.pi ) - lg1[0] - numpy.log(sa) - lB
    loglikelihood = nSamples * constT - sumNrmYp
    if trace > 0:
        print "Loglikelihood %f : p %f" %(loglikelihood,p)

    return  ( -1 * loglikelihood )


try:
    opts, args = getopt.getopt(sys.argv[1:], "h:i:M:m:r:l:u:", ["help", "input=", "M=", "m=", "r=", "lower=","upper=" ])
except getopt.GetoptError:
    # print help information and exit:
    print "Invalid options"
    sys.exit(2)

#######Default values
exptDir     = '.' 
fbinXList   = '%s/fbinL' %exptDir
segListFile = '%s/trainL2' %exptDir
dataDir     = 'out'
outDataDir  = 'out'
initialLow  = 0.0005
initialHigh = 1.8
initialP = 0.5
ggdPath = ''
nIter1    = 30 # the number of iterations for the estimation of the entire parameters
nIter2    = 25 # the number of iterations for the linear search
skipSize1 = 20
skipSize2 = 25

M  = 512 #fftLen
m  = 2
r  = 3
fbinStart = 1
fbinEnd   = M/2+1

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
    elif o in ("-l", "--lower"):
        initialLow  = float(a)
    elif o in ("-u", "--upper"):
        initialHigh = float(a)

# main part
print 'SegList %s : M=%d m=%d r=%d' %(segListFile, M, m, r)

FBinL = FGet(fbinXList)
for fbinXStr in FBinL:
    fbinX = int(fbinXStr)

    low  = initialLow
    high = initialHigh
    print 'fbinX=%d :' %(fbinX)    
    filename = '%s/LS_M=%d-m=%d-r=%d/FINAL/_M-%04d' %(outDataDir, M, m, r, fbinX)
    if os.path.exists(filename):
        print '%s already exists. Exit' %(filename)
        continue
    print filename

    cggdEstimator = MLE4CGGaussianD()
    # set the initali parameters if the file exists
    ggdFile = '%s_M-%04d' %(ggdPath,fbinX)
    if os.path.exists(ggdFile):
        print 'read initial parameters from %s' %ggdFile
        cggdEstimator.readParam( ggdFile, zeroMean=True, fixValues=False )
    else:
        print '%f is the initial value of the shape parameter' %initialP

    cggdEstimator = calcVariance( cggdEstimator, fbinX, dataDir, segListFile, M, m, r )

    # seek the shape parameter which provides the ML with the linear search
    for iterX1 in range(nIter1):
        low  = initialLow
        high = initialHigh
 
        mysys = minimize.gsl_function( calcNegLogLikelihood, [cggdEstimator, fbinX, dataDir, segListFile, M, m, r ] )
        mid  = cggdEstimator.getShapePar()
        if  ( mid <  low ) and ( mid > high ):
            mid = 0.7

        print "Start %d : %e %e %e" %(iterX1, low, high, mid)
        #minimizer = minimize.brent(mysys)
        minimizer = minimize.goldensection(mysys)
        minimizer.set(mid, low, high)

        #print "# Using  minimizer ", minimizer.name()
        for iterX2 in range(nIter2):
            try:
                status = minimizer.iterate()
            except:
                print "Unexpected error:"
                print "Itr %d : %e %e %e" %(iterX2, low, high, mid )
                raise
            print "Itr %d : %e %e %e" %(iterX2, low, high, mid )
            low    = minimizer.x_lower()
            high   = minimizer.x_upper()
            mid    = minimizer.minimum()
            cggdEstimator.setShapePar( mid )
            cggdEstimator.fixConst()

            if iterX2 % skipSize2 == 0:
                # exit if there is a file whose name is the same as the output file.
                filename = '%s/LS_M=%d-m=%d-r=%d/%04d_%04d/_M-%04d' % (outDataDir, M, m, r, iterX1, iterX2, fbinX)
                if os.path.exists(filename):
                    print '%s already exists. It is overwritten' %(filename)
                if not os.path.exists(os.path.dirname(filename)):
                    os.makedirs(os.path.dirname(filename))
                cggdEstimator.writeParam( filename )
            status = minimize.test_interval(low, high, 0.001, 0)
            if status == errno.GSL_SUCCESS:
                print "# Converged(LS) "
                break
        # get the scaling parameter
        cggdEstimator = MLEstimateScaleParameter( cggdEstimator, fbinX, dataDir, segListFile, M, m, r )
        if iterX1 % skipSize1 == 0:
            # output the inchoate result
            filename = '%s/LS_M=%d-m=%d-r=%d/%04d_FFFF/_M-%04d' % (outDataDir, M, m, r, iterX1, fbinX)
            if os.path.exists(filename):
                print '%s already exists. It is overwritten' %(filename)
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            cggdEstimator.writeParam( filename )

        if cggdEstimator.isConverged() == True:
            print "# Converged"
            break
        
    filename = '%s/LS_M=%d-m=%d-r=%d/FINAL/_M-%04d' % (outDataDir, M, m, r, fbinX)
    if os.path.exists(filename):
        print '%s already exists. It is overwritten' %(filename)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    cggdEstimator.writeParam( filename )


