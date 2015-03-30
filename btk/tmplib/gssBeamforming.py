import sys
import string  
import Numeric 
import os.path 
import pickle
import re
from types import FloatType
import getopt, sys

from sfe.common import *
from sfe.stream import *
from sfe.feature import *
from sfe.matrix import *
from sfe.utils import *

from pygsl import *
from pygsl import multiminimize

from btk import dbase
from btk.modulated import *
from btk.subbandBeamforming import *
from btk.beamformer import *
import mmiBeamformingG

# a weight for a regularization term
ALPHA         = 1.0E-02
# constants for a gradient algorithm
MAXItns       = 15
STOPTOLERANCE = 1.0E-03
TOLERANCE     = 1.0E-03
STEPSIZE      = 0.01

def calcEpsilon(  wq1, wq2, B1, B2, wa1, wa2, SigmaX ):
    """@brief calculate the cross correlation coefficients of two output signals (See Eq. (3.4))"""
    
    Ltmp = wq1 - Numeric.matrixmultiply( B1, wa1 )
    L    = Numeric.conjugate( Numeric.transpose(Ltmp) )
    R    = wq2 - Numeric.matrixmultiply( B2, wa2 )

    tmp = Numeric.matrixmultiply( L, SigmaX )
    epsilon = Numeric.matrixmultiply( tmp, R )

    return epsilon

def calcPLMeasure( wq1, wq2, B1, B2, wa1, wa2, SigmaX ):
    """@brief calculate the measure proposed by Parra and Lucas"""
    
    epsilon12 = calcEpsilon( wq1, wq2, B1, B2, wa1, wa2, SigmaX )
    epsilon12_2 = ( epsilon12 * Numeric.conjugate(epsilon12) )

#    print 'PLC ',epsilon12_2
    return epsilon12_2.real

def calcDeltaOfPLMeasure( wq2, B1, B2, wa2, SigmaX, epsilon12 ):

    B1_H = Numeric.conjugate( Numeric.transpose(B1) )
    B1_HxSigmaX = Numeric.matrixmultiply( B1_H, SigmaX )
    RB1  = wq2 - Numeric.matrixmultiply( B2, wa2 )
    delta = -1.0 * ( Numeric.matrixmultiply( B1_HxSigmaX, RB1) ) * epsilon12
    
    return delta


def funGSS(x, (wq1, B1, wq2, B2, SigmaX, alpha, fbinX, NC )):

    chanN = len(wq1)
    if chanN != len(wq2):
        print 'ERROR: The sizes of quiescent weight vectors must be the same'

    # Unpack current weights
    wa1   = Numeric.zeros(chanN-NC, Numeric.Complex)
    wa2   = Numeric.zeros(chanN-NC, Numeric.Complex)
    for chanX in range(chanN-NC):
        wa1[chanX] = x[2 * chanX] + 1j * x[2 * chanX + 1]

    for chanX in range(chanN-NC):
        wa2[chanX] = x[2 * chanX + 2 * (chanN - NC)] + 1j * x[2 * chanX + 1 + 2 * (chanN - NC)]

    # Calculate mutual information
    mmi = calcPLMeasure( wq1, wq2, B1, B2, wa1, wa2, SigmaX )
     
    # a regularization term
    rterm  =  alpha * innerproduct(wa1, conjugate(wa1)) + alpha * innerproduct(wa2, conjugate(wa2))
    mmi += rterm.real

    if fbinX == 100:
        print 'rterm = %g : alpha = %g' %(rterm.real, alpha)

    return mmi


def dfunGSS(x, (wq1, B1, wq2, B2, SigmaX, alpha, fbinX, NC )):

    chanN = len(wq1)
    if chanN != len(wq2):
        print 'ERROR: The sizes of quiescent weight vectors must be the same'

    # Unpack current weights
    wa1   = Numeric.zeros(chanN-NC, Numeric.Complex)
    wa2   = Numeric.zeros(chanN-NC, Numeric.Complex)
    for chanX in range(chanN-NC):
        wa1[chanX] = x[2 * chanX] + 1j * x[2 * chanX + 1]

    for chanX in range(chanN-NC):
        wa2[chanX] = x[2 * chanX + 2 * (chanN - NC)] + 1j * x[2 * chanX + 1 + 2 * (chanN - NC)]

    # Calculate complex gradient
    deltaWa1 = Numeric.zeros(chanN - NC, Numeric.Complex)
    deltaWa2 = Numeric.zeros(chanN - NC, Numeric.Complex)

    epsilon12 = calcEpsilon( wq1, wq2, B1, B2, wa1, wa2, SigmaX )
    deltaWa1 = calcDeltaOfPLMeasure( wq2, B1, B2, wa2, SigmaX, Numeric.conjugate(epsilon12) )
    deltaWa2 = calcDeltaOfPLMeasure( wq1, B2, B1, wa1, SigmaX, epsilon12 )
    
    if fbinX == 100:
        print 'epsilon12 = %g' %abs(epsilon12)

    # a regularization term
    deltaWa1 += alpha * wa1
    deltaWa2 += alpha * wa2

    if fbinX == 100:
        print 'wa1:', wa1
        print 'wa2:', wa2

    # Pack the gradient
    grad = Numeric.zeros(4 * (chanN - NC), Numeric.Float)
    for chanX in range(chanN - NC):
        grad[2*chanX]			    = deltaWa1[chanX].real
        grad[2*chanX + 1]		    = deltaWa1[chanX].imag

    for chanX in range(chanN - NC):
        grad[2*chanX + 2 * (chanN - NC)]     = deltaWa2[chanX].real
        grad[2*chanX + 1 + 2 * (chanN - NC)] = deltaWa2[chanX].imag

    #print grad
    return grad

def fdfunGSS(x, (wq1, B1, wq2, B2, SigmaX, alpha, fbinX, NC)):
    f  = funGSS(x, (wq1, B1, wq2, B2, SigmaX, alpha, fbinX, NC))
    df = dfunGSS(x, (wq1, B1, wq2, B2, SigmaX, alpha, fbinX, NC))

    #print f,df
    return f, df

def gssBeamformingWeights_f( fbinX, SigmaX, arrgeom, delayL, startpoint, sampleRate, fftLen, halfBandShift ):
    """@brief find the point which gives the minimum mutual information between two sources"""
    """@param """
    """@param """
    """@param """
    """@param """

    NC = 1
    chanN  = len(arrgeom)
    ndim   = 4 * ( chanN - NC )

    wq1        = calcArrayManifold_f( fbinX, fftLen, chanN, sampleRate, delayL[0], halfBandShift)
    B1         = calcBlockingMatrix(wq1)

    wq2        = calcArrayManifold_f( fbinX, fftLen, chanN, sampleRate, delayL[1], halfBandShift)
    B2         = calcBlockingMatrix(wq2)
    
    # initialize gsl functions
    sys    = multiminimize.gsl_multimin_function_fdf( funGSS, dfunGSS, fdfunGSS, [wq1, B1, wq2, B2, SigmaX, ALPHA, fbinX], ndim, NC )
    solver = multiminimize.conjugate_pr_eff( sys, ndim )
    solver.set(startpoint, STEPSIZE, STOPTOLERANCE )
    waAs = startpoint
    #print "Using solver ", solver.name() 

    #curval = 1.0E+30
    for itera in range(MAXItns):
        try: 
            status1 = solver.iterate()
        except errors.gsl_NoProgressError, msg:
            print 'gsl_NoProgressError'
            print msg
            break
        except:
            print "Unexpected error:", sys.exc_info()[0]
            raise
        gradient = solver.gradient()
        waAs = solver.getx()
        mi   = solver.getf()
        status2 = multiminimize.test_gradient( gradient, STOPTOLERANCE )

        del gradient
        if fbinX == 100 :
            print 'MI(GSS) %d %d %f' %(fbinX, itera, mi)
            #print '||w_a|| ', abs(waAs)
        if status2 == 0:
            #print 'MI Converged %d %d %f' %(fbinX, itera,mi)
            break

    # Unpack current weights
    # return UnpackWeights( chanN, NC, waAs )
    del wq1
    del wq2
    del B1
    del B2
    del solver
    del sys
    
    return waAs

def gssBeamformingWeights2_f( fbinX, SigmaX, arrgeom, delayL, startpoint, sampleRate, fftLen, halfBandShift ):
    """@brief find the point which gives the minimum mutual information between two sources"""
    """       This function add a constraint which suppress a jammer signal to the upper branch"""
    """@param """
    """@param """
    """@param """
    """@param """
    
    NC = 2
    chanN  = len(arrgeom)
    ndim   = 4 * ( chanN - NC )
    
    wds1       = calcArrayManifoldWoNorm_f( fbinX, fftLen, chanN, sampleRate, delayL[0], halfBandShift)
    wds2       = calcArrayManifoldWoNorm_f( fbinX, fftLen, chanN, sampleRate, delayL[1], halfBandShift)

    wq1 = calcNullBeamformer( wds1, wds2, NC )
    wq2 = calcNullBeamformer( wds2, wds1, NC )
    B1  = calcBlockingMatrix( wq1, NC )
    B2  = calcBlockingMatrix( wq2, NC )
    
    # initialize gsl functions
    sys    = multiminimize.gsl_multimin_function_fdf( funGSS, dfunGSS, fdfunGSS, [wq1, B1, wq2, B2, SigmaX, ALPHA, fbinX, NC], ndim )
    solver = multiminimize.conjugate_pr_eff( sys, ndim )
    solver.set(startpoint, STEPSIZE, TOLERANCE )
    waAs = startpoint
    #print "Using solver ", solver.name() 
    #curval = 1.0E+30
    for itera in range(MAXItns):
        try: 
            status1 = solver.iterate()
        except errors.gsl_NoProgressError, msg:
            print 'gsl_NoProgressError'
            print msg
            break
        except:
            print "Unexpected error:", sys.exc_info()[0]
            raise
        gradient = solver.gradient()
        waAs = solver.getx()
        mi   = solver.getf()
        status2 = multiminimize.test_gradient( gradient, STOPTOLERANCE )

        del gradient
        if fbinX == 100 :
            print 'MI(Gaussian2) %d %d %f' %(fbinX, itera, mi)
            print '||w_a|| ', abs(waAs)
        if status2 == 0:
            print 'MI Converged %d %d %f' %(fbinX, itera,mi)
            break

    # Unpack current weights
    # return UnpackWeights( chanN, NC, waAs )
    del wds1
    del wds2
    del wq1
    del wq2
    del B1
    del B2
    del solver
    del sys
    
    return waAs
