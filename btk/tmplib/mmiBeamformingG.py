import sys
import string  
import Numeric 
import os.path 
import pickle
import re
from types import FloatType
import getopt, sys

from cluster import *

from sfe.common import *
from sfe.stream import *
from sfe.feature import *
from sfe.matrix import *
from sfe.utils import *

from pygsl import *
from pygsl import multiminimize
from pygsl import sf
import pygsl.errors as errors

from btk import dbase
from btk.modulated import *
from btk.subbandBeamforming import *
from btk.beamformer import *

# a weight for a regularization term
ALPHA         = 1.0E-02
# constants for a gradient algorithm
MAXItns       = 30
STOPTOLERANCE = 1.0E-03
TOLERANCE     = 1.0E-03
STEPSIZE      = 0.01

def clusterPosition( speakerN, azimuthL, elevationL ):
    """@brief do k-mean clustering on the source positions estimated by SL system"""
    """        """
    """@return centroids[0,...,speakerN-1][0,1] """
    estSrcN = len(azimuthL)
    orgPos = []

    for srcX in range( estSrcN ):
        orgPos.append( (azimuthL[srcX],elevationL[srcX]))

    cl = KMeansClustering( orgPos )
    clusters = cl.getclusters( speakerN )
    
    centroids = Numeric.zeros( (speakerN,2), Numeric.Float )
    for spkX in range(speakerN):
        for elemX in range(len(clusters[spkX])):
            centroids[spkX][0] += clusters[spkX][elemX][0]
            centroids[spkX][1] += clusters[spkX][elemX][1]
        centroids[spkX][0] /= len(clusters[spkX])
        centroids[spkX][1] /= len(clusters[spkX])

    return centroids

def calcSigma2( wq, B, wa, SigmaX ):
    """@brief calculate the correlation coefficients of two output signals (See Eq. (3.2))"""
    """@param wq[nChan]           : a quiescent vector"""
    """@param B[nChan][nChan-NC]  : a blocking matrix"""
    """@param wa[nChan-NC]        : an active weight vector"""
    """@param SigmaX[nChan][nChan]: a covariance of input signals"""

    R  = wq - Numeric.matrixmultiply( B, wa )
    L  = Numeric.conjugate( Numeric.transpose(R) )

    tmp = Numeric.matrixmultiply( L, SigmaX )
    sigma2 = Numeric.matrixmultiply( tmp, R )

    return sigma2.real

def calcEpsilon(  wq1, wq2, B1, B2, wa1, wa2, SigmaX ):
    """@brief calculate the cross correlation coefficients of two output signals (See Eq. (3.4))"""
    
    Ltmp = wq1 - Numeric.matrixmultiply( B1, wa1 )
    L    = Numeric.conjugate( Numeric.transpose(Ltmp) )
    R    = wq2 - Numeric.matrixmultiply( B2, wa2 )

    tmp = Numeric.matrixmultiply( L, SigmaX )
    epsilon = Numeric.matrixmultiply( tmp, R )

    return epsilon

def calcMI( wq1, wq2, B1, B2, wa1, wa2, SigmaX ):
    sigma2_1 = calcSigma2( wq1, B1, wa1, SigmaX )
    sigma2_2 = calcSigma2( wq2, B2, wa2, SigmaX )
    epsilon12 = calcEpsilon(  wq1, wq2, B1, B2, wa1, wa2, SigmaX )

    rho2 =  ( epsilon12 * Numeric.conjugate(epsilon12) ) / ( sigma2_1 * sigma2_2 )
    mi = (-1.0/2.0) * Numeric.log( 1.0 - rho2 )
    #    print 'MI ',mi.real
    return mi.real

def calcDeltaRho2( wq1, wq2, B1, B2, wa1, wa2, SigmaX, sigma2_1, sigma2_2, epsilon12 ):

    B1_H = Numeric.conjugate( Numeric.transpose(B1) )
    B1_HxSigmaX = Numeric.matrixmultiply( B1_H, SigmaX )
    
    pwrEps =  epsilon12.real * epsilon12.real + epsilon12.imag * epsilon12.imag
    LB1 = ( pwrEps * sigma2_2 ) * ( wq1 - Numeric.matrixmultiply( B1, wa1 ) )
    LB2 = epsilon12 * sigma2_1 * sigma2_2 * ( wq2 - Numeric.matrixmultiply( B2, wa2 ) )

    deltaRho2 = Numeric.matrixmultiply( B1_HxSigmaX, ( LB1 - LB2 ) ) / ( sigma2_1 * sigma2_1 * sigma2_2 * sigma2_2 )
    
    return deltaRho2


XTIMES = 1.0
def funG(x, (wq1, B1, wq2, B2, SigmaX, alpha, fbinX, NC )):

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
    mmi = calcMI( wq1, wq2, B1, B2, wa1, wa2, SigmaX )
    mmi = mmi* XTIMES
    
    # a regularization term
    rterm  =  alpha * innerproduct(wa1, conjugate(wa1)) + alpha * innerproduct(wa2, conjugate(wa2))
    mmi += rterm.real

    if 0 : #fbinX == 100:
        print 'rterm = %g : alpha = %g' %(rterm.real, alpha)

    del wa1
    del wa2
    return mmi


def dfunG(x, (wq1, B1, wq2, B2, SigmaX, alpha, fbinX, NC )):

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

    sigma2_1 = calcSigma2( wq1, B1, wa1, SigmaX )
    sigma2_2 = calcSigma2( wq2, B2, wa2, SigmaX )
    epsilon12 = calcEpsilon( wq1, wq2, B1, B2, wa1, wa2, SigmaX )
    deltaRho2_1 = calcDeltaRho2( wq1, wq2, B1, B2, wa1, wa2, SigmaX, sigma2_1, sigma2_2, Numeric.conjugate(epsilon12) )
    deltaRho2_2 = calcDeltaRho2( wq2, wq1, B2, B1, wa2, wa1, SigmaX, sigma2_2, sigma2_1, epsilon12 )
    rho2 =  ( epsilon12 * Numeric.conjugate(epsilon12) ) / ( sigma2_1 * sigma2_2 )
    coefA = ( 1.0 ) / ( 2.0 * ( 1.0 - rho2 ) )
    deltaWa1 = coefA * deltaRho2_1 * XTIMES
    deltaWa2 = coefA * deltaRho2_2 * XTIMES

    # a regularization term
    deltaWa1 += alpha * wa1
    deltaWa2 += alpha * wa2

    if 0 : #fbinX == 100:
        print 'sigma2_1  = %g' %sigma2_1
        print 'sigma2_2  = %g' %sigma2_2
        print 'epsilon12 = %g' %abs(epsilon12)
        print 'rho2      = %g' %abs(rho2)
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
    del sigma2_1
    del sigma2_2
    del epsilon12
    del deltaRho2_1
    del deltaRho2_2
    del rho2
    del coefA
    del wa1
    del wa2
    del deltaWa1
    del deltaWa2
    
    return grad

def fdfunG(x, (wq1, B1, wq2, B2, SigmaX, alpha, fbinX, NC )):
    f  = funG(x, (wq1, B1, wq2, B2, SigmaX, alpha, fbinX, NC ))
    df = dfunG(x, (wq1, B1, wq2, B2, SigmaX, alpha, fbinX, NC ))

    #print f,df
    return f, df

def funInstantG(x, (wq1, B1, wq2, B2, SigmaXL, alpha, fbinX, NC )):
    """@brief calculate mutual information based on Gauss assumption. """
    """       This function calculates the average of the mutual      """
    """       information over all frames. """
    """@note  The covariance matrix must be calculated at each frame or each block"""
    
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

    # Calculate the average of the mutual information
    mmi = 0.0
    frameN = len( SigmaXL )
    for sX in range(1,frameN):
        mi_t = calcMI( wq1, wq2, B1, B2, wa1, wa2, SigmaXL[sX] )
        mmi += ( mi_t * XTIMES )
    
    mmi = mmi / ( frameN - 1.0 )
    
    # a regularization term
    rterm  =  alpha * innerproduct(wa1, conjugate(wa1)) + alpha * innerproduct(wa2, conjugate(wa2))
    mmi += rterm.real

    if 0 : #fbinX == 100:
        print 'rterm = %g : alpha = %g' %(rterm.real, alpha)

    del wa1
    del wa2
    return mmi


def dfunInstantG(x, (wq1, B1, wq2, B2, SigmaXL, alpha, fbinX, NC )):

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

    frameN = len( SigmaXL )
    for sX in range(1,frameN):
        sigma2_1 = calcSigma2( wq1, B1, wa1, SigmaXL[sX] )
        sigma2_2 = calcSigma2( wq2, B2, wa2, SigmaXL[sX] )
        epsilon12 = calcEpsilon( wq1, wq2, B1, B2, wa1, wa2, SigmaXL[sX] )
        deltaRho2_1 = calcDeltaRho2( wq1, wq2, B1, B2, wa1, wa2, SigmaXL[sX], sigma2_1, sigma2_2, Numeric.conjugate(epsilon12) )
        deltaRho2_2 = calcDeltaRho2( wq2, wq1, B2, B1, wa2, wa1, SigmaXL[sX], sigma2_2, sigma2_1, epsilon12 )
        rho2 =  ( epsilon12 * Numeric.conjugate(epsilon12) ) / ( sigma2_1 * sigma2_2 )
        coefA = ( 1.0 ) / ( 2.0 * ( 1.0 - rho2 ) )
        deltaWa1 += ( coefA * deltaRho2_1 * XTIMES )
        deltaWa2 += ( coefA * deltaRho2_2 * XTIMES )
        del sigma2_1
        del sigma2_2
        del epsilon12
        del deltaRho2_1
        del deltaRho2_2
        del rho2
        del coefA
        

    deltaWa1 = deltaWa1 / ( frameN - 1.0 )
    deltaWa2 = deltaWa2 / ( frameN - 1.0 )
    
    # a regularization term
    deltaWa1 += alpha * wa1
    deltaWa2 += alpha * wa2

    # Pack the gradient
    grad = Numeric.zeros(4 * (chanN - NC), Numeric.Float)
    for chanX in range(chanN - NC):
        grad[2*chanX]			    = deltaWa1[chanX].real
        grad[2*chanX + 1]		    = deltaWa1[chanX].imag

    for chanX in range(chanN - NC):
        grad[2*chanX + 2 * (chanN - NC)]     = deltaWa2[chanX].real
        grad[2*chanX + 1 + 2 * (chanN - NC)] = deltaWa2[chanX].imag

    #print grad
    del wa1
    del wa2
    del deltaWa1
    del deltaWa2
    return grad

def fdfunInstantG(x, (wq1, B1, wq2, B2, SigmaXL, alpha, fbinX, NC )):
    f  = funInstantG(x, (wq1, B1, wq2, B2, SigmaXL, alpha, fbinX, NC ))
    df = dfunInstantG(x, (wq1, B1, wq2, B2, SigmaXL, alpha, fbinX, NC ))

    #print f,df
    return f, df

def mmiBeamformingWeights_f( fbinX, SigmaX, arrgeom, delayL, startpoint, sampleRate, fftLen, halfBandShift, KindOfSigma = -1 ):
    """@brief find the point which gives the minimum mutual information between two sources"""
    """@param """
    """@param """
    """@param """
    """@param """
    
    fftLen2 = fftLen / 2

    NC = 1
    chanN  = len(arrgeom)
    ndim   = 4 * ( chanN - NC )
    
    wq1        = calcArrayManifold_f( fbinX, fftLen, chanN, sampleRate, delayL[0], halfBandShift)
    B1         = calcBlockingMatrix(wq1)

    wq2        = calcArrayManifold_f( fbinX, fftLen, chanN, sampleRate, delayL[1], halfBandShift)
    B2         = calcBlockingMatrix(wq2)

    # initialize gsl functions
    if KindOfSigma==1 or KindOfSigma==2 :
        sys    = multiminimize.gsl_multimin_function_fdf( funInstantG, dfunInstantG, fdfunInstantG, [wq1, B1, wq2, B2, SigmaX, ALPHA, fbinX, NC], ndim )
    else:
        sys    = multiminimize.gsl_multimin_function_fdf( funG, dfunG, fdfunG, [wq1, B1, wq2, B2, SigmaX, ALPHA, fbinX, NC], ndim )
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
            print 'MI(Gaussian) %d %d %f' %(fbinX, itera, mi)
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

def mmiBeamformingWeights2_f( fbinX, SigmaX, arrgeom, delayL, startpoint, sampleRate, fftLen, halfBandShift, KindOfSigma = -1 ):
    """@brief find the point which gives the minimum mutual information between two sources"""
    """       This function add a constraint which suppress a jammer signal to the upper branch"""
    """@param """
    """@param """
    """@param """
    """@param """
    
    fftLen2 = fftLen / 2

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
    if KindOfSigma==1 or KindOfSigma==2 :
        sys    = multiminimize.gsl_multimin_function_fdf( funInstantG, dfunInstantG, fdfunInstantG, [wq1, B1, wq2, B2, SigmaX, ALPHA, fbinX, NC], ndim )
    else :
        sys    = multiminimize.gsl_multimin_function_fdf( funG, dfunG, fdfunG, [wq1, B1, wq2, B2, SigmaX, ALPHA, fbinX, NC], ndim )
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
            #print '||w_a|| ', abs(waAs)
        if status2 == 0:
            #print 'MI Converged %d %d %f' %(fbinX, itera,mi)
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

def packWeights( chanN, NC, unpackWa1, unpackWa2 ):
    """@brief Pack the (adaptive) weights """
    packWa = Numeric.zeros(4 * (chanN - NC), Numeric.Float)
    for chanX in range(chanN - NC):
        packWa[2*chanX]			    = unpackWa1[chanX].real
        packWa[2*chanX + 1]		    = unpackWa1[chanX].imag

    for chanX in range(chanN - NC):
        packWa[2*chanX + 2 * (chanN - NC)]     = unpackWa2[chanX].real
        packWa[2*chanX + 1 + 2 * (chanN - NC)] = unpackWa2[chanX].imag

    return packWa

def UnpackWeights( chanN, NC, waAs ):
    """@brief Unpack current weights """
    
    wa1A = Numeric.zeros(chanN-NC, Numeric.Complex)
    wa2A = Numeric.zeros(chanN-NC, Numeric.Complex)
    weights = []
    for chanX in range(chanN-NC):
        wa1A[chanX] = waAs[2 * chanX] + 1j * waAs[2 * chanX + 1]
    for chanX in range(chanN-NC):
        wa2A[chanX] = waAs[2 * chanX + 2 * (chanN - NC)] + 1j * waAs[2 * chanX + 1 + 2 * (chanN - NC)]

    weights.append(wa1A)
    weights.append(wa2A)
    
    return weights
      
def SeparateUnpacedWeights( fftLen, chanN, NC, waAs ):
    """@brief Unpack current weights """
    
    wa1A = Numeric.zeros(2 * (chanN - NC), Numeric.Float) 
    wa2A = Numeric.zeros(2 * (chanN - NC), Numeric.Float) 
    for chanX in range(chanN-NC):
        wa1A[2*chanX]   = waAs[2 * chanX]
        wa1A[2*chanX+1] = waAs[2 * chanX + 1]
        
    for chanX in range(chanN-NC):
        wa2A[2*chanX]   = waAs[2 * chanX + 2 * (chanN - NC)]
        wa2A[2*chanX+1] = waAs[2 * chanX + 1 + 2 * (chanN - NC)]

    weights = []
    weights.append(wa1A)
    weights.append(wa2A)
    
    return weights

def calcCovarOverAll( frameN, chanN, analysisFBs ):
    """@brief calculate covariance matricies over all frequency bins"""
    """@param frameN: the number of samples"""
    """@param chanN  : the number of channels(microphones)"""
    """@param analysisFBs """
    """@return SigmaXL[fftLen][chanN][chanN] list of covariance matricies"""

    print 'calculate a covariance matrix with %d samples' %(frameN)
    fftLen = analysisFBs[0].fftLen()
    SigmaXL = []
    for fbinX in range(fftLen):
        SigmaX = Numeric.zeros( (chanN,chanN), Numeric.Complex )
        SigmaXL.append(SigmaX)

    # zero mean at this time... , mean = Numeric.zeros(chanN).astype(Numeric.Complex) 
    snapShotArray = SnapShotArrayPtr(fftLen, chanN )
    #print 'size ',analysisFBs[0].size(),snapShotArray.fftLen()

    for sX in range(frameN):
        ichan = 0
        for analFB in analysisFBs:
            sbSample = Numeric.array(analFB.next())
            snapShotArray.newSample(sbSample, ichan)
            ichan += 1

        snapShotArray.update()
        for fbinX in range(fftLen):
            snapshot = snapShotArray.getSnapShot(fbinX)
            # zero mean
            SigmaXL[fbinX] += Numeric.outerproduct(snapshot, conjugate(snapshot))
            del snapshot
            
    for fbinX in range(fftLen):
        SigmaXL[fbinX] /= frameN

    del snapShotArray
    return SigmaXL

def calcInstantCovar( frameN, chanN, analysisFBs, FGTF=0.1 ):
    """@brief calculate covariance matricies over all frequency bins at each frame"""
    """@param frameN: the number of samples"""
    """@param chanN  : the number of channels(microphones)"""
    """@param analysisFBs """
    """@return SigmaXL[fftLen][chanN][chanN] list of covariance matricies"""

    print 'calculate a covariance matrix at each frame with a forgetting factor %f...' %FGTF
    print '%d samples are used' %(frameN)
    fftLen = analysisFBs[0].fftLen()
    SigmaXLL = []
    
    for fbinX in range(fftLen):
        SigmaXL = Numeric.zeros( (frameN,chanN,chanN), Numeric.Complex )
        SigmaXLL.append( SigmaXL )
        
    # zero mean at this time... , mean = Numeric.zeros(chanN).astype(Numeric.Complex) 
    snapShotArray = SnapShotArrayPtr(fftLen, chanN )
    #print 'size ',analysisFBs[0].size(),snapShotArray.fftLen()

    for sX in range(frameN):
        ichan = 0
        for analFB in analysisFBs:
            sbSample = Numeric.array(analFB.next())
            snapShotArray.newSample(sbSample, ichan)
            ichan += 1

        snapShotArray.update()
        for fbinX in range(fftLen):
            snapshot = snapShotArray.getSnapShot(fbinX)
            # zero mean
            SigmaXLL[fbinX][sX] = Numeric.outerproduct(snapshot, conjugate(snapshot))
            # averaging the covariance matrices...
            if sX > 0 :
                SigmaXLL[fbinX][sX] = FGTF * SigmaXLL[fbinX][sX-1] + ( 1.0 - FGTF ) * SigmaXLL[fbinX][sX]
            elif sX==1 :
                SigmaXLL[fbinX][0]  = SigmaXLL[fbinX][1]
            del snapshot
    
    del snapShotArray

    return SigmaXLL

def getAllSnapShots( sFrame, eFrame, chanN, analysisFBs ):
    """@brief get spectral samples """
    """@param sFrame: the start frame"""
    """@param eFrame: the end frame"""
    """@param chanN  : the number of channels(microphones)"""
    """@param analysisFBs : an object which has spectral samples"""
    """@return XL[frame][fftLen][chanN] : input subband snapshots"""

    frameN = eFrame - sFrame
    print '%d samples are used' %(frameN)
    fftLen = analysisFBs[0].fftLen()
    XL = []
    
    # zero mean at this time... , mean = Numeric.zeros(chanN).astype(Numeric.Complex)
    snapShotArray = SnapShotArrayPtr(fftLen, chanN )
    #print 'size ',analysisFBs[0].size(),snapShotArray.fftLen()

    for sX in range(sFrame,eFrame):
        ichan = 0
        for analFB in analysisFBs:
            sbSample = Numeric.array(analFB.next())
            snapShotArray.newSample( sbSample, ichan )
            ichan += 1

        snapShotArray.update()
        X_t = [] # X_t[fftLen][chanN]
        for fbinX in range(fftLen):
            # X_t_f = copy.deepcopy( snapShotArray.getSnapShot(fbinX) )
            #X_t.append( copy.deepcopy( snapShotArray.getSnapShot(fbinX) ) )
            X_t.append( copy.copy( snapShotArray.getSnapShot(fbinX) ) )

        XL.append( X_t )

    del snapShotArray
    return XL

def calcBlockCovar( frameN, chanN, analysisFBs, shift, blockSize ):
    """@brief calculate covariance matricies with samples in blocks"""
    """@note it returns the sequence of instantaneous covariance matrices"""
    """@param frameN: the number of samples"""
    """@param chanN  : the number of channels(microphones)"""
    """@param analysisFBs """
    """@return SigmaXLL[fftLen][frameN][chanN][chanN] list of covariance matricies"""

    fftLen = analysisFBs[0].fftLen()
    SigmaXLL = []
    nBlock   = ( frameN - blockSize ) / shift
    print '%d samples are used' %(frameN)
    print '%d blocks are used' %(nBlock)
    print 'frame shift %d, block size %d' %(shift,blockSize)
    
    for fbinX in range(fftLen):
        SigmaXL = Numeric.zeros( (nBlock,chanN,chanN), Numeric.Complex )
        SigmaXLL.append( SigmaXL )
        
    # zero mean at this time... , mean = Numeric.zeros(chanN).astype(Numeric.Complex)
    samples = getAllSnapShots( 0, frameN, chanN, analysisFBs )
    blkX = 0
    for start in range(0,frameN,shift):
        end = start + blockSize

        for idx in range(start,end):
            for fbinX in range(fftLen):
                # zero mean
                SigmaXLL[fbinX][blkX] += Numeric.outerproduct( samples[idx][fbinX], conjugate(samples[idx][fbinX]) )

        #print "block %d : %d - %d" %(blkX,start,end)
        SigmaXLL[fbinX][blkX] /= blockSize
        blkX += 1
        if blkX == nBlock :
            break

    del samples
    return SigmaXLL

def calcCovar( frameN, chanN, analysisFBs, KindOfSigma = -1, FGTF = 0.1, shift = 6, blockSize = 12 ):
    """@brief a wrapper function to calculate covariance matrices"""
    """@param frameN: the number of samples"""
    """@param chanN  : the number of channels(microphones)"""
    """@param analysisFBs """
    
    if KindOfSigma == 1 :
        return calcInstantCovar( frameN, chanN, analysisFBs, FGTF )
    elif KindOfSigma == 2 :
        return calcBlockCovar( frameN, chanN, analysisFBs, shift, blockSize )
        
    return calcCovarOverAll( frameN, chanN, analysisFBs )
    

def frameX2blockX( frameX, shift, blockSize ):
    """@brief  find the index of a block which contains the frame."""
    """@param  frameX """
    """@param  shift """
    """@param  blockSize """
    
    frameX = int(frameX)
    shift  = int(shift)
    blockSize = int(blockSize)
    if frameX < shift :
        return 0
    
    sb = blockSize / 4
    eb = blockSize * 3 / 4
    if eb <= sb:
        eb = sb + 1

    blockXL = []
    for idx in range(sb,eb):
        if ( frameX - idx ) % shift == 0 :
            blockXL.append( ( ( frameX - idx ) / shift ) )

    maxV = max( blockXL )
    del blockXL
    return maxV
