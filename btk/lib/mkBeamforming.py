import sys
import string
import Numeric
import os.path
import pickle
import re
from types import FloatType
import getopt, sys
import copy

from sfe.common import *
from sfe.stream import *
from sfe.feature import *
from sfe.matrix import *
from sfe.utils import *

#from pygsl import *
from pygsl import multiminimize
from pygsl import sf
import pygsl.errors as errors

from btk import dbase
from btk.modulated import *
from btk.subbandBeamforming import *
from btk.beamformer import *

APPZERO = 1.0E-20

class MKSubbandBeamformer:
    def __init__(self, spectralSources, nSource, NC, alpha, halfBandShift ):

        if NC > 2:
            print 'not yet implemented in the case of NC > 2'
            sys.exit()
        if nSource > 2:
            print "not support more than 2 sources"
            sys.exit(1)
        if halfBandShift==True:
            print "not support halfBandShift==True yet"
            sys.exit(1)
            
        self._halfBandShift = halfBandShift
        self._NC = NC
        
        # ouptputs of analysis filter banks
        self._spectralSources = spectralSources
        # the number of channels
        self._nChan           = len(spectralSources)
        # fft length = the number of subbands
        self._fftLen          = spectralSources[0].fftLen()
        # regularization term
        self._alpha = alpha        
        # the number of sound sources
        self._nSource = nSource        
        # input vectors [frameN][chanN]
        self._observations    = []
        # covariance matrix of input vectors [fftLen/2+1][chanN][chanN]
        self._SigmaX = []
        # quiescent vectors : _wq[nSource][fftLen2+1]
        self._wq = []
        # blocking matricies : _B[nSource][fftLen2+1]
        self._B = []
        # the entire GSC 's weight, wq - B * wa : _wo[nSource][fftLen2+1]        
        self._wo = []
        for srcX in range(self._nSource):
            self._wo.append( Numeric.zeros( (self._fftLen/2+1,self._nChan), Numeric.Complex) )
        
    def accumObservations(self, sFrame, eFrame, R=1 ):
        """@brief accumulate observed subband components for adaptation """
        """@param sFrame: the start frame"""
        """@param eFrame: the end frame"""
        """@param R : R = 2**r, where r is a decimation factor"""
        """@return self._observations[frame][fftLen][chanN] : input subband snapshots"""

        fftLen = self._fftLen
        chanN  = self._nChan
        if R < 1:
            R = 1

        self._observations = []
        
        # zero mean at this time... , mean = Numeric.zeros(chanN).astype(Numeric.Complex)
        snapShotArray = SnapShotArrayPtr( fftLen, chanN )
        print 'from %d to %d, fftLen %d' %( sFrame, eFrame, snapShotArray.fftLen() )

        #for sX in range(sFrame,eFrame):
        for sX in range(eFrame):
            ichan = 0
            for analFB in self._spectralSources:
                sbSample = Numeric.array(analFB.next())
                snapShotArray.newSample( sbSample, ichan )
                ichan += 1

            snapShotArray.update()
            if sX >= sFrame and sX < eFrame :
                X_t = [] # X_t[fftLen][chanN]
                if sX % R == 0:
                    for fbinX in range(fftLen):
                        X_t.append( copy.deepcopy( snapShotArray.getSnapShot(fbinX) ) )
                    self._observations.append( X_t )

        for analFB in self._spectralSources:
            analFB.reset()
            
        del snapShotArray
        return self._observations

    def calcCov(self):
        """@brief calculate covariance matricies of inputs over all frequency bins"""
        """@return the covariance matricies of input vectors : SigmaX[fftLen][chanN][chanN]"""

        if len(self._observations) == 0:
            print "Zero observation! Call getObservations() first!"
            sys.exit()

        samples = self._observations
        frameN  = len( samples )
        fftLen  = self._fftLen
        fftLen2 = fftLen/2
        chanN   = self._nChan
        
        SigmaX = []
        for fbinX in range(fftLen2+1):
            SigmaX.append( Numeric.zeros( (chanN,chanN), Numeric.Complex ) )

        # zero mean at this time... , mean = Numeric.zeros(chanN).astype(Numeric.Complex)
        for sX in range(frameN):
            for fbinX in range(fftLen2+1):
                # zero mean assumption
                SigmaX[fbinX] += Numeric.outerproduct( samples[sX][fbinX], conjugate(samples[sX][fbinX]) )

        for fbinX in range(fftLen2+1):
            SigmaX[fbinX] /= frameN

        self._SigmaX = SigmaX
        
        return self._SigmaX

    def calcGSCOutput_f(self, wo, Xft ):
        """@breif calculate outputs of the GSC at a subband frequency bin"""
        """@param wo[nChan]  : the entire beamformer's weight"""
        """@param Xft[nChan] : the input vector"""
        """@return an output value of a GSC beamformer at a subband frequency bin"""
        """@note this function supports half band shift only"""

        wH  = Numeric.transpose( Numeric.conjugate( wo ) )
        Yt  = Numeric.innerproduct( wH, Xft )

        return Yt

    def getSourceN(self):
        return  self._nSource

    def getChanN(self):
        return  self._nChan

    def getSampleN(self):
        return len( self._observations )

    def getFftLen(self):
        return self._fftLen

    def getWq(self, srcX, fbinX):
        return self._wq[srcX][fbinX]

    def getB(self, srcX, fbinX):
        return self._B[srcX][fbinX]

    def getAlpha(self):
        return self._alpha

    def calcFixedWeights(self, sampleRate, delays ):
        # @brief calculate the quiescent vectors and blocking matricies
        # @param sampleRate : sampling rate (Hz)
        # @param delays[nSource][nChan] : 
        fftLen2 = self._fftLen / 2
        self._wq = []
        self._B  = []

        if self._NC == 1:
            for srcX in range(self._nSource):
                wq_n = []
                B_n  = []      
                for fbinX in range(fftLen2+1):
                    wq_nf = calcArrayManifold_f( fbinX, self._fftLen, self._nChan, sampleRate, delays[0], self._halfBandShift )
                    B_nf  = calcBlockingMatrix(wq_nf)
                    wq_n.append(wq_nf)
                    B_n.append(B_nf)
                self._wq.append(wq_n)
                self._B.append(B_n)
        elif self._nSource==2 and self._NC == 2:
            wq1 = []
            wq2 = []
            B1  = []
            B2  = []                
            for fbinX in range(fftLen2+1):
                wds1 = calcArrayManifoldWoNorm_f( fbinX, self._fftLen, self._nChan, sampleRate, delays[0], self._halfBandShift)
                wds2 = calcArrayManifoldWoNorm_f( fbinX, self._fftLen, self._nChan, sampleRate, delays[1], self._halfBandShift)
                wq1_nf = calcNullBeamformer( wds1, wds2, self._NC )
                wq2_nf = calcNullBeamformer( wds2, wds1, self._NC )
                B1_nf  = calcBlockingMatrix( wq1_nf, self._NC )
                B2_nf  = calcBlockingMatrix( wq2_nf, self._NC )
                wq1.append(wq1_nf)
                wq2.append(wq2_nf)
                B1.append(B1_nf)                
                B2.append(B2_nf)                
            self._wq.append(wq1)
            self._wq.append(wq2)
            self._B.append(B1)
            self._B.append(B2)    
        else:
            print 'not yet implemented in the case of NC > 2'
            sys.exit()

    def UnpackWeights( self, waAs ):
        """@brief Unpack the active weight vector at a frequency bin"""
        nSource = self._nSource
        chanN   = self._nChan
        NC      = self._NC

        weights = []
        idx = 0
        for srcX in range(nSource):
            waA = Numeric.zeros(chanN-NC, Numeric.Complex)
            for chanX in range(chanN-NC):
                waA[chanX] = waAs[2 * chanX + idx ] + 1j * waAs[2 * chanX + 1 + idx]
            weights.append( waA )
            #print '|wa|', Numeric.sqrt( innerproduct(waA, conjugate(waA)) )
            idx += ( 2 * (chanN - NC) )

        return weights

# @memo fun_MK() and dfun_MK() are call back functions for pygsl.
#       You can easily implement a new MK beamformer by writing a new class derived from
#       a class 'MKSubbandBeamformer' which have methods, normalizeWa( wa ),
#       calcKurtosis( srcX, fbinX, wa ) and gradient( srcX, fbinX, wa ).
def fun_MK(x, (fbinX, MKSubbandBeamformerPtr, NC) ):
    # @brief calculate the objective function for the gradient algorithm
    # @param x[2(chanN-NC)] : active weights (packed)
    # @param fbinX: the frequency bin index you process 
    # @param MNSubbandBeamformerPtr: the class object for calculating functions 
    # @param NC: the number of constrants (not yet implemented)

    chanN   = MKSubbandBeamformerPtr.getChanN()
    frameN  = MKSubbandBeamformerPtr.getSampleN()
    fftLen  = MKSubbandBeamformerPtr.getFftLen()
    sourceN = MKSubbandBeamformerPtr.getSourceN()
        
    # Unpack current weights : x[2*nSource*(chanN - NC )] -> wa[nSource][chanN-NC]
    wa = []
    idx = 0
    for srcX in range(sourceN):
        wa.append( Numeric.zeros( chanN-NC, Numeric.Complex) )
        for chanX in range(chanN-NC):
            wa[srcX][chanX] = x[2 * chanX+ idx] + 1j * x[2 * chanX + 1+ idx]
        idx += ( 2 * (chanN - NC) )

    wa    = MKSubbandBeamformerPtr.normalizeWa( fbinX, wa )
    # Calculate the objective function, the negative of the kurtosis
    nkurt = 0.0
    for srcX in range(sourceN):
        nkurt -= MKSubbandBeamformerPtr.calcKurtosis( srcX, fbinX, wa )
    # a regularization term
    rterm = 0.0
    alpha = MKSubbandBeamformerPtr.getAlpha()
    for srcX in range(sourceN):    
        rterm  +=  alpha * innerproduct(wa, conjugate(wa)) 
    nkurt  += rterm.real

    return nkurt

def dfun_MK(x, (fbinX, MKSubbandBeamformerPtr, NC ) ):
    # @brief calculate the derivatives of the objective function for the gradient algorithm
    # @param x[2(chanN-NC)] : active weights (packed)
    # @param fbinX: the frequency bin index you process 
    # @param MKSubbandBeamformerPtr: the class object for calculating functions 
    # @param NC: the number of constrants 
    
    chanN  = MKSubbandBeamformerPtr.getChanN()
    frameN = MKSubbandBeamformerPtr.getSampleN()
    fftLen = MKSubbandBeamformerPtr.getFftLen()
    sourceN = MKSubbandBeamformerPtr.getSourceN()

    # Unpack current weights : x[2*nSource*(chanN - NC )] -> wa[nSource][chanN-NC]
    wa = []
    idx = 0
    for srcX in range(sourceN):
        wa.append( Numeric.zeros( chanN-NC, Numeric.Complex) )
        for chanX in range(chanN-NC):
            wa[srcX][chanX] = x[2 * chanX+ idx] + 1j * x[2 * chanX + 1+ idx]
        idx += ( 2 * (chanN - NC) )

    wa    = MKSubbandBeamformerPtr.normalizeWa( fbinX, wa )    
    # Calculate a gradient
    deltaWa = []
    for srcX in range(sourceN):
        deltaWa_n = - MKSubbandBeamformerPtr.gradient( srcX, fbinX, wa )
        deltaWa.append( deltaWa_n )

    # add the derivative of the regularization term
    alpha = MKSubbandBeamformerPtr.getAlpha()
    for srcX in range(sourceN):
        deltaWa[srcX] += alpha * wa[srcX]
    
    # Pack the gradient
    grad = Numeric.zeros(2 * sourceN * (chanN - NC), Numeric.Float)
    idx = 0
    for srcX in range(sourceN):
        for chanX in range(chanN - NC):
            grad[2*chanX+ idx]     = deltaWa[srcX][chanX].real
            grad[2*chanX + 1+ idx] = deltaWa[srcX][chanX].imag
        idx += ( 2 * (chanN - NC) )

    if fbinX == 10:
        print 'grad', grad

    return grad

def fdfun_MK(x, (fbinX, MKSubbandBeamformerPtr, NC ) ):
    f  = fun_MK(x, (fbinX, MKSubbandBeamformerPtr, NC ) )
    df = dfun_MK(x, (fbinX, MKSubbandBeamformerPtr, NC ) )

    return f, df

# @class maximum empirical kurtosis beamformer 
# usage:
# 1. construct an object, mkBf = MKSubbandBeamformerGGDr( spectralSources  )
# 2. calculate the fixed weights, mkBf.calcFixedWeights( sampleRate, delay )
# 3. accumulate input vectors, mkBf.accumObservations( sFrame, eFrame, R )
# 4. calculate the covariance matricies of the inputs, mkBf.calcCov()
# 5. estimate active weight vectors, mkBf.estimateActiveWeights( fbinX, startpoint )
class MEKSubbandBeamformer_pr(MKSubbandBeamformer):
    def __init__(self, spectralSources, nSource=1, NC=1, alpha = 1.0E-02, halfBandShift=False  ):
        MKSubbandBeamformer.__init__(self, spectralSources, nSource, NC, alpha, halfBandShift )

    def normalizeWa(self, fbinX, wa):
        return wa
    
    def calcEntireWeights_f(self, fbinX, wa_f ):
        """@breif calculate the entire weight vector of the beamformer for each bin"""
        """@param fbinX  : the index of the subband frequency bin"""
        """@param wa_f[nSource][nChan-NC]    """

        for srcX in range(self._nSource):
            self._wo[srcX][fbinX] = self._wq[srcX][fbinX] - Numeric.matrixmultiply( self._B[srcX][fbinX], wa_f[srcX] )
            
        return self._wo

    def calcKurtosis( self, srcX, fbinX, wa_f ):
        # @brief calculate empirical kurtosis :
        #        \frac{1}{T} \sum_{t=0}^{T-1} Y^4 - 3 ( \frac{1}{T} \sum_{t=0}^{T-1} Y^2 )
        # @param srcX: the source index you process
        # @param fbinX  : the index of the subband frequency bin"""
        # @param wa_f[nSource][nChan-NC]
        frameN = len( self._observations )

        exY2 = 0.0
        exY4 = 0.0
        for frX in range(frameN):
            self.calcEntireWeights_f( fbinX, wa_f )
            Y  = self.calcGSCOutput_f( self._wo[srcX][fbinX], self._observations[frX][fbinX] )
            Y2 = Y * Numeric.conjugate( Y )
            Y4 = Y2 * Y2
            exY2 += ( Y2.real / frameN )
            exY4 += ( Y4.real / frameN )

        kurt = exY4 - 3 * exY2 * exY2
        return kurt

    def gradient( self, srcX, fbinX, wa_f ):
        # @brief calculate the derivative of empirical kurtosis w.r.t. wa_H
        # @param srcX: the source index you process
        # @param fbinX  : the index of the subband frequency bin"""
        # @param wa_f[nSource][nChan-NC]
        frameN = len( self._observations )

        dexY2 = Numeric.zeros( ( self._nChan - self._NC ), Numeric.Complex )
        dexY4 = Numeric.zeros( ( self._nChan - self._NC ), Numeric.Complex )
        exY2 = 0.0
        BH = Numeric.transpose( Numeric.conjugate( self._B[srcX][fbinX] ) )
        for frX in range(frameN):
            self.calcEntireWeights_f( fbinX, wa_f )
            Y  = self.calcGSCOutput_f( self._wo[srcX][fbinX], self._observations[frX][fbinX] )
            BHX = - Numeric.matrixmultiply( BH, self._observations[frX][fbinX] ) # BH * X
            Y2 = Y * Numeric.conjugate( Y )
            dexY4 += ( 2 * Y2 * BHX * Numeric.conjugate( Y ) / frameN )
            dexY2 += ( BHX * Numeric.conjugate( Y ) / frameN )
            exY2  += ( Y2.real / frameN )

        deltaKurt = dexY4 - 6 * exY2 * dexY2

        return deltaKurt

    def estimateActiveWeights( self, fbinX, startpoint, MAXITNS=40, TOLERANCE=1.0E-03, STOPTOLERANCE = 1.0E-02, DIFFSTOPTOLERANCE= 1.0E-05, STEPSIZE=0.01 ):
        # @brief estimate active weight vectors at a frequency bin
        # @param fbinX: the frequency bin index you process
        # @param startpoint: the initial active weight vector
        # @param NC: the number of constrants (not yet implemented)
        # @param MAXITNS: the maximum interation for the gradient algorithm
        # @param TOLERANCE : tolerance for the linear search
        # @param STOPTOLERANCE : tolerance for the gradient algorithm
        
        if fbinX > self._fftLen/2 :
            print "fbinX %d > fftLen/2 %d?" %(fbinX,self._fftLen/2)

        ndim   = 2 * self._nSource * ( self._nChan - self._NC )
        # initialize gsl functions
        sys    = multiminimize.gsl_multimin_function_fdf( fun_MK, dfun_MK, fdfun_MK, [fbinX, self, self._NC], ndim )
        solver = multiminimize.conjugate_pr( sys, ndim )
        solver.set(startpoint, STEPSIZE, TOLERANCE )
        waAs = startpoint
        #print "Using solver ", solver.name()
        mi = 10000.0
        preMi = 10000.0
        for itera in range(MAXITNS):
            try: 
                status1 = solver.iterate()
            except errors.gsl_NoProgressError, msg:
                print "No progress error %f" %mi
                print msg
                break
            except:
                print "Unexpected error:"
                raise
            gradient = solver.gradient()
            waAs = solver.getx()
            mi   = solver.getf()
            status2 = multiminimize.test_gradient( gradient, STOPTOLERANCE )

            if fbinX % 10 == 0:
                print 'EK %d %d %e' %(fbinX, itera, mi)
            if status2==0 :
                print 'EK Converged %d %d %e' %(fbinX, itera,mi)
                break
            diff = abs( preMi - mi )
            if diff < DIFFSTOPTOLERANCE:
                print 'EK Converged %d %d %e (%e)' %(fbinX, itera,mi, diff)
                break
            preMi = mi

        #print '=== %d' %(fbinX)
        return waAs

# @class maximum empirical kurtosis beamformer.
#        The entire weight is normalized at each step in the steepest gradient algorithm.
# usage:
# 1. construct an object, mkBf = MEKSubbandBeamformer_nrm( spectralSources  )
# 2. calculate the fixed weights, mkBf.calcFixedWeights( sampleRate, delay )
# 3. accumulate input vectors, mkBf.accumObservations( sFrame, eFrame, R )
# 4. calculate the covariance matricies of the inputs, mkBf.calcCov()
# 5. estimate active weight vectors, mkBf.estimateActiveWeights( fbinX, startpoint )
class MEKSubbandBeamformer_nrm(MEKSubbandBeamformer_pr):
    def __init__(self, spectralSources, nSource=1, NC=1, beta=-1, alpha = 0.1, halfBandShift=False  ):
        MKSubbandBeamformer.__init__(self, spectralSources, nSource, NC, alpha, halfBandShift )
        self._beta = beta

    def normalizeWeight( self, srcX, fbinX, wa ):
        nrm_wa2 = innerproduct(wa, conjugate(wa))
        nrm_wa  = sqrt( nrm_wa2.real )
        if self._beta < 0:
            beta = sqrt( innerproduct(self._wq[srcX][fbinX],conjugate(self._wq[srcX][fbinX])) )
        else:
            beta = self._beta
        if nrm_wa >= 1.0:
            wa  = beta * wa / nrm_wa

        return wa

    def normalizeWa(self, fbinX, wa_f):
        wa = []
        for srcX in range(self._nSource):
            wa.append( self.normalizeWeight( srcX, fbinX, wa_f[srcX] ) )
            
        return wa
    
    def calcEntireWeights_f(self, fbinX, wa_f ):
        """@breif calculate and normalize the entire weight vector of the beamformer for each bin"""
        """@param fbinX  : the index of the subband frequency bin"""
        """@param wa_f[nSource][nChan-NC]    """

        for srcX in range(self._nSource):
            wa = self.normalizeWeight(  srcX, fbinX, wa_f[srcX] )
            self._wo[srcX][fbinX] = self._wq[srcX][fbinX] - Numeric.matrixmultiply( self._B[srcX][fbinX], wa )
            
        return self._wo

    def estimateActiveWeights( self, fbinX, startpoint, MAXITNS=40, TOLERANCE=1.0E-03, STOPTOLERANCE = 1.0E-02, DIFFSTOPTOLERANCE= 1.0E-05, STEPSIZE=0.01 ):
        # @brief estimate active weight vectors at a frequency bin
        # @param fbinX: the frequency bin index you process
        # @param startpoint: the initial active weight vector
        # @param NC: the number of constrants (not yet implemented)
        # @param MAXITNS: the maximum interation for the gradient algorithm
        # @param TOLERANCE : tolerance for the linear search
        # @param STOPTOLERANCE : tolerance for the gradient algorithm
        
        if fbinX > self._fftLen/2 :
            print "fbinX %d > fftLen/2 %d?" %(fbinX,self._fftLen/2)

        ndim   = 2 * self._nSource * ( self._nChan - self._NC )
        # initialize gsl functions
        sys    = multiminimize.gsl_multimin_function_fdf( fun_MK, dfun_MK, fdfun_MK, [fbinX, self, self._NC], ndim )
        solver = multiminimize.steepest_descent( sys, ndim )
        solver.set(startpoint, STEPSIZE, TOLERANCE )
        waAs = startpoint
        #print "Using solver ", solver.name()
        mi = 10000.0
        preMi = 10000.0
        for itera in range(MAXITNS):
            try: 
                status1 = solver.iterate()
            except errors.gsl_NoProgressError, msg:
                print "No progress error %f" %mi
                print msg
                break
            except:
                print "Unexpected error:"
                raise
            gradient = solver.gradient()
            waAs = solver.getx()
            mi   = solver.getf()
            status2 = multiminimize.test_gradient( gradient, STOPTOLERANCE )

            if fbinX % 10 == 0 :
                print 'EK %d %d %e' %(fbinX, itera, mi)
            if status2==0 :
                print 'EK Converged %d %d %e' %(fbinX, itera,mi)
                break
            diff = abs( preMi - mi )
            if diff < DIFFSTOPTOLERANCE:
                print 'EK Converged %d %d %e (%e)' %(fbinX, itera,mi, diff)
                break
            preMi = mi

        #print '=== %d' %(fbinX)
        # Unpack current weights and normalize them
        wa = Numeric.zeros( self._nChan - self._NC, Numeric.Complex)
        for chanX in range( self._nChan - self._NC ):
            wa[chanX] = waAs[2 * chanX] + 1j * waAs[2 * chanX + 1]
        wa = self.normalizeWeight( 0, fbinX, wa )
        for chanX in range( self._nChan - self._NC ):
            waAs[2*chanX]     = wa[chanX].real
            waAs[2*chanX + 1] = wa[chanX].imag

        #print waAs
        return waAs

