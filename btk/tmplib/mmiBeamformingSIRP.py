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

from pygsl import *
from pygsl import multiminimize
from pygsl import sf
import pygsl.errors as errors

from btk import dbase
from btk.modulated import *
from btk.subbandBeamforming import *
from btk.beamformer import *

ALPHA         = 1.0E-02
MAXITNS       = 15
TOLERANCE     = 1.0E-03 # The accuracy of the line minimization
STOPTOLERANCE = 1.0E-03
DIFFSTOPTOLERANCE = 5.0E-06
STEPSIZE = 0.01
ARGMAX_GAMMA = 70
gamma2 = gammaPdfPtr(2)
gamma4 = gammaPdfPtr(4)

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
            X_t.append( copy.deepcopy( snapShotArray.getSnapShot(fbinX) ) )

        XL.append( X_t )

    del snapShotArray
    return XL

    
def calcCovar( samples ):
    """@brief calculate covariance matricies over all frequency bins"""
    """@param samples[][][] : Subband snapshots [frameN][fftLen][chanN]"""
    """@return SigmaXL[fftLen][chanN][chanN] list of covariance matricies"""

    frameN = len( samples )
    fftLen = len( samples[0] )
    chanN  = len( samples[0][0] )
    #print 'frameN %d, fftLen %d, chanN %d' %(frameN,fftLen,chanN)
    
    SigmaXL = []
    for fbinX in range(fftLen):
        SigmaX = Numeric.zeros( (chanN,chanN), Numeric.Complex )
        SigmaXL.append(SigmaX)

    # zero mean at this time... , mean = Numeric.zeros(chanN).astype(Numeric.Complex)
    for sX in range(frameN):
        for fbinX in range(fftLen):
            # zero mean assumption
            SigmaXL[fbinX] += Numeric.outerproduct( samples[sX][fbinX], conjugate(samples[sX][fbinX]) )

    for fbinX in range(fftLen):
        SigmaXL[fbinX] /= frameN

    return SigmaXL

def calcOverallWeight_f( wqL, BL, waL, nOutput ):
    """@breif calculate the overall weight vector of the beamformer for each bin"""
    """@param wq[nChan]       """
    """@param B[nChan][nChan-NC]"""       
    """@param wa[nChan-NC]    """

    woL = []
    nChan = len( wqL[0] )
    
    for oX in range(nOutput):
        wo = Numeric.zeros( ( nChan, 1 ), Numeric.Complex )
        wo = wqL[oX] - Numeric.matrixmultiply( BL[oX], waL[oX] )
        woL.append( wo )
        
    return woL

def calcGSCOutput_f( fbinX, snapShot_f, wo, halfBandShift ):
    """@breif calculate outputs of the GSC at a subband frequency bin"""
    """@param fbinX """
    """@param snapShot[nChan] """
    """@param wo[nChan] """
    """@param halfBandShift"""
    """@return an output value of a GSC beamformer at a subband frequency bin"""
    """@note this function supports half band shift only"""

    if halfBandShift==False:
        print "Only half band shift is supported"
        sys.exit()

    val   = Numeric.zeros( 1, Numeric.Complex )
    wH    = Numeric.transpose( Numeric.conjugate( wo ) )

    # Calculate complete array output.
    val = Numeric.innerproduct( wH, snapShot_f )

    return val

def calcCovarOutput_f( fbinX, samples, woL, halfBandShift, nOutput ):
    """@brief calculate the covariance matrix of beamformer outputs """
    """@param fbinX : the order of the frequency bin"""
    """@param samples[frameN][fftLen][chanN] : frequency components"""
    """@param woL[2] : the list of overall weights of beamformers"""
    
    if halfBandShift==False:
        print "Only half band shift is supported"
        sys.exit()

    frameN = len( samples )
    outputs = Numeric.zeros( (frameN, nOutput), Numeric.Complex )
    SigmaY  = Numeric.zeros( (nOutput,nOutput), Numeric.Complex )

    for frX in range(frameN):
        for oX in range(nOutput):
            outputs[frX][oX] = calcGSCOutput_f( fbinX, samples[frX][fbinX], woL[oX], halfBandShift )
            
        # zero mean assumption
        SigmaY += Numeric.outerproduct( outputs[frX], conjugate(outputs[frX]) )

    SigmaY /= frameN

    return (outputs,SigmaY)

def calcDet22( mat ):
    """ @brief calculate the determinant of 2 x 2 matrix """
    det = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
    return det.real

def getInverseMat22( mat, det ):
    """ @brief calculate the inverse matrixof 2 x 2 matrix """
    invMat = Numeric.zeros( (2,2), Numeric.Complex )
    invMat[0][0] = mat[1][1]/det
    invMat[0][1] = -mat[0][1]/det
    invMat[1][0] = -mat[1][0]/det
    invMat[1][1] = mat[0][0]/det
    return invMat

def calcS( y, invSigmaY ):
    
    sL = Numeric.matrixmultiply( Numeric.conjugate( y ), invSigmaY )
    s  = Numeric.innerproduct( sL, y )
    #print "S",s
    return s.real

def deltaDet( wo1, wo2, B1, SigmaX, simga2, epsilon2 ):
    
    R1    = epsilon2 * ( wo2 )
    R2    = simga2   * ( wo1 )
    L     = Numeric.matrixmultiply( Numeric.conjugate( Numeric.transpose( B1 ) ), SigmaX )
    delta = Numeric.matrixmultiply( L, (R1 - R2) )

    return delta

def delta_s( wo1, wo2, B1, SigmaX, sigma2, epsilon2, y1, y2, x, s, det, deltaDet ):

    L1 = ( epsilon2 * Numeric.conjugate(y2) - sigma2 * Numeric.conjugate(y1) ) * x
    L2 = Numeric.matrixmultiply( ( Numeric.conjugate(y1) * y2 ) * SigmaX, wo2 )
    L3 = Numeric.matrixmultiply( y2 * Numeric.conjugate(y2) * SigmaX, wo1 )

    deltaN = Numeric.matrixmultiply( Numeric.conjugate( Numeric.transpose( B1 ) ), ( L1 + L2 - L3 ) ) - ( s * deltaDet )
    delta = deltaN / det

    return delta

def deltaAbs_y ( abs_y, wo, B, x ):
    
    x2 = Numeric.outerproduct( x, Numeric.conjugate( x ) )
    L1 = Numeric.matrixmultiply( Numeric.conjugate( Numeric.transpose( B ) ), x2 )
    L2 = Numeric.matrixmultiply( L1, wo )
    delta = -0.5 * L2 / abs_y 

    return delta

def calcX_g2( y2, sigmaY2 ):
    return (( 3.0 / 4.0 ) * ( y2 / sigmaY2 ) )

def calcX_g4( s ):
    return (( 3.0 / 2.0 ) * s )

def calcLogGamma_IC( sigmaY, abs_y, x_g2 ):
    # -1.5632291352502243 = log(2*0.2443) - 0.5 * log(2*pi) - 0.25 * log( 2* 3.0 / 8.0 )
    logL = -1.5632291352502243 - 0.5 * Numeric.log(sigmaY)  - 1.5 * Numeric.log(abs_y) + gamma2.calcLog( x_g2, 13 )
    return logL

def calcLogGamma( s, det, x_g4 ):
    #  -1.2692791215804451 = log(4*0.1949) - 0.5 * log(2*pi) - 0.25 * log(2*3.0/4.0)
    logL = -1.2692791215804451 - 0.75 * Numeric.log(s)  - Numeric.log(det) + gamma4.calcLog( x_g4, 13 )
    return logL

def calcLogK0_IC( sigma, abs_y ):
    # -0.57236494292470008 = - 0.5 * Numeric.log( Numeric.pi )
    logL =  -0.57236494292470008 - Numeric.log( sigma ) - Numeric.log( abs_y ) - 2.0 * abs_y / sigma
    return logL

def calcLogK0( s, det ):
    # -1.7170948287741004 = - 1.5 * Numeric.log( Numeric.pi )
    logL = -1.7170948287741004 + Numeric.log( Numeric.sqrt( 2.0 ) + 4.0 * Numeric.sqrt( s ) ) - Numeric.log( det ) - 1.5 * Numeric.log( s ) - 2.0 * Numeric.sqrt( 2 * s )
    return logL

def calcLogLaplace_IC( sigma2, argKa ):
    """@brief calculate the log-likelihood of Laplace density, where components are independent"""
    """@param det """
    """@param argKa : 2 * sqrt(2) * |y| / sigma"""
    # 0.1207822376352452 = Numeric.log( 2.0 ) - 0.5 * Numeric.log( Numeric.pi )

    K0 = sf.bessel_K0( argKa )
    logL = 0.1207822376352452 -  Numeric.log( sigma2 ) + Numeric.log( K0[0] )
    
    return logL

def calcLogLaplace( det, s, argKb ):
    """@brief calculate the joint log-likelihood of Laplace density"""
    """@param det """
    """@param s   """
    """@param argKb : 4 * sqrt(s) """
    # 1.0554938934656808 = Numeric.log( 16.0 ) - 1.5 * Numeric.log( Numeric.pi )

    K1 = sf.bessel_K1( argKb )
    logL = 1.0554938934656808 -  Numeric.log( det ) - 0.5 * Numeric.log( s ) + Numeric.log( K1[0] )
        
    return logL

def delta_sigma( wo, B, SigmaX, sigma ):
    L1 = Numeric.matrixmultiply( Numeric.conjugate( Numeric.transpose( B ) ), SigmaX )
    delta = Numeric.matrixmultiply( L1, wo )
    delta = -0.5 * delta / sigma

    return delta

def delta_sigma2( wo, B, SigmaX, sigma ):
    L1 = Numeric.matrixmultiply( Numeric.conjugate( Numeric.transpose( B ) ), SigmaX )
    delta = -1.0 * Numeric.matrixmultiply( L1, wo )

    return delta

def calcLogDeltaGamma_IC( sigma, abs_y, sigma2, y2, delta_sigma, deltaAbs_y, x_g2 ):
    g2 = gamma2.calcLog( x_g2, 13 )
    tmpV = ( 1.5 * y2 / sigma2 ) * ( ( deltaAbs_y / abs_y ) - ( delta_sigma / sigma ) )
    delta_g2 = gamma2.calcDerivative1( x_g2, 13 ) * tmpV
    delta = - ( 1.5 * deltaAbs_y / abs_y ) - ( 0.5 * delta_sigma / sigma ) + delta_g2 / g2
    
    return delta

def calcLogDeltaGamma( det, deltaDet, s, delta_s, x_g4 ):
    g4 = gamma4.calcLog( x_g4, 13 )
    tmpV = 1.5 * delta_s
    delta_g4 = gamma4.calcDerivative1( x_g4, 13 ) * tmpV 
    delta = - ( 0.75 * delta_s / s ) - ( deltaDet / det ) + ( delta_g4 / g4 )

    return delta
    
def calcLogDeltaK0_IC( sigma, abs_y, delta_sigma, deltaAbs_y ):
    L1 = ( ( 2.0 * abs_y / ( sigma * sigma ) ) - ( 1.0 / sigma ) ) * delta_sigma
    L2 = ( ( 1.0 / abs_y ) + ( 2.0 / sigma ) ) * deltaAbs_y

    return ( L1 - L2 )

def calcLogDeltaK0( det, deltaDet, s, delta_s ):

    sqrt_2 = 1.4142135623730951
    sqrt_s = Numeric.sqrt(s)
    R1 = ( 2.0 / ( sqrt_2 * sqrt_s + 4 * s ) ) - ( 3.0 / ( 2.0 * s ) ) - ( sqrt_2 / sqrt_s )
    delta = ( - deltaDet / det ) + R1 * delta_s

    return delta

def calcLogDeltaLaplace_IC( sigma, delta_sigma2, argKa, abs_y, deltaAbs_y ):
    """@brief """
    """@param det """
    """@param argKa : 2 * sqrt(2) * |y| / sigma"""
    sigma2 = sigma * sigma
    sqrt_2 = 1.4142135623730951
    K0 = sf.bessel_K0( argKa )
    K1 = sf.bessel_K1( argKa )
    L1 = - delta_sigma2 / sigma2
    R1 = ( K1[0] / K0[0] ) * ( 2.0 *  sqrt_2 / sigma2 )
    R2 = sigma * deltaAbs_y - ( 0.5 * abs_y / sigma ) * delta_sigma2

    return ( L1 - R1 * R2 )

def calcLogDeltaLaplace( det, deltaDet, s, argKb, delta_s ):
    """@brief """
    """@param """
    """@param """
    """@param argK : 4 * sqrt(s) """
    
    L1 = - deltaDet / det
    K0 = sf.bessel_K0( argKb )
    K1 = sf.bessel_K1( argKb )
    K2 = sf.bessel_Kn( 2, argKb )
    R1 = ( 1.0 / ( 2.0 * s ) ) + ( ( K0[0] + K2[0] ) / ( Numeric.sqrt(s) * K1[0] ) )

    return ( L1 - R1 * delta_s )

XTIMES = 1.0
def fun_Gamma(x, (wqL, BL, SigmaX, samples, alpha, fbinX, halfBandShift, NC )):
    """@brief calculate the mutual information at a step"""
    """@param x[2(chanN-NC)] : active weights (packed)"""
    """@param wqL[nChan] : queiscent weights"""
    """@param BL[nChan][nChan-NC] : blocking matrices"""
    """@param SigmaX[nChan][nChan] : the covariance matrix of inputs"""
    """@param samples[frameN][fftLen][chanN] : inputs"""

    chanN = len(wqL[0])
    if chanN != len(wqL[1]):
        print 'ERROR: The sizes of quiescent weight vectors must be the same'

    frameN = len( samples )
    fftLen = len( samples[0] )
    
    # Unpack current weights
    waL = []
    wa1   = Numeric.zeros(chanN-NC, Numeric.Complex)
    wa2   = Numeric.zeros(chanN-NC, Numeric.Complex)
    for chanX in range(chanN-NC):
        wa1[chanX] = x[2 * chanX] + 1j * x[2 * chanX + 1]
    for chanX in range(chanN-NC):
        wa2[chanX] = x[2 * chanX + 2 * (chanN - NC)] + 1j * x[2 * chanX + 1 + 2 * (chanN - NC)]
    waL.append( wa1 )
    waL.append( wa2 )
    
    # Calculate mutual information
    lPr = Numeric.zeros(1, Numeric.Complex)
    woL = calcOverallWeight_f( wqL, BL, waL, 2 )
    
    (y,SigmaY) = calcCovarOutput_f( fbinX, samples, woL, halfBandShift, 2 )
    det = calcDet22( SigmaY )
    iSigmaY = getInverseMat22( SigmaY, det )
    sigma_11 = Numeric.sqrt( SigmaY[0][0].real )
    sigma_22 = Numeric.sqrt( SigmaY[1][1].real )
    for frX in range(frameN):
        s = calcS( y[frX], iSigmaY )
        abs_y1 = Numeric.absolute( y[frX][0] )
        abs_y2 = Numeric.absolute( y[frX][1] )
        x_g21 = calcX_g2( abs_y1 * abs_y1, SigmaY[0][0].real )
        x_g22 = calcX_g2( abs_y2 * abs_y2, SigmaY[1][1].real )
        x_g4  = calcX_g4( s )
        if( x_g21 >= ARGMAX_GAMMA or x_g22 >= ARGMAX_GAMMA or x_g4 >= ARGMAX_GAMMA ):
            print 'O %d %e %e %e' %(frX,x_g21,x_g22,x_g4)
            continue
        
        lPr += calcLogGamma( s, det, x_g4 )
        lPr -= calcLogGamma_IC( sigma_11, abs_y1, x_g21 )
        lPr -= calcLogGamma_IC( sigma_22, abs_y2, x_g22 )
    
    mi = lPr * XTIMES / frameN
    #mi = lPr
    #print "fun_Gamma, MI",fbinX,lPr,waL
    
    # a regularization term
    rterm  =  alpha * innerproduct(wa1, conjugate(wa1)) + alpha * innerproduct(wa2, conjugate(wa2))
    mi += rterm.real

    if fbinX == 100:
        print 'rterm = %g : alpha = %g' %(rterm.real, alpha)
    
    return mi.real

def dfun_Gamma(x, (wqL, BL, SigmaX, samples, alpha, fbinX, halfBandShift,NC )):

    chanN = len(wqL[0])
    frameN = len( samples )
    fftLen = len( samples[0] )
    
    # Unpack current weights
    waL   = []
    wa1   = Numeric.zeros(chanN-NC, Numeric.Complex)
    wa2   = Numeric.zeros(chanN-NC, Numeric.Complex)
    for chanX in range(chanN-NC):
        wa1[chanX] = x[2 * chanX] + 1j * x[2 * chanX + 1]
    for chanX in range(chanN-NC):
        wa2[chanX] = x[2 * chanX + 2 * (chanN - NC)] + 1j * x[2 * chanX + 1 + 2 * (chanN - NC)]
    waL.append( wa1 )
    waL.append( wa2 )
    
    # Calculate complex gradient
    deltaWa1 = Numeric.zeros(chanN - NC, Numeric.Complex)
    deltaWa2 = Numeric.zeros(chanN - NC, Numeric.Complex)
    woL = calcOverallWeight_f( wqL, BL, waL, 2 )
    
    (y,SigmaY) = calcCovarOutput_f( fbinX, samples, woL, halfBandShift, 2 )
    det = calcDet22( SigmaY )
    iSigmaY = getInverseMat22( SigmaY, det )

    dDet1 = deltaDet( woL[0], woL[1], BL[0], SigmaX, SigmaY[1][1].real, SigmaY[1][0] )
    sigma_11 = Numeric.sqrt( SigmaY[0][0].real )
    d_sigma1 = delta_sigma( woL[0], BL[0], SigmaX, sigma_11 )

    dDet2 = deltaDet( woL[1], woL[0], BL[1], SigmaX, SigmaY[0][0].real, SigmaY[0][1] )
    sigma_22 = Numeric.sqrt( SigmaY[1][1].real )
    d_sigma2 = delta_sigma( woL[1], BL[1], SigmaX, sigma_22 )
    
    #print "Sigma",SigmaY,iSigmaY,Numeric.matrixmultiply(SigmaY,iSigmaY)
    for frX in range(frameN):
        s = calcS( y[frX], iSigmaY )
        abs_y1 = Numeric.absolute( y[frX][0] )
        abs_y2 = Numeric.absolute( y[frX][1] )
        y12 = abs_y1 * abs_y1
        y22 = abs_y2 * abs_y2
        x_g21 = calcX_g2( y12, SigmaY[0][0].real )
        x_g22 = calcX_g2( y22, SigmaY[1][1].real )
        x_g4  = calcX_g4( s )
        if( x_g21 >= ARGMAX_GAMMA or x_g22 >= ARGMAX_GAMMA or x_g4 >= ARGMAX_GAMMA ):
            continue
        
        # the first sound source
        # the gradient of a log joint probability
        d_s = delta_s( woL[0], woL[1], BL[0], SigmaX, SigmaY[1][1].real, SigmaY[1][0], y[frX][0], y[frX][1], samples[frX][fbinX], s, det, dDet1 )
        deltaWa1 += calcLogDeltaGamma( det, dDet1, s, d_s, x_g4 )
        # the gradient of log probabilities of ICs
        dAbs_y = deltaAbs_y( abs_y1, woL[0], BL[0], samples[frX][fbinX] )
        deltaWa1 -= calcLogDeltaGamma_IC( sigma_11, abs_y1, SigmaY[0][0].real, y12, d_sigma1, dAbs_y, x_g21 )
        
        # the second sound source
        # the gradient of a log joint probability
        d_s = delta_s( woL[1], woL[0], BL[1], SigmaX, SigmaY[0][0].real, SigmaY[0][1], y[frX][1], y[frX][0], samples[frX][fbinX], s, det, dDet2 )
        deltaWa2 += calcLogDeltaGamma( det, dDet2, s, d_s, x_g4 )
        # the gradient of log probabilities of ICs
        dAbs_y = deltaAbs_y( abs_y2, woL[1], BL[1], samples[frX][fbinX] )
        deltaWa2 -= calcLogDeltaGamma_IC( sigma_22, abs_y2, SigmaY[1][1].real, y22, d_sigma2, dAbs_y, x_g22 )

    deltaWa1 = deltaWa1 * XTIMES / frameN
    deltaWa2 = deltaWa2 * XTIMES / frameN

    # a regularization term
    deltaWa1 += alpha * wa1
    deltaWa2 += alpha * wa2

    # Pack the gradient
    grad = Numeric.zeros(4 * (chanN - NC), Numeric.Float)
    for chanX in range(chanN - NC):
        grad[2*chanX]                       = deltaWa1[chanX].real
        grad[2*chanX + 1]                   = deltaWa1[chanX].imag

    for chanX in range(chanN - NC):
        grad[2*chanX + 2 * (chanN - NC)]     = deltaWa2[chanX].real
        grad[2*chanX + 1 + 2 * (chanN - NC)] = deltaWa2[chanX].imag

    #print grad
    del woL
    del waL
    del deltaWa1
    del deltaWa2
    return grad

def fdfun_Gamma(x, (wqL, BL, SigmaX, samples, alpha, fbinX, halfBandShift, NC ) ):
    f  = fun_Gamma(x, (wqL, BL, SigmaX, samples, alpha, fbinX, halfBandShift, NC ) )
    df = dfun_Gamma(x, (wqL, BL, SigmaX, samples, alpha, fbinX, halfBandShift, NC ) )

    return f, df

def fun_K0(x, (wqL, BL, SigmaX, samples, alpha, fbinX, halfBandShift, NC )):
    """@brief calculate the mutual information at a step"""
    """@param x[2(chanN-NC)] : active weights (packed)"""
    """@param wqL[nChan] : queiscent weights"""
    """@param BL[nChan][nChan-NC] : blocking matrices"""
    """@param SigmaX[nChan][nChan] : the covariance matrix of inputs"""
    """@param samples[frameN][fftLen][chanN] : inputs"""

    chanN = len(wqL[0])
    if chanN != len(wqL[1]):
        print 'ERROR: The sizes of quiescent weight vectors must be the same'

    frameN = len( samples )
    fftLen = len( samples[0] )
    
    # Unpack current weights
    waL = []
    wa1   = Numeric.zeros(chanN-NC, Numeric.Complex)
    wa2   = Numeric.zeros(chanN-NC, Numeric.Complex)
    for chanX in range(chanN-NC):
        wa1[chanX] = x[2 * chanX] + 1j * x[2 * chanX + 1]
    for chanX in range(chanN-NC):
        wa2[chanX] = x[2 * chanX + 2 * (chanN - NC)] + 1j * x[2 * chanX + 1 + 2 * (chanN - NC)]
    waL.append( wa1 )
    waL.append( wa2 )
    
    # Calculate mutual information
    lPr = Numeric.zeros(1, Numeric.Complex)
    woL = calcOverallWeight_f( wqL, BL, waL, 2 )
    
    (y,SigmaY) = calcCovarOutput_f( fbinX, samples, woL, halfBandShift, 2 )
    det = calcDet22( SigmaY )
    iSigmaY = getInverseMat22( SigmaY, det )
    sigma_11 = Numeric.sqrt( SigmaY[0][0].real )
    sigma_22 = Numeric.sqrt( SigmaY[1][1].real )
    for frX in range(frameN):
        s = calcS( y[frX], iSigmaY )
        lPr += calcLogK0( s, det )

        abs_y = Numeric.absolute( y[frX][0] )
        lPr -= calcLogK0_IC( sigma_11, abs_y )

        abs_y = Numeric.absolute( y[frX][1] )
        lPr -= calcLogK0_IC( sigma_22, abs_y )
    
    mi = lPr * XTIMES / frameN
    #mi = lPr
    #print "fun_K0, MI",fbinX,lPr,waL
    
    # a regularization term
    rterm  =  alpha * innerproduct(wa1, conjugate(wa1)) + alpha * innerproduct(wa2, conjugate(wa2))
    mi += rterm.real

    if fbinX == 100:
        print 'rterm = %g : alpha = %g' %(rterm.real, alpha)
    
    return mi.real

def dfun_K0(x, (wqL, BL, SigmaX, samples, alpha, fbinX, halfBandShift,NC )):

    chanN = len(wqL[0])
    frameN = len( samples )
    fftLen = len( samples[0] )
    
    # Unpack current weights
    waL   = []
    wa1   = Numeric.zeros(chanN-NC, Numeric.Complex)
    wa2   = Numeric.zeros(chanN-NC, Numeric.Complex)
    for chanX in range(chanN-NC):
        wa1[chanX] = x[2 * chanX] + 1j * x[2 * chanX + 1]
    for chanX in range(chanN-NC):
        wa2[chanX] = x[2 * chanX + 2 * (chanN - NC)] + 1j * x[2 * chanX + 1 + 2 * (chanN - NC)]
    waL.append( wa1 )
    waL.append( wa2 )
    
    # Calculate complex gradient
    deltaWa1 = Numeric.zeros(chanN - NC, Numeric.Complex)
    deltaWa2 = Numeric.zeros(chanN - NC, Numeric.Complex)
    woL = calcOverallWeight_f( wqL, BL, waL, 2 )
    
    (y,SigmaY) = calcCovarOutput_f( fbinX, samples, woL, halfBandShift, 2 )
    det = calcDet22( SigmaY )
    iSigmaY = getInverseMat22( SigmaY, det )

    dDet1 = deltaDet( woL[0], woL[1], BL[0], SigmaX, SigmaY[1][1].real, SigmaY[1][0] )
    sigma_11 = Numeric.sqrt( SigmaY[0][0].real )
    d_sigma1 = delta_sigma( woL[0], BL[0], SigmaX, sigma_11 )

    dDet2 = deltaDet( woL[1], woL[0], BL[1], SigmaX, SigmaY[0][0].real, SigmaY[0][1] )
    sigma_22 = Numeric.sqrt( SigmaY[1][1].real )
    d_sigma2 = delta_sigma( woL[1], BL[1], SigmaX, sigma_22 )
    
    #print "Sigma",SigmaY,iSigmaY,Numeric.matrixmultiply(SigmaY,iSigmaY)
    for frX in range(frameN):
        s = calcS( y[frX], iSigmaY )
        
        # the first sound source
        # the gradient of a log joint probability
        d_s = delta_s( woL[0], woL[1], BL[0], SigmaX, SigmaY[1][1].real, SigmaY[1][0], y[frX][0], y[frX][1], samples[frX][fbinX], s, det, dDet1 )
        deltaWa1 += calcLogDeltaK0( det, dDet1, s, d_s )
        # the gradient of log probabilities of ICs
        abs_y = Numeric.absolute( y[frX][0] )
        dAbs_y = deltaAbs_y( abs_y, woL[0], BL[0], samples[frX][fbinX] )
        deltaWa1 -= calcLogDeltaK0_IC( sigma_11, abs_y, d_sigma1, dAbs_y )
        
        # the second sound source
        d_s = delta_s( woL[1], woL[0], BL[1], SigmaX, SigmaY[0][0].real, SigmaY[0][1], y[frX][1], y[frX][0], samples[frX][fbinX], s, det, dDet2 )
        deltaWa2 += calcLogDeltaK0( det, dDet2, s, d_s )        
        abs_y = Numeric.absolute( y[frX][1] )
        dAbs_y = deltaAbs_y( abs_y, woL[1], BL[1], samples[frX][fbinX] )
        deltaWa2 -= calcLogDeltaK0_IC( sigma_22, abs_y,  d_sigma2, dAbs_y )

    deltaWa1 = deltaWa1 * XTIMES / frameN
    deltaWa2 = deltaWa2 * XTIMES / frameN

    # a regularization term
    deltaWa1 += alpha * wa1
    deltaWa2 += alpha * wa2

    # Pack the gradient
    grad = Numeric.zeros(4 * (chanN - NC), Numeric.Float)
    for chanX in range(chanN - NC):
        grad[2*chanX]                       = deltaWa1[chanX].real
        grad[2*chanX + 1]                   = deltaWa1[chanX].imag

    for chanX in range(chanN - NC):
        grad[2*chanX + 2 * (chanN - NC)]     = deltaWa2[chanX].real
        grad[2*chanX + 1 + 2 * (chanN - NC)] = deltaWa2[chanX].imag

    #print grad
    del woL
    del waL
    del deltaWa1
    del deltaWa2
    return grad

def fdfun_K0(x, (wqL, BL, SigmaX, samples, alpha, fbinX, halfBandShift, NC ) ):
    f  = fun_K0(x, (wqL, BL, SigmaX, samples, alpha, fbinX, halfBandShift, NC ) )
    df = dfun_K0(x, (wqL, BL, SigmaX, samples, alpha, fbinX, halfBandShift, NC ) )

    return f, df

def fun_Laplace(x, (wqL, BL, SigmaX, samples, alpha, fbinX, halfBandShift, NC )):
    """@brief calculate the mutual information at a step"""
    """@param x[2(chanN-NC)] : active weights (packed)"""
    """@param wqL[nChan] : queiscent weights"""
    """@param BL[nChan][nChan-NC] : blocking matrices"""
    """@param SigmaX[nChan][nChan] : the covariance matrix of inputs"""
    """@param samples[frameN][fftLen][chanN] : inputs"""

    chanN = len(wqL[0])
    if chanN != len(wqL[1]):
        print 'ERROR: The sizes of quiescent weight vectors must be the same'

    frameN = len( samples )
    fftLen = len( samples[0] )
    
    # Unpack current weights
    waL = []
    wa1   = Numeric.zeros(chanN-NC, Numeric.Complex)
    wa2   = Numeric.zeros(chanN-NC, Numeric.Complex)
    for chanX in range(chanN-NC):
        wa1[chanX] = x[2 * chanX] + 1j * x[2 * chanX + 1]
    for chanX in range(chanN-NC):
        wa2[chanX] = x[2 * chanX + 2 * (chanN - NC)] + 1j * x[2 * chanX + 1 + 2 * (chanN - NC)]
    waL.append( wa1 )
    waL.append( wa2 )
    
    # Calculate mutual information
    lPr = Numeric.zeros(1, Numeric.Complex)
    woL = calcOverallWeight_f( wqL, BL, waL, 2 )
    
    (y,SigmaY) = calcCovarOutput_f( fbinX, samples, woL, halfBandShift, 2 )
    det = calcDet22( SigmaY )
    iSigmaY = getInverseMat22( SigmaY, det )
    sigma_11 = Numeric.sqrt( SigmaY[0][0].real )
    sigma_22 = Numeric.sqrt( SigmaY[1][1].real )
    sqrt_2 = 1.4142135623730951
    for frX in range(frameN):
        s = calcS( y[frX], iSigmaY )
        argKb = 4.0 * Numeric.sqrt(s)
        lPr += calcLogLaplace( det, s, argKb )

        abs_y = Numeric.absolute( y[frX][0] )
        argKa = 2.0 * sqrt_2 * abs_y / sigma_11
        lPr -= calcLogLaplace_IC( SigmaY[0][0].real, argKa )

        abs_y = Numeric.absolute( y[frX][1] )
        argKa = 2.0 * sqrt_2 * abs_y / sigma_22
        lPr -= calcLogLaplace_IC( SigmaY[1][1].real, argKa )
    
    mi = lPr * XTIMES / frameN
    #mi = lPr
    #print "fun_Laplace, MI",fbinX,lPr,waL
    
    # a regularization term
    rterm  =  alpha * innerproduct(wa1, conjugate(wa1)) + alpha * innerproduct(wa2, conjugate(wa2))
    mi += rterm.real

    if fbinX == 100:
        print 'rterm(L) = %g : alpha = %g' %(rterm.real, alpha)
    
    return mi.real

def dfun_Laplace(x, (wqL, BL, SigmaX, samples, alpha, fbinX, halfBandShift, NC )):

    chanN = len(wqL[0])
    frameN = len( samples )
    fftLen = len( samples[0] )
    
    # Unpack current weights
    waL   = []
    wa1   = Numeric.zeros(chanN-NC, Numeric.Complex)
    wa2   = Numeric.zeros(chanN-NC, Numeric.Complex)
    for chanX in range(chanN-NC):
        wa1[chanX] = x[2 * chanX] + 1j * x[2 * chanX + 1]
    for chanX in range(chanN-NC):
        wa2[chanX] = x[2 * chanX + 2 * (chanN - NC)] + 1j * x[2 * chanX + 1 + 2 * (chanN - NC)]
    waL.append( wa1 )
    waL.append( wa2 )
    
    # Calculate complex gradient
    deltaWa1 = Numeric.zeros(chanN - NC, Numeric.Complex)
    deltaWa2 = Numeric.zeros(chanN - NC, Numeric.Complex)
    woL = calcOverallWeight_f( wqL, BL, waL, 2 )
    
    (y,SigmaY) = calcCovarOutput_f( fbinX, samples, woL, halfBandShift, 2 )
    det = calcDet22( SigmaY )
    iSigmaY = getInverseMat22( SigmaY, det )

    dDet1 = deltaDet( woL[0], woL[1], BL[0], SigmaX, SigmaY[1][1].real, SigmaY[1][0] )
    sigma_11 = Numeric.sqrt( SigmaY[0][0].real )
    d_sigma2_1 = delta_sigma2( woL[0], BL[0], SigmaX, sigma_11 )

    dDet2 = deltaDet( woL[1], woL[0], BL[1], SigmaX, SigmaY[0][0].real, SigmaY[0][1] )
    sigma_22 = Numeric.sqrt( SigmaY[1][1].real )
    d_sigma2_2 = delta_sigma2( woL[1], BL[1], SigmaX, sigma_22 )
    sqrt_2 = 1.4142135623730951
     
    #print "Sigma",SigmaY,iSigmaY,Numeric.matrixmultiply(SigmaY,iSigmaY)
    for frX in range(frameN):
        s = calcS( y[frX], iSigmaY )
        argKb = 4.0 * Numeric.sqrt(s)
        
        # the first sound source
        # the gradient of a log joint probability
        d_s = delta_s( woL[0], woL[1], BL[0], SigmaX, SigmaY[1][1].real, SigmaY[1][0], y[frX][0], y[frX][1], samples[frX][fbinX], s, det, dDet1 )
        deltaWa1 += calcLogDeltaLaplace( det, dDet1, s, argKb, d_s )
        # the gradient of log probabilities of ICs
        abs_y = Numeric.absolute( y[frX][0] )
        argKa = 2.0 * sqrt_2 * abs_y / sigma_11
        dAbs_y = deltaAbs_y( abs_y, woL[0], BL[0], samples[frX][fbinX] )
        deltaWa1 -= calcLogDeltaLaplace_IC( sigma_11, d_sigma2_1, argKa, abs_y, dAbs_y )
        
        # the second sound source
        d_s = delta_s( woL[1], woL[0], BL[1], SigmaX, SigmaY[0][0].real, SigmaY[0][1], y[frX][1], y[frX][0], samples[frX][fbinX], s, det, dDet2 )
        deltaWa2 += calcLogDeltaLaplace( det, dDet2, s, argKb, d_s )
        abs_y = Numeric.absolute( y[frX][1] )
        argKa = 2.0 * sqrt_2 * abs_y / sigma_22
        dAbs_y = deltaAbs_y( abs_y, woL[1], BL[1], samples[frX][fbinX] )
        deltaWa2 -= calcLogDeltaLaplace_IC( sigma_22, d_sigma2_2, argKa, abs_y, dAbs_y )

    deltaWa1 = deltaWa1 * XTIMES / frameN
    deltaWa2 = deltaWa2 * XTIMES / frameN

    # a regularization term
    deltaWa1 += alpha * wa1
    deltaWa2 += alpha * wa2

    # Pack the gradient
    grad = Numeric.zeros(4 * (chanN - NC), Numeric.Float)
    for chanX in range(chanN - NC):
        grad[2*chanX]                       = deltaWa1[chanX].real
        grad[2*chanX + 1]                   = deltaWa1[chanX].imag

    for chanX in range(chanN - NC):
        grad[2*chanX + 2 * (chanN - NC)]     = deltaWa2[chanX].real
        grad[2*chanX + 1 + 2 * (chanN - NC)] = deltaWa2[chanX].imag

    #print grad
    return grad

def fdfun_Laplace(x, (wqL, BL, SigmaX, samples, alpha, fbinX, halfBandShift, NC ) ):
    f  = fun_Laplace(x, (wqL, BL, SigmaX, samples, alpha, fbinX, halfBandShift, NC ) )
    df = dfun_Laplace(x, (wqL, BL, SigmaX, samples, alpha, fbinX, halfBandShift, NC ) )

    return f, df

def mmiBeamformingWeights_f( fbinX, samples, SigmaX, arrgeom, delayL, startpoint, sampleRate, fftLen, halfBandShift, pdfKind='K0' ):
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

    wqL = []
    BL  = []
    wqL.append( wq1 )
    BL.append( B1 )
    wqL.append( wq2 )
    BL.append( B2 )
    
    # initialize gsl functions
    if pdfKind == 'K0':
        sys    = multiminimize.gsl_multimin_function_fdf( fun_K0, dfun_K0, fdfun_K0, [wqL, BL, SigmaX, samples, ALPHA, fbinX, halfBandShift, NC], ndim )
    elif pdfKind == 'Gamma':
        sys    = multiminimize.gsl_multimin_function_fdf( fun_Gamma, dfun_Gamma, fdfun_Gamma, [wqL, BL, SigmaX, samples, ALPHA, fbinX, halfBandShift, NC], ndim )
    else:
        pdfKind = 'Laplace'
        sys    = multiminimize.gsl_multimin_function_fdf( fun_Laplace, dfun_Laplace, fdfun_Laplace, [wqL, BL, SigmaX, samples, ALPHA, fbinX, halfBandShift, NC], ndim )
    
    solver = multiminimize.conjugate_pr_eff( sys, ndim )
    #solver = multiminimize.vector_bfgs( sys, ndim )
    #solver = multiminimize.steepest_descent( sys, ndim )
    solver.set(startpoint, STEPSIZE, TOLERANCE )
    waAs = startpoint
    #print "Using solver ", solver.name()
    mi = 10000.0
    preMi = 10000.0
    for itera in range(MAXITNS):
        try: 
            status1 = solver.iterate()
        except errors.gsl_NoProgressError, msg:
            print "No progress error"
            print msg
            break
        except:
            print "Unexpected error:", sys.exc_info()[0]
            raise
        gradient = solver.gradient()
        waAs = solver.getx()
        mi   = solver.getf()
        status2 = multiminimize.test_gradient( gradient, STOPTOLERANCE )

        if fbinX == 0 or fbinX == 100 or fbinX==200 :
            print 'MI(%s) %d %d %f' %(pdfKind, fbinX, itera, mi)
            print '||w_a|| ', abs(waAs)
        if status2==0 :
            print 'MI(%s) Converged %d %d %f' %(pdfKind, fbinX, itera,mi)
            break
        diff = abs( preMi - mi )
        if diff < DIFFSTOPTOLERANCE:
            print 'MI(%s) Converged %d %d %f (%f)' %(pdfKind, fbinX, itera,mi, diff)
            break
        preMi = mi

    # Unpack current weights
    wa1A = Numeric.zeros(chanN-NC, Numeric.Complex)
    wa2A = Numeric.zeros(chanN-NC, Numeric.Complex)
    weights = []
    for chanX in range(chanN-NC):
        wa1A[chanX] = waAs[2 * chanX] + 1j * waAs[2 * chanX + 1]
    for chanX in range(chanN-NC):
        wa2A[chanX] = waAs[2 * chanX + 2 * (chanN - NC)] + 1j * waAs[2 * chanX + 1 + 2 * (chanN - NC)]

    weights.append(wa1A)
    weights.append(wa2A)
    del wqL
    del BL
    
    return weights

def mmiBeamformingWeights2_f( fbinX, samples, SigmaX, arrgeom, delayL, startpoint, sampleRate, fftLen, halfBandShift, pdfKind='K0' ):
    """@brief find the point which gives the minimum mutual information between two sources"""
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
    
    wqL = []
    BL  = []
    wqL.append( wq1 )
    BL.append( B1 )
    wqL.append( wq2 )
    BL.append( B2 )
    
    # initialize gsl functions
    if pdfKind == 'K0':
        sys    = multiminimize.gsl_multimin_function_fdf( fun_K0, dfun_K0, fdfun_K0, [wqL, BL, SigmaX, samples, ALPHA, fbinX, halfBandShift, NC], ndim )
    elif pdfKind == 'Gamma':
        sys    = multiminimize.gsl_multimin_function_fdf( fun_Gamma, dfun_Gamma, fdfun_Gamma, [wqL, BL, SigmaX, samples, ALPHA, fbinX, halfBandShift, NC], ndim )
    else:
        pdfKind = 'Laplace'
        sys    = multiminimize.gsl_multimin_function_fdf( fun_Laplace, dfun_Laplace, fdfun_Laplace, [wqL, BL, SigmaX, samples, ALPHA, fbinX, halfBandShift, NC], ndim )
    
    solver = multiminimize.conjugate_pr_eff( sys, ndim )
    #solver = multiminimize.vector_bfgs( sys, ndim )
    #solver = multiminimize.steepest_descent( sys, ndim )
    solver.set(startpoint, STEPSIZE, TOLERANCE )
    waAs = startpoint
    #print "Using solver ", solver.name()
    mi = 10000.0
    preMi = 10000.0
    for itera in range(MAXITNS):
        try: 
            status1 = solver.iterate()
        except errors.gsl_NoProgressError, msg:
            print "No progress error"
            print msg
            break
        except:
            print "Unexpected error:", sys.exc_info()[0]
            raise
        gradient = solver.gradient()
        waAs = solver.getx()
        mi   = solver.getf()
        status2 = multiminimize.test_gradient( gradient, STOPTOLERANCE )

        if fbinX == 0 or fbinX == 100 or fbinX==200 :
            print 'MI(%s) %d %d %f' %(pdfKind, fbinX, itera, mi)
            print '||w_a|| ', abs(waAs)
        if status2==0 :
            print 'MI(%s) Converged %d %d %f' %(pdfKind,fbinX, itera,mi)
            break
        diff = abs( preMi - mi )
        if diff < DIFFSTOPTOLERANCE:
            print 'MI(%s) Converged %d %d %f (%f)' %(pdfKind,fbinX, itera,mi, diff)
            break
        preMi = mi

    # Unpack current weights
    wa1A = Numeric.zeros(chanN-NC, Numeric.Complex)
    wa2A = Numeric.zeros(chanN-NC, Numeric.Complex)
    weights = []
    for chanX in range(chanN-NC):
        wa1A[chanX] = waAs[2 * chanX] + 1j * waAs[2 * chanX + 1]
    for chanX in range(chanN-NC):
        wa2A[chanX] = waAs[2 * chanX + 2 * (chanN - NC)] + 1j * waAs[2 * chanX + 1 + 2 * (chanN - NC)]

    weights.append(wa1A)
    weights.append(wa2A)

    del wqL
    del BL
    return weights
