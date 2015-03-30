#! /usr/bin/env python

from btk import dbase
import os, sys, time
import os.path
from btk.fmatrix import *
from btk.utils import *
from btk.subBandBeamforming import *
from CepstralFrontend import *
# from btk.cepstralFrontend import *
from lmsBeamforming import *
# from btk.lmsBeamforming import *

# Specify the geometry of the array.
arrgeom = array([[   0. , -307.5,    0. ],
                 [   0. , -266.5,    0. ],
                 [   0. , -225.5,    0. ],
                 [   0. , -184.5,    0. ],
                 [   0. , -143.5,    0. ],
                 [   0. , -102.5,    0. ],
                 [   0. ,  -61.5,    0. ],
                 [   0. ,  -20.5,    0. ],
                 [   0. ,   20.5,    0. ],
                 [   0. ,   61.5,    0. ],
                 [   0. ,  102.5,    0. ],
                 [   0. ,  143.5,    0. ],
                 [   0. ,  184.5,    0. ],
                 [   0. ,  225.5,    0. ],
                 [   0. ,  266.5,    0. ],
                 [   0. ,  307.5,    0. ]])

def fopen(filename, mode = 'r', bufsize = -1, trials = 1000, secs = 5):
    """
    A file open routine that copes with network delays by attempting to open a file several times
    filename, mode, bufsize: see builtin function open() for description

    trials: number of attempts that will be made to open the file
    secs: number of seconds the routine sleeps until it retries
    """
    while (trials > 1):
        try:
            res = open(filename, mode, bufsize)
            return res
        except IOError:
            trials -= 1
            print "fopen(): Failed to open file %s in mode %s, will retry %d times." % (filename, mode, trials)
            time.sleep(secs)
    return open(filename, mode, bufsize)


def safeIO(func, filehandle, trials = 1000, secs = 5):
    """
    A routine to cope with network delays by repeatedly attempting to perform func on filehandle

    func: function that has one single free paramter (consider using lambda form), that is a filename or descriptor
    filehandle: filename or descriptor for a function to operate on
    trials: number of attempts that will be made to perform func
    secs: number of seconds the routine sleeps until it retries
    """
    while (trials > 1):
        try:
            res = func(filehandle)
            return res
        except IOError:
            trials -= 1
            print "safeIO(): Failed to use file %s, will retry %d times." % (filehandle, trials)
            time.sleep(secs)
    return func(filehandle)


def procUtts():

    # Specify filter bank parameters.
    nChan       = len(arrgeom)
    fftLen      = 512
    nBlocks     = 0
    subSampRate = 2
    sampleRate  = 16000.0

    # Specify beamforming parameters
    playBack    = 0
    plot        = 0
    betaval     = .99
    gammaval    = 1e6
    boundval    = 1e-7
    tapsval     = 1
    depval      = 0
    bftypeval   = 0
    noisyval    = 0


    # Specify source location.
    x_s = 2075.0
    y_s = 63.5
    z_s = 0.0

    # Specify paths for array data and database.
    inDataDir   = '/project/micArrayData/segmented_data/esst-testset_shorten'
    outDataDir  = '/project/draub/mfcc_lms_g1e6_b1e-7_oracle'
    vitPathDir  = '/project/draub/cbk_ds-oracle/'
#     vitPathDir  = '/project/draub/cbk_ds/'
    dataBaseDir = '/home/jmcd/BeamformingExpts/1000/'
    spkFile     = 'spk.DB2006.test.cur'
    spkLst      = FGet(spkFile)

    # Status report
    print 'This is %s running with : \n' % sys.argv[0]
    print 'nChan       = ', nChan
    print 'fftLen      = ', fftLen
    print 'nBlocks     = ', nBlocks
    print 'subSampRate = ', subSampRate
    print 'sampleRate  = ', sampleRate
    print 'playBack    = ', playBack
    print 'plot        = ', plot
    print 'betaval     = ', betaval
    print 'gammaval    = %e' % gammaval
    print 'boundval    = %e' % boundval
    print 'tapsval     = ', tapsval
    print 'depval      = ', depval
    print 'bftypeval   = ', bftypeval
    print 'noisyval    = %e' % noisyval
    print '\n'
    print 'Source location :'
    print '\tx_s = ', x_s
    print '\ty_s = ', y_s
    print '\tz_s = ', z_s
    print '\n'
    print 'Path settings : '
    print 'inDataDir   = ', inDataDir
    print 'outDataDir  = ', outDataDir
    print 'vitPathDir  = ', vitPathDir
    print 'dataBaseDir = ', dataBaseDir
    print 'spkFile     = ', spkFile
    print '\n\n'




    db = dbase.DB200x(dataBaseDir+'DB2006-spk', dataBaseDir+'DB2006-utt')

    analysisFBs = []
    for i in range(nChan):
        analysisFBs.append(FftFB(fftLen, nBlocks, subSampRate))
#         analysisFBs.append(AnalysisFB(fftLen, nBlocks, subSampRate))

    # Build the analysis chain and MFCC generation
    # beamFormer = SubBandBeamformerDS(analysisFBs)
    # beamFormer = SubBandBeamformerGriffithsJim(analysisFBs, fixedDiagLoad = diagLoad, plotting = plot)
    # beamFormer = SubBandBeamformerRLS(analysisFBs, fixedDiagLoad = diagLoad, initDiagLoad = diagLoad, plotting = plot)
    # beamFormer = SubBandBeamformerMPDRSMI(analysisFBs, diagLoad = diagLoad, plotting = plot)
    beamFormer = SubBandBeamformerLmsHmm(analysisFBs, plotting = plot, beta = betaval, gamma = gammaval, bound = boundval, taps = tapsval, dep = depval, bftype = bftypeval ,noisy = noisyval)
    delays = calcDelays(x_s, y_s, z_s, arrgeom)
    beamFormer.calcArrayManifoldVectors(sampleRate, delays)

    delays = calcDelays(1e10, -1e10, z_s, arrgeom)
    beamFormer.calcNoiseManifold(sampleRate, delays)

    print beamFormer
    print '\n\n'
    print type(beamFormer)
    print '\n\n'

#     mel = MelFB(beamFormer)
#     frontend = CepstralFeatures(mel)

    frontend = beamFormer

    powfront  = PowerFrontend()

    # Loop over speakers.
    ttlSamples = 0
    for spkId in spkLst:
        # Loop over utterances for this speaker.
        beamFormer.nextSpkr(spkId)
        spk = db.getSpeaker(spkId)
        uttCnt = 0
        for utt in spk.utt:
            print "Processing %s" %(utt.file)

            nextViterbi = ("%s/%s.path" %(vitPathDir, os.path.basename(utt.file)[:-3]+"_"+spkId[:3]))
            print "we now attempt to hook the beamformer up to a viterbi path ... ", nextViterbi
            beamFormer.nextUtt(nextViterbi)
            print "the viterbi path is hooked up"
            try:
                # Initialize a 'soundSource' for each channel in the array.
                for i in range(nChan):
                    nextFile = ("%s/%s.%s.adc.shn" %(inDataDir, utt.file, (i+1)))
                    soundSource = safeIO(lambda x: OffsetCorrectedFileSoundSource(x, blkLen = fftLen,  lastBlk = "unmodified"), nextFile)
                    analysisFBs[i].nextUtt(soundSource)

                output = []

                # get mfccs from frontend
                print "next we try to obtain mfccs"
                mfccs = []
                for mfcc in frontend:
                    mfccs.append(mfcc)


                # obtain output file and write mfccs
                outFile = ("%s/%s.mfcc" %(outDataDir, utt.file))
                outDir = os.path.dirname(outFile)
                if (not os.path.exists(outDir)):
                    os.makedirs(outDir)
                mfccFile = fopen(outFile, 'w')
                fmatrixWrite(mfccs, mfccFile)
                mfccFile.close()

                #take powers from first channel
                nextFile = ("%s/%s.%s.adc.shn" %(inDataDir, utt.file, 1))
                safeIO(lambda x: powfront.attachFile(x), nextFile)

                # get powers from frontend
                powers = []
                for p in powfront:
                    powers.append(p)

                # obtain output file and write powers
                outFile = ("%s/%s.pow" %(outDataDir, utt.file))
                powFile = fopen(outFile, 'w')
                fmatrixWrite(powers, powFile)
                powFile.close()

            except IOError, x :
                print "\tIOError : " + `x` + ' on speaker ' + spkId

    print "\n\nMFCCs all written\n\n"


procUtts()
