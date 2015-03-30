from btk.digitalFilterBank import *
from Numeric import *
import math

if __name__ == "__main__":

    fftLen      = 512
    fftLen2     = fftLen/2
    nBlocks     = 32
    nBlocks2    = nBlocks/2
    subSampRate = 2
    sample      = zeros(fftLen, "d")

    anal  = AnalysisFilterBank(fftLen,  nBlocks, subSampRate)
    synth = SynthesisFilterBank(fftLen, nBlocks, subSampRate)

    nextR  = -nBlocks2 * fftLen + fftLen / subSampRate;
    nextX  = 0;
    
    cnt = 0
    for i in range(40):
        for j in range(fftLen):
            sample[j] = cnt % 50 # + cos((2.0 * math.pi * cnt) / (fftLen/4))
            cnt += 1
        anal.nextSampleBlock(sample)

        for i in range(subSampRate):
            nextSpecSample = anal.getBlock(nextR)
            synth.nextSpectralBlock(nextSpecSample)

            if (nextR >= nBlocks2 * fftLen):
                nextSpeechSample = synth.getBlock(nextX)
                print "Resynthesized Speech Samples:"
                for j in range(fftLen2):
                    print "  %10.4f" % nextSpeechSample[j]
               	nextX += fftLen / subSampRate;

            nextR += fftLen / subSampRate
