from btk.digitalFilterBank import *
from btk import sound, play
from Numeric import *
import math

if __name__ == "__main__":

    fftLen      = 1024
    fftLen2     = fftLen/2
    nBlocks     = 32
    nBlocks2    = nBlocks/2
    subSampRate = 2

    snd = sound.SoundMicArray(fftLen, ("/home/jmcd/data/mic-array/esst-testset_shorten/cd28/e045a/e045ach2_095.16.1.adc.shn",))

    anal  = AnalysisFilterBank(fftLen,  nBlocks, subSampRate)
    synth = SynthesisFilterBank(fftLen, nBlocks, subSampRate)

    nextR  = -nBlocks2 * fftLen + fftLen / subSampRate
    nextX  = 0

    output = []

    cnt = 0
    for sample in snd:
        if len(sample[0]) != fftLen:
            break

        anal.nextSampleBlock(sample[0])

        for i in range(subSampRate):
            nextSpecSample = anal.getBlock(nextR)
            synth.nextSpectralBlock(nextSpecSample)

            if (nextR >= nBlocks2 * fftLen):
                nextSpeechSample = synth.getBlock(nextX)
                for j in range(fftLen2):
                    output.append(nextSpeechSample[j])
               	nextX += fftLen / subSampRate

            nextR += fftLen / subSampRate

    outArray = array(output, "i")

    play.playData(outArray, 16000)

