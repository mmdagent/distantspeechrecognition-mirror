from btk import sound, play

dataDir = "/home/jmcd/data/mic-array/esst-testset_shorten/cd28/e045a"

files = []
for i in range(1,17):
    files.append("%s/e045ach2_095.16.%s.adc.shn" %(dataDir, i))

snd = sound.SoundMicArray(4*6096, files, 0, -1)

play.playSoundSource(snd,3)

for sndBuffer in snd:
    sndBuffer.plot()
    raw_input("press key for next window")





