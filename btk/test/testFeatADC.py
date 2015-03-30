from btk import sound, _featureADC

#filename = "data/snd1/r625c002.16.shn"
filename = "data/sndarray/e029ach2_061.16.15.adc.shn"
samplingRate = 0.0
byteOrder = "shorten"
header = "auto"
force = 1
chX = 1
chN = 1
cfrom = 0
cto = -1
force = 0

snddata = sound.SoundBuffer(filename, header, byteOrder, chX, chN, cfrom, cto, force)

print "Samplingrate: %s" % _featureADC.getADCsamplRate()[1]
print "byteemode = %s" % _featureADC.getADCbyteMode()
#snddata.plot(40000,60000)
snddata.play()





