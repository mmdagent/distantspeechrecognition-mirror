from btk.dbase import *
import string

db = Dbase("clean-spk", "clean-utt")

speaker = 'g040bk_THO'
prefix = "data/db1"

# iterate over all utterances of 'g040bk_THO'
for utt in db.getSpeaker(speaker).utt:
    print string.join(utt.data["TRL"])
    feature = utt.getFeature(prefix)
    feature.plot(300, 800)
#    raw_input("Juhu")
    feature.play()
