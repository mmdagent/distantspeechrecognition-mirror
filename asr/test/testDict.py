from asr import common
from asr.dict import *
from asr.fsm import *

ps = PhonesSetPtr()
ps.read('/home/jmcd/data/verbmobil/english/S11/desc/phonesSet')

nm       = 'PhonemeTree'
fileName = '/home/jmcd/data/verbmobil/english/S11/clusterLH/desc/distribTree.2250p.gz'

pt = PhonemeTreePtr(nm = nm, phonesSet = ps)
pt.read(fileName)

# t = TagsPtr()
# t.read('/home/jmcd/data/verbmobil/english/S11/desc/tags') dict =
# DictionaryPtr(ps["PHONES"], t)
# dict.read('/home/jmcd/data/verbmobil/english/COMMON/dict/Dict_2003+.training')

