from asr.common   import *
from asr.feature  import *
from asr.gaussian import *
from asr.adapt    import *
from asr.matrix   import *
from asr.dict     import *

ps = PhonesSetPtr()
ps.read('/home/jmcd/data/verbmobil/english/S11power/desc/phonesSet')
t = TagsPtr()
t.read('/home/jmcd/data/verbmobil/english/S11power/desc/tags')
dict = DictionaryPtr(ps["PHONES"], t)
dict.read('/home/jmcd/data/verbmobil/english/COMMON/dict/Dict_2003+.training')
mdsP = ModelSetPtr()
tree = TreePtr(ps["PHONES"], ps, t, mdsP)
tms = TmSetPtr('/home/jmcd/data/verbmobil/english/S11power/desc/tmSet')
# tree.read('/home/jmcd/data/verbmobil/english/S11/clusterLH/desc/distribTree.2250p.gz')
