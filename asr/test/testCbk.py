from asr import common
from asr.feature import *
from asr.gaussian import *
from asr.adapt import *
from asr.matrix import *

cbkDescFile  = "/home/jmcd/Expts/2220/codebookSet.48.gz"
distDescFile = "/project/verbmobil/english/S11/clusterLH/desc/distribSet.2250p.gz"
cbkFile  = "/home/jmcd/Expts/2358/Weights/7300i.cbs.gz"
distFile = "/home/jmcd/Expts/2358/Weights/7300i.dss.gz"

spkParam = "/home/jmcd/Expts/2282/param2/param"
spkLabel = "CLW_e045ach1"

fs = FeatureSetPtr()
cbs = CodebookSetAdaptPtr(descFile = cbkDescFile, fs = fs)
dss = DistribSetBasicPtr(cbs = cbs, descFile  = distDescFile)
# pt = ParamTreePtr(spkParam, spkLabel)

# cbs.load(cbkFile)
# dss.load(distFile)

