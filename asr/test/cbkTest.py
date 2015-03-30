from asr.common import *
from asr.feature import *
from asr.gaussian import *

cbkDescFile = '/home/jmcd/BeamformingExpts/aux-models01a/clusterLH/desc/codebookSet.2250.gz'
cbkFile  = '/home/jmcd/BeamformingExpts/aux-models01a/train/Weights/5i.cbs.gz'
cbkNamesFile = '/home/jmcd/src/janus6/test/cbkNames/e057ach2_087_WJH.path'
fs = FeatureSetPtr()
cbs = CodebookSetBasicPtr(descFile = cbkDescFile, fs = fs)
cbs.load(cbkFile)

dstDescFile = '/home/jmcd/BeamformingExpts/aux-models01a/clusterLH/desc/distribSet.2250p.gz'
distFile  = '/home/jmcd/BeamformingExpts/aux-models01a/train/Weights/5i.dss.gz'
dss = DistribSetBasicPtr(descFile = dstDescFile, cbs = cbs)
dss.load(distFile)

from asr.path import *
cbpath = CodebookPathPtr(cbs)
cbpath.read(cbkNamesFile)

for cb in cbpath:
    print cb.name()
print ''

for cb in cbs:
    print cb.name()
print ''

cb = cbs.find('IH-b(13)')
refX = 0
print cb.name()
print 'Mean Vector:'
for compX in range(cb.featLen()):
    print cb.mean(refX,compX)

print ''
print 'Inverse Diagonal Covariances:'
for compX in range(cb.featLen()):
    print cb.invCov(refX,compX)
