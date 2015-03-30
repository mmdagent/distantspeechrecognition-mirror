from common import *
from fsm import *
from feature import *
from codebook import *
from decode import *

fsmInFile = 'test.txt'
fsmOutFile = 'test-out.txt'
wfst = WFSTDecodePtr()
wfst.read(fsmInFile)
wfst.write(fsmOutFile)

cbkDescFile = '/home/jmcd/BeamformingExpts/aux-models01a/clusterLH/desc/codebookSet.2250.gz'
cbkFile  = '/home/jmcd/BeamformingExpts/aux-models01a/train/Weights/5i.cbs.gz'
cbkNamesFile = '/home/jmcd/src/janus6/test/cbkNames/e057ach2_087_WJH.path'
fs = FeatureSet()
cbs = CodebookSetBasicPtr(descFile = cbkDescFile, fs = fs)
cbs.load(cbkFile)

dstDescFile = '/home/jmcd/BeamformingExpts/aux-models01a/clusterLH/desc/distribSet.2250p.gz'
distFile  = '/home/jmcd/BeamformingExpts/aux-models01a/train/Weights/5i.dss.gz'
dss = DistribSetBasicPtr(descFile = dstDescFile, cbs = cbs)
dss.load(distFile)

decode = DecoderPtr(wfst, dss)
