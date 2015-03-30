import Numeric
import pickle

from btk.modulated import *

def design():
    M =   256
    m =   2
    r =   1
    v =   0.01

    # Create and write analysis prototype
    analysis = AnalysisOversampledDFTDesignPtr(M = M, m = m, r = r, wpFactor = M)
    h = analysis.design()
    print 'h = '
    # print h

    analysisFileName = 'prototypes/h-M=%d-m=%d-r=%d.txt' %(M, m, r)
    
    fp = open(analysisFileName, 'w')
    pickle.dump(h, fp, True)
    fp.close()

    # Create and write synthesis prototype
    synthesis = SynthesisOversampledDFTDesignPtr(h, M = M, m = m, r = r, v = v, wpFactor = M)
    g = synthesis.design()
    print 'g = '
    # print g

    synthesisFileName = 'prototypes/g-M=%d-m=%d-r=%d.txt' %(M, m, r)
    fp = open(synthesisFileName, 'w')
    pickle.dump(g, fp, True)
    fp.close()

design()
