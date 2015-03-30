import sfe.common
from btk.sad import *
import Numeric

import pickle

# Input parameters
frameN		= 3000
dimN		= 39

# Output files
basisFileName	= 'basis.pickle'
whitenFileName	= 'whiten.pickle'

# Initialize matrices
input		= Numeric.zeros([frameN, dimN], Numeric.Float)
basis		= Numeric.zeros([dimN, dimN], Numeric.Float)
eigenVal	= Numeric.zeros(dimN, Numeric.Float)
whiten		= Numeric.zeros(dimN, Numeric.Float)

# Read input
# ROB: Need to read the input!

# Perform principal component analysis
pca             = PCAPtr(dimN = dimN)
pca.pca_svd(input, basis, eigenVal, whiten)

# Save results of principal component analysis
fp = open(basisFileName, 'w')
pickle.dump(basis, fp, 1)
fp.close()

fp = open(whitenFileName, 'w')
pickle.dump(whiten, fp, 1)
fp.close()
