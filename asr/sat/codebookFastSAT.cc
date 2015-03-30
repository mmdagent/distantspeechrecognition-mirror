//
//			         Millennium
//                    Distant Speech Recognition System
//                                  (dsr)
//
//  Module:  asr.sat
//  Purpose: Speaker-adapted ML and discriminative HMM training.
//  Author:  John McDonough
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or (at
// your option) any later version.
// 
// This program is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.


#include <gsl/gsl_blas.h>

#include <math.h>
#include <set>
#include "sat/codebookFastSAT.h"
#include "common/mach_ind_io.h"
#include "train/distribTrain.h"


// ----- methods for class `CodebookFastSAT' -----
//
CodebookFastSAT::CodebookFastSAT(const String& nm, UnShrt rfN, UnShrt dmN,
				 CovType cvTyp, VectorFloatFeatureStreamPtr feat)
  : CodebookTrain(nm, rfN, dmN, cvTyp, feat), _fastAccu(NULL) { }

CodebookFastSAT::CodebookFastSAT(const String& nm, UnShrt rfN, UnShrt dmN,
				 CovType cvTyp, const String& featureName)
  : CodebookTrain(nm, rfN, dmN, cvTyp, featureName), _fastAccu(NULL) { }

CodebookFastSAT::~CodebookFastSAT() { }

void CodebookFastSAT::zeroFastAccu()
{
  if (_fastAccu.isNull() == false)
    _fastAccu->zero();
}

void CodebookFastSAT::saveFastAccu(FILE* fp) const
{
  _fastAccu->save(fp, name());
}

void CodebookFastSAT::loadFastAccu(FILE* fp, float addFactor, const String& name)
{
  _fastAccu->load(fp, addFactor, name);
}

void CodebookFastSAT::reallocFastAccu(const TransformerTreePtr& tree, unsigned olen)
{
  typedef set<unsigned> RCSet;
  typedef RCSet::const_iterator RCSetIter;

  // determine the set of leaf nodes
  RCSet treeRCSet;
  bool onlyLeaves = true;
  for (TransformerTree::ConstIterator itr(tree, onlyLeaves); itr.more(); itr++)
    treeRCSet.insert(itr.regClass());

  // determine current set of regression classes for all Gaussians
  UnShrt subN = 0;
  for (UnShrt refX = 0; refX < refN(); refX++) {
    RCSet finalRCSet;

    RCSet mixRCSet;
    for (UnShrt clsX = 1; clsX <= _regClass[refX][0]; clsX++)
      mixRCSet.insert(_regClass[refX][clsX]);

    if (refN() == 1) {

      // single-mixture training: keep only current regression classes
      finalRCSet = mixRCSet;

    } else {

      // expurgate set of leaf nodes: to be included, a leaf must either
      // be in the current set of regression classes, or be a child of a
      // regression class in the current set
      for (RCSetIter titr = treeRCSet.begin(); titr != treeRCSet.end(); titr++) {
	unsigned tClass   = (*titr);
	for (RCSetIter mitr = mixRCSet.begin(); mitr != mixRCSet.end(); mitr++) {
	  unsigned mClass = (*mitr);
	  if (isAncestor(mClass, tClass))
	    finalRCSet.insert(tClass);
	}
      }
    }

    unsigned size = finalRCSet.size();
    if (size == 0)
      throw jconsistency_error("No valid regression classes.\n");

    UnShrt thisRegClass = _regClass[refX][1];
    _regClass[refX] = (UnShrt*) realloc(_regClass[refX], (size + 1) * sizeof(UnShrt));
    _regClass[refX][0] = size;

    UnShrt clsX = 0;
    for (RCSetIter itr = finalRCSet.begin(); itr != finalRCSet.end(); itr++)
      if ((*itr) == thisRegClass)
	_regClass[refX][++clsX] = thisRegClass;

    for (RCSetIter itr = finalRCSet.begin(); itr != finalRCSet.end(); itr++)
      if ((*itr) != thisRegClass)
	_regClass[refX][++clsX] = (*itr);

    if (size > subN) subN = size;
  }
  if (subN == 0)
    throw jconsistency_error("No valid regression classes.\n");

  // re-allocate final set of fast accumulators
  if (_fastAccu.isNull() || subN > _fastAccu->subN() || olen > _fastAccu->orgDimN()) {
    // cout << "Reallocating accumulators for codebook " << _name << endl;

    UnShrt nblks = nSubFeat();
    UnShrt dimN  = featLen();
    _fastAccu = new FastAccu(subN, dimN, _nSubFeat, olen, refN(), nblks, _covType);
  }
}


// ----- methods for class `CodebookFastSAT::FastAccu' -----
//
CodebookFastSAT::FastAccu::
FastAccu(UnShrt sbN, UnShrt dmN, UnShrt subFeatN, UnShrt odmN, UnShrt rfN, UnShrt nblk,
	 CovType cvTyp)
  : Accu(sbN, dmN, subFeatN, rfN, cvTyp, odmN), _orgDimN(odmN), _scalar(new double**[_subN]),
    _nblks(nblk)
{
  for (unsigned isub = 0; isub < _subN; isub++) {
    _scalar[isub] = new double*[_refN];
    for (unsigned refX = 0; refX < _refN; refX++) {
      _scalar[isub][refX] = new double[_nblks];
    }
  }
}

CodebookFastSAT::FastAccu::~FastAccu()
{
  if (_scalar == NULL) return;

  for (UnShrt subX = 0; subX < _subN; subX++) {
    for (UnShrt refX = 0; refX < _refN; refX++)
      delete[] _scalar[subX][refX];
    delete[] _scalar[subX];
  }
  delete[] _scalar;
}

void CodebookFastSAT::FastAccu::zero()
{
  Accu::zero();

  if (_scalar == NULL) return;

  for (UnShrt subX = 0; subX < _subN; subX++)
    for (UnShrt refX = 0; refX < _refN; refX++)
      for (UnShrt blkX = 0; blkX < _nblks; blkX++)
	_scalar[subX][refX][blkX] = 0.0;
}

void CodebookFastSAT::FastAccu::_saveInfo(const String& name, FILE* fp)
{
  bool countsOnlyFlag = false;
  Accu::_saveInfo(name, countsOnlyFlag, fp);

  write_int(fp, _orgDimN);
  write_int(fp, _nblks);
}

void CodebookFastSAT::FastAccu::_save(FILE* fp)
{
  bool countsOnlyFlag = false;
  Accu::_save(countsOnlyFlag, fp);

  if (_scalar == NULL) return;

  for (UnShrt subX =0; subX < _subN; subX++)
    for (UnShrt refX = 0; refX < _refN; refX++)
      for (UnShrt blkX = 0; blkX < _nblks; blkX++)
	write_float(fp, (float) _scalar[subX][refX][blkX]);
}

void CodebookFastSAT::FastAccu::save(FILE* fp, const String& name)
{
  _saveInfo(name, fp);
  _save(fp);
  _dumpMarker(fp);
}

bool CodebookFastSAT::FastAccu::_loadInfo(FILE* fp)
{
  bool countsOnlyFlag = Accu::_loadInfo(fp);

  if (countsOnlyFlag == true)
    throw jconsistency_error("Mean and variance statistics must be included in fast accumulators.");

  UnShrt orgDimN = read_int(fp);
  UnShrt nblks   = read_int(fp);

  if (orgDimN != _orgDimN)
    throw jdimension_error("'orgDimN' does not match (%d vs. %d)", orgDimN, _orgDimN);
  if (nblks != _nblks)
    throw jdimension_error("'nblks' does not match (%d vs. %d)", nblks, _nblks);

  return countsOnlyFlag;
}

void CodebookFastSAT::FastAccu::_load(FILE* fp, float addFactor, bool onlyCountsFlag)
{
  Accu::_load(fp, addFactor, onlyCountsFlag);

  if (_scalar == NULL) return;

  for (UnShrt subX = 0; subX < _subN; subX++)
    for (UnShrt refX = 0; refX < _refN; refX++)
      for (UnShrt blkX = 0; blkX < _nblks; blkX++)
	_scalar[subX][refX][blkX] += addFactor * read_float(fp);
}

void CodebookFastSAT::FastAccu::load(FILE* fp, float addFactor, const String& name)
{
  bool onlyCountsFlag = _loadInfo(fp);
  _load(fp, addFactor, onlyCountsFlag);
  _checkMarker(fp);
}

unsigned CodebookFastSAT::FastAccu::size(const String& name, bool onlyCountsFlag)
{
  unsigned sz =			// basic size of accumulator
    _size(name, _orgDimN, _subN, onlyCountsFlag);

  sz += sizeof(int);		// for '_orgDimN'
  sz += sizeof(int);		// for '_nblks'

  if (_scalar != NULL)
    sz +=			// add size of scalar[refN][nblks]
      (_subN * _refN * _nblks * sizeof(float));

  return sz;
}


// ----- methods for class `CodebookFastSAT::GaussDensity' -----
//
void CodebookFastSAT::GaussDensity::extendMean(int len)
{
  if (cbk()->_origRV != NULL) {
    gsl_matrix_float_free(cbk()->_rv); cbk()->_rv = cbk()->_origRV; cbk()->_origRV = NULL;
  }

  // allocate space for new means
  gsl_matrix_float* shortRV = cbk()->_rv;
  if (len == shortRV->size2) return;

  cbk()->_rv = gsl_matrix_float_alloc(cbk()->_refN, len);
  cbk()->_orgDimN = len;

  // copy over original values
  for (UnShrt refX = 0; refX < cbk()->_refN; refX++) {
    NaturalVector newMean(cbk()->_rv->data + refX * cbk()->_rv->size2, cbk()->_rv->size2, cbk()->nSubFeat());
    NaturalVector oldMean(shortRV->data + refX * shortRV->size2,    shortRV->size2,       cbk()->nSubFeat());
    newMean = oldMean;
  }

  gsl_matrix_float_free(shortRV);
}

gsl_matrix* CodebookFastSAT::GaussDensity::temp1             = NULL;
gsl_vector* CodebookFastSAT::GaussDensity::vec1              = NULL;
gsl_vector* CodebookFastSAT::GaussDensity::vec2              = NULL;

UnShrt	    CodebookFastSAT::GaussDensity::_blockLen         = 0;
UnShrt	    CodebookFastSAT::GaussDensity::_origBlockLen     = 0;
UnShrt	    CodebookFastSAT::GaussDensity::_nBlocks          = 0;
bool	    CodebookFastSAT::GaussDensity::_allocated        = false;
const int   CodebookFastSAT::GaussDensity::FastAccNotPresent = -1;

void CodebookFastSAT::GaussDensity::_checkBlocks(UnShrt nblk, int olen)
{
  _blockLen = featLen() / nblk;  _origBlockLen = olen / nblk;
  _nBlocks  = nblk;

  if (_allocated == false) { _allocWorkSpace(); return; }

  if (temp1->size1 != _blockLen || temp1->size2 != _origBlockLen) {
    _deallocWorkSpace(); _allocWorkSpace();
  }
}

void CodebookFastSAT::GaussDensity::_allocWorkSpace()
{
  printf("\nAllocating CodebookFastSAT::GaussDensity working space (%d x %d) ... ",
	 _blockLen, _origBlockLen);  fflush(stdout);

  temp1 = gsl_matrix_alloc(_blockLen, _origBlockLen);

  vec1  = gsl_vector_alloc(_blockLen);
  vec2  = gsl_vector_alloc(_origBlockLen);
  
  _allocated = true;

  printf("Done\n\n");
}

void CodebookFastSAT::GaussDensity::_deallocWorkSpace()
{
  printf("\nDe-allocating CodebookFastSAT::GaussDensity working space ... ");  fflush(stdout);

  gsl_matrix_free(temp1);  temp1 = NULL;
  gsl_vector_free(vec1);   vec1 = NULL;
  gsl_vector_free(vec2);   vec2 = NULL;
  
  _allocated = false;

  printf("Done\n\n");  fflush(stdout);
}

void CodebookFastSAT::GaussDensity::setRegClasses(const RCArray& rcsarray)
{
  int regClassN = rcsarray.size();
  cbk()->_regClass[refX()] =
    (UnShrt*) realloc(cbk()->_regClass[refX()],
		      (regClassN + 1) * sizeof(UnShrt));
  cbk()->_regClass[refX()][0] = regClassN;
  int clsX = 0;
  for (RCAIter itr = rcsarray.begin(); itr != rcsarray.end(); itr++)
    cbk()->_regClass[refX()][++clsX] = (*itr)._regClass;
}

const double CodebookFastSAT::GaussDensity::Small_ck = 1.0E-06;

void CodebookFastSAT::GaussDensity::normFastAccu(const TransformerTreePtr& tree, unsigned olen)
{
  if ((numProb() == 0.0) && (denProb() == 0.0)) return;

  double ck = postProb();
  double dk = denProb();

  if (olen == 0) olen = orgFeatLen();

  bool onlyLeaves = true;
  for (TransformerTree::ConstIterator itr(tree, onlyLeaves); itr.more(); itr++) {

    int accX = findFastAccX(itr.regClass());
    if (accX == FastAccNotPresent) continue;

    if (accX >= fastAccu()->subN())
      throw jindex_error("Accumulator index (%d) greater than number of accumulators (%d).",
			 accX, fastAccu()->subN());

    fastAccu()->addCount(accX, refX(), numProb());
    fastAccu()->addDenCount(accX, refX(), dk);

    if (fabs(ck) < Small_ck) continue;

    const TransformBasePtr& trans(itr.transformer());
    const TransformMatrix& transformMatrix(trans->transformMatrix());
    _checkBlocks(transformMatrix.nBlocks(), olen);

    static NaturalVector trnsMean;
    if (trnsMean.featLen() == 0)
      trnsMean.resize(trans->featLen(), trans->nSubFeat());
    trans->transform(origMean(), trnsMean);

    for (UnShrt nb = 0; nb < _nBlocks; nb++) {
      UnShrt            offsetIndex = nb * _blockLen;
      const gsl_matrix* xformMatrix = transformMatrix.matrix(nb);

      gsl_matrix_memcpy(temp1, xformMatrix);
      for (UnShrt m = 0; m < _blockLen; m++) {
	float inv = invVar(m + offsetIndex);
	for (UnShrt n = 0; n < _origBlockLen; n++)
	  gsl_matrix_set(temp1, m, n, gsl_matrix_get(temp1, m, n) * inv);
      }

      const gsl_vector* offset = transformMatrix.offset(nb);
      for (UnShrt m = 0; m < _blockLen; m++)
	gsl_vector_set(vec1, m, sumO(m + offsetIndex)
		       - (ck * gsl_vector_get(offset, m)));
      gsl_blas_dgemv(CblasTrans, 1.0, temp1, vec1, 0.0, vec2);

      UnShrt origOffsetIndex = nb * _origBlockLen;
      for (UnShrt m = 0; m < _origBlockLen; m++)
	gsl_matrix_set((gsl_matrix*) fastAccu()->rv()[accX], refX(), m + origOffsetIndex,
		       gsl_matrix_get(fastAccu()->rv()[accX], refX(), m + origOffsetIndex) + gsl_vector_get(vec2, m));

      gsl_vector* fastSumOsq = (gsl_vector*) fastAccu()->sumOsq()[accX][refX()];
      for (UnShrt m = 0; m < _blockLen; m++) {
	double vElem = gsl_vector_get(vec1, m);
	fastAccu()->_scalar[accX][refX()][nb] +=
	  invVar(m + offsetIndex) * vElem * vElem / ck;

	if (isnan(fastAccu()->_scalar[accX][refX()][nb]))
	  throw jnumeric_error("Scalar is NaN.");

	double trn = trnsMean(m + offsetIndex);
	double sqr = trn * trn;
	double val = gsl_vector_get(fastSumOsq, m + offsetIndex);
	gsl_vector_set(fastSumOsq, m + offsetIndex,
		       val + sumOsq(m + offsetIndex) - 2.0 * trn * sumO(m + offsetIndex) + ck * sqr);
      }
    }
  }
}

// Must return `int' to allow for `not present' case
//
int CodebookFastSAT::GaussDensity::findFastAccX(UnShrt rClass)
{
  UnShrt noRegClass = cbk()->_regClass[refX()][0];

  for (int i = 1; i <= noRegClass; i++)
    if (cbk()->_regClass[refX()][i] == rClass)
      return i - 1;			// found it

  return FastAccNotPresent;		// not present
}

void CodebookFastSAT::GaussDensity::fixGConsts()
{
  float logDet = 0.0;
  for (UnShrt i = 0; i < featLen(); i++)
    logDet += log(var(i));

  if (isnan(logDet)) {
    for (UnShrt i = 0; i < featLen(); i++)
      cout << i << "  :  " << var(i) << endl;
    throw jnumeric_error("Determinant is NaN");
  }

  cbk()->_determinant[refX()] = logDet;
}

void CodebookFastSAT::GaussDensity::setRegClass(UnShrt reg)
{
  if (cbk()->_regClass == NULL) {
    UnShrt refN = cbk()->_refN;
    cbk()->_regClass = (UnShrt**) malloc(refN * sizeof(UnShrt*));
    for (UnShrt i = 0; i < refN; i++) cbk()->_regClass[i] = NULL;
  }

  cbk()->_regClass[refX()] =
    (UnShrt*) realloc(cbk()->_regClass[refX()], 2 * sizeof(UnShrt));

  cbk()->_regClass[refX()][0] = 1;
  cbk()->_regClass[refX()][1] = reg;
}

void CodebookFastSAT::GaussDensity::replaceRegClass(UnShrt best, UnShrt nextBest)
{
  if (cbk()->_regClass == NULL)
    throw jconsistency_error("No regression class array.");

  UnShrt current = cbk()->_regClass[refX()][1];
  if (isAncestor(current, best)     == false ||
      isAncestor(current, nextBest) == false)
    throw jconsistency_error("Tried to replace %d with %d and %d.", current, best, nextBest);

  UnShrt nRegClasses = cbk()->_regClass[refX()][0] + 1;
  UnShrt* regClasses = cbk()->_regClass[refX()] =
    (UnShrt*) realloc(cbk()->_regClass[refX()], (nRegClasses+1) * sizeof(UnShrt));

  for(UnShrt i = nRegClasses; i >= 3; i--)
    regClasses[i] = regClasses[i-1];

  regClasses[0] = nRegClasses;
  regClasses[1] = best;
  regClasses[2] = nextBest;
}

void CodebookFastSAT::GaussDensity::replaceIndex(int bestIndex, int index1, int index2)
{
  if (cbk()->_regClass == NULL)
    throw jconsistency_error("No regression class array.");

  UnShrt* regClasses  = cbk()->_regClass[refX()];
  UnShrt  nRegClasses = regClasses[0];

  UnShrt indexBestIndex = 0;
  bool isPresBest = false, isPres1 = false, isPres2 = false;
  for (UnShrt i = 1; i <= nRegClasses; i++) {
    if ( regClasses[i] == bestIndex ) { isPresBest = true; indexBestIndex = i; }
    if ( regClasses[i] == index1 )      isPres1    = true;
    if ( regClasses[i] == index2 )      isPres2    = true;
  }

  if ( isPresBest == false ) return;

  int toAdd = 0;

  if ( isPres1 == false ) toAdd++;
  if ( isPres2 == false ) toAdd++;

  if ( toAdd == 0 )
    throw jconsistency_error("Regression class list already contains best (%d), as well as %d and %d.",
			     bestIndex, index1, index2);

  nRegClasses += toAdd;

  regClasses = cbk()->_regClass[refX()] =
    (UnShrt*) realloc(cbk()->_regClass[refX()], (nRegClasses+1) * sizeof(UnShrt));

  regClasses[indexBestIndex] = (isPres1 == false) ? index1 : index2;

  if ( toAdd == 2 )
    regClasses[nRegClasses] = index2;
}

UnShrt* CodebookFastSAT::GaussDensity::descLen()
{
  if ( cbk()->_descLen == NULL ) {
    UnShrt refN = cbk()->_refN;
    cbk()->_descLen = new UnShrt*[refN];
    for (UnShrt i = 0; i < refN; i++)
      cbk()->_descLen[i] = NULL;
  }

  if ( cbk()->_descLen[refX()] == NULL ) {
    cbk()->_descLen[refX()] = (UnShrt*) malloc(_nBlocks * sizeof(UnShrt));
    for (UnShrt j = 0; j < _nBlocks; j++)
      cbk()->_descLen[refX()][j] = 1;
  }

  return cbk()->_descLen[refX()];
}


// ----- method for class `AccMap' -----
//
AccMap::AccMap(const CodebookSetFastSAT& cb, bool fastAccusFlag, bool countsOnlyFlag)
{
  long int offset = 0;

  // determine locations of state accumulators
  for (CodebookSetBasic::_ConstIterator itr(cb._cblist); itr.more(); itr++) {
    CodebookFastSATPtr& cbk(Cast<CodebookFastSATPtr>(*itr));

    bool zeroOcc = fastAccusFlag ?
      cbk->fastAccu()->zeroOccupancy() : cbk->accu()->zeroOccupancy();

    if (zeroOcc) continue;

    if (_stateMap.find(cbk->name()) != _stateMap.end())
      throw jkey_error("Entry for %s is not unique.", cbk->name().c_str());

    _stateMap.insert(_ValueType(cbk->name(), offset));

    offset += fastAccusFlag ?
      cbk->fastAccu()->size(cbk->name(), countsOnlyFlag) :
      cbk->accu()->size(cbk->name(), countsOnlyFlag);
  }
}


// ----- methods for class `CodebookSetFastSAT' -----
//
CodebookSetFastSAT::CodebookSetFastSAT(const String& descFile, FeatureSetPtr& fs, const String& cbkFile)
  : CodebookSetTrain(/* descFile= */ "", fs), _zeroFastAccs(true)
{
  if (descFile == "") return;

  freadAdd(descFile, ';', this);

  if (cbkFile == "") return;

  load(cbkFile);
}

void CodebookSetFastSAT::_addCbk(const String& name, UnShrt rfN, UnShrt dmN, CovType cvTp,
				 const String& featureName)
{
  CodebookFastSATPtr cbk(new CodebookFastSAT(name, rfN, dmN, cvTp, featureName));

  _cblist.add(name, cbk);
}

void CodebookSetFastSAT::_addCbk(const String& name, UnShrt rfN, UnShrt dmN, CovType cvTp,
				 VectorFloatFeatureStreamPtr feat)
{
  _cblist.add(name, CodebookFastSATPtr(new CodebookFastSAT(name, rfN, dmN, cvTp, feat)));
}

void CodebookSetFastSAT::saveFastAccus(const String& fileName) const
{
  FILE* fp = fileOpen(fileName, "w");

  cout << "Saving fast SAT accumulators to '" << fileName << "'." << endl;

  bool fastAccusFlag  = true;
  bool countsOnlyFlag = false;
  AccMap accMap(*this, fastAccusFlag, countsOnlyFlag);
  accMap.write(fp);

  for (_ConstIterator itr(_cblist); itr.more(); itr++) {
    CodebookFastSATPtr& cbk(Cast<CodebookFastSATPtr>(*itr));

    if (cbk->fastAccu()->zeroOccupancy()) continue;

    cbk->saveFastAccu(fp);
  }

  fileClose(fileName, fp);
}

void CodebookSetFastSAT::
loadFastAccus(const String& fileName, unsigned nParts, unsigned part)
{
  FILE* fp = fileOpen(fileName, "r");

  cout << "Loading fast SAT accumulators from '" << fileName << "'" << endl;

  AccMap accMap(fp);
  long int startOfAccs = ftell(fp);

  for (_Iterator itr(_cblist, nParts, part); itr.more(); itr++) {
    CodebookFastSATPtr& cbk(Cast<CodebookFastSATPtr>(*itr));
    long int current = accMap.state(cbk->name());

    if (current == AccMap::NotPresent) continue;

    current += startOfAccs;

    if (current != ftell(fp))
      fseek(fp, current, SEEK_SET);

    cbk->loadFastAccu(fp, 1.0, cbk->name());
  }

  fileClose(fileName, fp);
}

void CodebookSetFastSAT::zeroFastAccus(unsigned nParts, unsigned part)
{
  for (_Iterator itr(_cblist, nParts, part); itr.more(); itr++) {
    CodebookFastSATPtr& cbk(Cast<CodebookFastSATPtr>(*itr));
    cbk->zeroFastAccu();
  }
}

unsigned CodebookSetFastSAT::descLength()
{
  unsigned dl = 0;

  unsigned ttlGaussComps = 0;
  for (_ConstIterator itr(_cblist); itr.more(); itr++) {
    CodebookFastSATPtr& cbk(Cast<CodebookFastSATPtr>(*itr));
    for (CodebookFastSAT::Iterator gitr(cbk); gitr.more(); gitr++) {
      ttlGaussComps++;
      UnShrt* dla = gitr.mix().descLen();
      for (UnShrt isub = 0; isub < nSubFeat(); isub++)
	dl += dla[isub];
    }
  }

  double avgDescLength = ((double) dl) / ((double) ttlGaussComps);
  cout << "Global average description length = " << avgDescLength << endl;

  return dl;
}

// Note: we assume here all transformation trees have
// exactly the same set of regression classes
void CodebookSetFastSAT::normFastAccus(const TransformerTreePtr& transTree, unsigned olen)
{
  if (olen == 0) olen = orgFeatLen();

  printf("Normalizing fast SAT accus ... ");  fflush(stdout);

  if(olen % cepNSubFeat() != 0)
    throw jdimension_error("Cannot have original feature length of %d and %d sub-features.",
			   olen, cepNSubFeat());

  setRegClassesToOne();
  if (_zeroFastAccs == true) {
    for (_ConstIterator itr(_cblist); itr.more(); itr++) {
      CodebookFastSATPtr& cbk(Cast<CodebookFastSATPtr>(*itr));
      cbk->reallocFastAccu(transTree, olen);
      cbk->zeroFastAccu();
    }
    _zeroFastAccs = false;
  }

  for (_ConstIterator itr(_cblist); itr.more(); itr++) {
    CodebookFastSATPtr& cbk(Cast<CodebookFastSATPtr>(*itr));
    for (CodebookFastSAT::Iterator gitr(cbk); gitr.more(); gitr++) {
      gitr.mix().normFastAccu(transTree, olen);
    }
  }

  printf("Done.\n");  fflush(stdout);
}

// This really should be eliminated!
void CodebookSetFastSAT::setRegClassesToOne() 
{
  for (_ConstIterator itr(_cblist); itr.more(); itr++) {
    CodebookFastSATPtr& cbk(Cast<CodebookFastSATPtr>(*itr));
    for (CodebookFastSAT::Iterator gitr(cbk); gitr.more(); gitr++) {
      if (gitr.mix().regClass() == 0)
	gitr.mix().setRegClass();
    }
  }
}

void replaceIndex(CodebookSetFastSATPtr& cbs, int bestIndex, int index1, int index2)
{
  for (CodebookSetFastSAT::GaussianIterator itr(cbs); itr.more(); itr++)
    itr.mix().replaceIndex(bestIndex, index1, index2);
}
