//
//			         Millennium
//                    Automatic Speech Recognition System
//                                  (asr)
//
//  Module:  asr.train
//  Purpose: Speaker adaptation parameter estimation and conventional
//           ML and discriminative HMM training with state clustering.
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

#include <math.h>
#include "train/codebookTrain.h"
#include "train/distribTrain.h"
#include "common/mach_ind_io.h"


// ----- methods for class `CodebookTrain' -----
//
CodebookTrain::CodebookTrain(const String& nm, UnShrt rfN, UnShrt dmN, CovType cvTp,
			     VectorFloatFeatureStreamPtr feat)
  : CodebookAdapt(nm, rfN, dmN, cvTp, feat),
    _accu(NULL)
{
  setScoreAll();
}

CodebookTrain::CodebookTrain(const String& nm, UnShrt rfN, UnShrt dmN, CovType cvTp,
			     const String& featureName)
  : CodebookAdapt(nm, rfN, dmN, cvTp, featureName),
    _accu(NULL)
{
  setScoreAll();
}

CodebookTrain::~CodebookTrain() { }

void CodebookTrain::update(bool verbose)
{
  accu()->update(_count, _rv, _cv);

  if (verbose) {
    unsigned subX = 0;
    double counts = 0.0;
    for (unsigned refX = 0; refX < _refN; refX++)
      counts += _count[refX];
    printf("Updated codebook %s : Total Counts %f\n", name().c_str(), counts);
  }
}

void CodebookTrain::updateMMI(double E)
{
  accu()->updateMMI(_count, _rv, _cv, E);
}

void CodebookTrain::allocAccu()
{
  _accu = new Accu(/*subN=*/ 1, _dimN, _nSubFeat, _refN, _covType);
}

void CodebookTrain::saveAccu(FILE* fp, bool onlyCountsFlag) const
{
  if (accu().isNull())
    throw jio_error("Accumulator for codebook %s is NULL.", name().c_str());

  accu()->save(fp, name(), onlyCountsFlag);
}

void CodebookTrain::loadAccu(FILE* fp, float factor)
{
  if (accu().isNull())
    throw jio_error("Accumulator for codebook %s is NULL.", name().c_str());

  accu()->load(fp, factor);
}

float CodebookTrain::accumulate(float factor, const float* dsVal, int frameX, float* mixVal)
{
  if (accu().isNull())
    throw jconsistency_error("Must allocate codebook accumulators before accumulating FB statistics.");

  float lhood = _scoreAll(frameX, dsVal, mixVal);

  if (isnan(lhood))
    throw jnumeric_error("Utterance log-likelihood is NaN");

  // normalize likelihoods to obtain posterior probabilities
  double sum = 0.0;
  for (unsigned refX = 0; refX < _refN; refX++)
    sum += mixVal[refX];
  for (unsigned refX = 0; refX < _refN; refX++)
    mixVal[refX] /= sum;

  const gsl_vector_float* pattern = _feature->next(frameX);

  accu()->accumulate(pattern->data, factor, mixVal);

  return lhood;
}

void CodebookTrain::zeroAccu()
{
  if (accu().isNull())
    throw jconsistency_error("Codebook accumulator is NULL.");

  accu()->zero();
}

unsigned CodebookTrain::split(float minCount, float splitFactor)
{
  unsigned maxIdx = 0;
  float maxCount  = _count[0];
  for (unsigned refX = 1; refX < refN(); refX++) {
    if (_count[refX] > maxCount) {
      maxCount = _count[refX];
      maxIdx   = refX;
    }
  }

  printf("Codebook %s: Max count (%f) for Gaussian %d\n.", _name.c_str(), maxCount, maxIdx);

  if (minCount > 0.0 && maxCount < minCount)
    throw jparameter_error("Cannot split codebook %s; maxCount (%f) < minCount (%f)\n.",
			   _name.c_str(), maxCount, minCount);

  _split(maxIdx, splitFactor);

  return maxIdx;
}

void CodebookTrain::_split(unsigned refX, float splitFactor)
{
  // printf("Splitting codebook %s\n", _name.c_str());

  // split mean vector
  gsl_matrix_float* oldRV = _rv;
  _rv = gsl_matrix_float_alloc(_refN + 1, _orgDimN);
  memcpy(_rv->data, oldRV->data, sizeof(float) * oldRV->size2 * oldRV->size1);
  for (unsigned dimX = 0; dimX < _orgDimN; dimX++) {
    float perturb = splitFactor / sqrt(gsl_vector_float_get(_cv[refX], dimX));
    gsl_matrix_float_set(_rv, _refN, dimX, gsl_matrix_float_get(oldRV, refX, dimX) - perturb);
    gsl_matrix_float_set(_rv,  refX, dimX, gsl_matrix_float_get(oldRV, refX, dimX) + perturb);
  }
  gsl_matrix_float_free(oldRV);

  // copy covariance
  gsl_vector_float** oldCV = _cv;
  _cv = new gsl_vector_float*[_refN + 1];
  memcpy(_cv, oldCV, _refN * sizeof(gsl_vector_float*));
  _cv[_refN] = gsl_vector_float_alloc(_dimN);
  gsl_vector_float_memcpy(_cv[_refN], _cv[refX]);
  delete[] oldCV;

  // copy determinant
  float* oldDet = _determinant;
  _determinant = new float[_refN + 1];
  memcpy(_determinant, oldDet, _refN * sizeof(float));
  _determinant[_refN] = _determinant[refX];
  delete[] oldDet;

  // split '_count'
  float* oldCount = _count;
  _count = new float[_refN + 1];
  memcpy(_count, oldCount, _refN * sizeof(float));
  _count[_refN]  = _count[refX] / 2.0;
  _count[refX]  /= 2.0;
  delete[] oldCount;

  // reallocate '_accu'
  if (_accu.isNull() == false) allocAccu();

  // deallocate old _regClass' and '_descLen'
  if (_regClass) {
    for (UnShrt i = 0; i < _refN; i++)
      free(_regClass[i]);
    delete[] _regClass; _regClass = NULL;
  }
  if (_descLen) {
    for (UnShrt i = 0; i < _refN; i++)
      free(_descLen[i]);
    delete[] _descLen;  _descLen  = NULL;
  }

  _refN++;
}

void CodebookTrain::fixGConsts()
{
  for (unsigned refX = 0; refX < _refN; refX++) {
    double logDet = 0.0;
    for (unsigned dimX = 0; dimX < featLen(); dimX++) {
      logDet += log(gsl_vector_float_get(_cv[refX], dimX));

      if (isnan(logDet)) {
	for (int i = 0; i < featLen(); i++)
	  cout << i << "  :  " << gsl_vector_float_get(_cv[refX], dimX) << endl;
	throw jconsistency_error("Determinant is NaN");
      }

      _determinant[refX] = logDet;
    }
  }
}

void CodebookTrain::invertVariance()
{
  for (unsigned refX = 0; refX < _refN; refX++)
    for (unsigned dimX = 0; dimX < featLen(); dimX++)
      gsl_vector_float_set(_cv[refX], dimX, 1.0 / gsl_vector_float_get(_cv[refX], dimX));
}


// ----- methods for class `CodebookTrain::GaussDensity' -----
//
void CodebookTrain::GaussDensity::fixGConst()
{
  double logDet = 0.0;
  for (int i = 0; i < featLen(); i++)
    logDet += log(var(i));

  if (isnan(logDet)) {
    for (int i = 0; i < featLen(); i++)
      cout << i << "  :  " << var(i) << endl;
    throw jconsistency_error("Determinant is NaN");
  }

  cbk()->_determinant[refX()] = logDet;
}


// ----- methods for class `CodebookTrain::Accu' -----
//
CodebookTrain::Accu::Accu(UnShrt sbN, UnShrt dmN, UnShrt subFeatN, UnShrt rfN, CovType cvTyp, UnShrt odmN)
  : _subN(sbN), _dimN(dmN), _subFeatN(subFeatN), _refN(rfN), _covType(cvTyp), _subX(0)
{
  if (odmN == 0) odmN = dmN;

  _count    = new double*[_subN];
  _denCount = new double*[_subN];
  _rv       = new gsl_matrix*[_subN];
  _sumOsq   = new gsl_vector**[_subN];

  for (UnShrt subX = 0; subX < _subN; subX++) {
    _count[subX]    = new double[_refN];
    _denCount[subX] = new double[_refN];
    _rv[subX]       = gsl_matrix_alloc(_refN, odmN);
    _sumOsq[subX]   = new gsl_vector*[_refN];

    for (UnShrt refX = 0; refX < _refN; refX++)
      _sumOsq[subX][refX] = gsl_vector_alloc(_dimN);
  }
  zero();
}

CodebookTrain::Accu::~Accu()
{
  for (UnShrt subX = 0; subX < _subN; subX++) {
    delete[] _count[subX];
    delete[] _denCount[subX];

    gsl_matrix_free(_rv[subX]);

    for (UnShrt refX=0; refX < _refN; refX++)
      gsl_vector_free(_sumOsq[subX][refX]);

    delete[] _sumOsq[subX];
  }

  delete[] _count;
  delete[] _denCount;
  delete[] _rv;
  delete[] _sumOsq;
}

void CodebookTrain::Accu::zero()
{
  for (UnShrt subX = 0; subX < _subN; subX++) {
    gsl_matrix_set_zero(_rv[subX]);
    for (UnShrt refX = 0; refX < _refN; refX++) {
      _count[subX][refX] = _denCount[subX][refX] = 0.0;
      gsl_vector_set_zero(_sumOsq[subX][refX]);
    }
  }
}

void CodebookTrain::Accu::_saveInfo(const String& name, int countsOnlyFlag, FILE* fp)
{
  write_string(fp, name.chars());
  write_int(   fp, _refN);
  write_int(   fp, _dimN);
  write_int(   fp, _subN);
  write_int(   fp, _covType);
  write_int(   fp, countsOnlyFlag);
}

const int CodebookTrain::Accu::MarkerMagic = 123456789;

// dumpMarker: dump a marker into file fp
void CodebookTrain::Accu::_dumpMarker(FILE* fp)
{
  write_int(fp, MarkerMagic);
}

// checkMarker: check if file fp has a marker next
void CodebookTrain::Accu::_checkMarker(FILE* fp)
{
  int marker = read_int(fp);
  if (marker != MarkerMagic)
    throw jio_error("CheckMarker: Marker expected in accu file (%d vs. %d).",
		    marker, MarkerMagic);
}

void CodebookTrain::Accu::_save(bool onlyCountsFlag, FILE* fp)
{
  for (UnShrt subX = 0; subX < _subN; subX++) {
    for (UnShrt refX = 0; refX < _refN; refX++) {
      write_float(fp, (float) _count[subX][refX]);
      write_float(fp, (float) _denCount[subX][refX]);

      if (onlyCountsFlag) continue;

      for (UnShrt dimX = 0; dimX < _rv[subX]->size2; dimX++)
	write_float(fp, (float) gsl_matrix_get(_rv[subX], refX, dimX));
      _saveCovariance(subX, refX, fp);
    }
  }
}

void CodebookTrain::Accu::_saveCovariance(unsigned subX, unsigned refX, FILE* fp)
{
  for (unsigned i = 0; i < _dimN; i++)
    write_float(fp, gsl_vector_get(_sumOsq[subX][refX], i));
}

void CodebookTrain::Accu::_loadCovariance(unsigned subX, unsigned refX, FILE* fp, float factor)
{
  for (unsigned i = 0; i < _dimN; i++)
    gsl_vector_set(_sumOsq[subX][refX], i, gsl_vector_get(_sumOsq[subX][refX], i) + factor * read_float(fp));
}

void CodebookTrain::Accu::_addCovariance(unsigned subX, unsigned refX, const float* pattern, float factor)
{
  for (unsigned i = 0; i < _dimN; i++)
    gsl_vector_set(_sumOsq[subX][refX], i, gsl_vector_get(_sumOsq[subX][refX], i) + factor * pattern[i] * pattern[i]);
}

void CodebookTrain::Accu::save(FILE* fp, const String& name, bool onlyCountsFlag)
{
  _saveInfo(name, onlyCountsFlag, fp);
  _save(onlyCountsFlag, fp);
  _dumpMarker(fp);
}

static const UnShrt MaxNameSize = 100;

bool CodebookTrain::Accu::_loadInfo(FILE* fp)
{
  static char name[MaxNameSize];

  read_string(fp, name);
  UnShrt refN         = read_int(fp);
  UnShrt dimN         = read_int(fp);
  UnShrt subN         = read_int(fp);
  UnShrt type         = read_int(fp);
  bool countsOnlyFlag = read_int(fp);

  if (refN != _refN)
    throw jdimension_error("'refN' does not match (%d vs. %d)", refN, _refN);
  if (dimN != _dimN)
    throw jdimension_error("'dimN' does not match (%d vs. %d)", dimN, _dimN);
  if (subN != _subN)
    throw jdimension_error("'subN' does not match (%d vs. %d)", subN, _subN);
  if (type != (int) _covType)
    throw jdimension_error("'type' does not match (%d vs. %d)", type, _covType);

  return countsOnlyFlag;
}

void CodebookTrain::Accu::add(const AccuPtr& ac, double factor)
{
  for (UnShrt subX = 0; subX < _subN; subX++) {
    for (UnShrt refX = 0; refX < _refN; refX++) {
      _count[subX][refX]    +=  factor * ac->_count[subX][refX];
      _denCount[subX][refX] +=  factor * ac->_denCount[subX][refX];

      for (unsigned dimX = 0; dimX < _dimN; dimX++) {
	gsl_matrix_set(_rv[subX], refX, dimX,  gsl_matrix_get(_rv[subX], refX, dimX) + factor * gsl_matrix_get(ac->_rv[subX], refX, dimX));
	gsl_vector_set(_sumOsq[subX][refX], dimX,  gsl_vector_get(_sumOsq[subX][refX], dimX) + factor * gsl_vector_get(ac->_sumOsq[subX][refX], dimX));
      }
    }
  }
}

void CodebookTrain::Accu::_load(FILE* fp, float factor, bool onlyCountsFlag)
{
  for (UnShrt subX = 0; subX < _subN; subX++) {
    for (UnShrt refX = 0; refX < _refN; refX++) {
      float count = read_float(fp);
      _count[subX][refX] += factor * count;

      float denCount = read_float(fp);
      _denCount[subX][refX] += factor * denCount;

      if (onlyCountsFlag) continue;

      for (UnShrt dimX = 0; dimX < _rv[subX]->size2; dimX++) {
	float x = read_float(fp);
	gsl_matrix_set(_rv[subX], refX, dimX,  gsl_matrix_get(_rv[subX], refX, dimX) + factor * x);
      }
      _loadCovariance(subX, refX, fp, factor);
    }
  }
}

void CodebookTrain::Accu::load(FILE* fp, float factor)
{
  bool onlyCountsFlag = _loadInfo(fp);
  _load(fp, factor, onlyCountsFlag);
  _checkMarker(fp);
}

void CodebookTrain::Accu::update(float* count, gsl_matrix_float* rv, gsl_vector_float** cv)
{
  _updateCount(count);
  _updateMean(count, rv);
  _updateCovariance(count, rv, cv);
}

void CodebookTrain::Accu::updateMMI(float* count, gsl_matrix_float* rv, gsl_vector_float** cv, double E)
{
  _updateCount(count);
  _updateMMI(count, rv, cv, E);
}

void CodebookTrain::Accu::_updateCount(float* count)
{
  for (unsigned refX = 0; refX < _refN; refX++) {
    double sum = 0.0;
    for (unsigned subX = 0; subX < _subN; subX++)
      sum += _count[subX][refX] /* - _denCount[subX][refX] */;
    count[refX] = sum;
  }
}

double CodebookTrain::Accu::MinimumCount = 5.0;

void CodebookTrain::Accu::_updateMean(const float* count, gsl_matrix_float* rv)
{
  for (unsigned refX = 0; refX < _refN; refX++) {

    if (count[refX] <= MinimumCount) continue;

    for (unsigned dimX = 0; dimX < _dimN; dimX++) {
      double sum = 0.0;
      for (unsigned subX = 0; subX < _subN; subX++)
	sum += gsl_matrix_get(_rv[subX], refX, dimX);
      gsl_matrix_float_set(rv, refX, dimX, sum / count[refX]);
    }
  }
}

void CodebookTrain::Accu::_updateCovariance(const float* count, const gsl_matrix_float* rv, gsl_vector_float** cv)
{
  for (unsigned refX = 0; refX < _refN; refX++) {

    if (count[refX] <= MinimumCount) continue;

    for (unsigned dimX = 0; dimX < _dimN; dimX++) {
      double sumSq = 0.0;
      for (unsigned subX = 0; subX < _subN; subX++)
	sumSq += gsl_vector_get(_sumOsq[subX][refX], dimX);

      double mu = gsl_matrix_float_get(rv, refX, dimX);
      gsl_vector_float_set(cv[refX], dimX, (sumSq / count[refX]) - (mu * mu));
    }
  }
}

void CodebookTrain::Accu::_updateMMI(const float* count, gsl_matrix_float* rv, gsl_vector_float** cv, double E)
{
  for (unsigned refX = 0; refX < _refN; refX++) {

    if (count[refX] <= MinimumCount) {
      for (unsigned dimX = 0; dimX < _dimN; dimX++)
	gsl_vector_float_set(cv[refX], dimX, 1.0 / gsl_vector_float_get(cv[refX], dimX));
      continue;
    }

    double D = E * denProb(refX);

    for (unsigned dimX = 0; dimX < _dimN; dimX++) {

      float mu       = gsl_matrix_float_get(rv, refX, dimX);
      float sigma    = 1.0 / gsl_vector_float_get(cv[refX], dimX);

      float musum    = sumO(refX, dimX) + mu * D;
      float muhat    = musum / (numProb(refX) - denProb(refX) + D);

      float sigmasum = sumOsq(refX, dimX) + (sigma + mu * mu) * D;
      float sigmahat = sigmasum / (numProb(refX) - denProb(refX) + D) - muhat * muhat;

      gsl_matrix_float_set(rv, refX, dimX, muhat);  gsl_vector_float_set(cv[refX], dimX, sigmahat);
    }
  }
}

void CodebookTrain::Accu::accumulate(const float* pattern, float factor, const float* addCount)
{
  for (UnShrt refX = 0; refX < _refN; refX++) {
    float gamma =  addCount[refX] * factor;

    if (gamma == 0.0) continue;

    for (UnShrt subX = 0; subX < _subN; subX++) {

      if ( gamma > 0.0 )
	_count[subX][refX]    += gamma;
      else if ( gamma < 0.0 )
	_denCount[subX][refX] -= gamma;

      for (UnShrt dimX = 0; dimX < _dimN; dimX++)
 	gsl_matrix_set(_rv[subX], refX, dimX,  gsl_matrix_get(_rv[subX], refX, dimX) + pattern[dimX] * gamma);

      _addCovariance(subX, refX, pattern, gamma);
    }
  }
}

unsigned CodebookTrain::Accu::
_size(const String& name, UnShrt meanLen, UnShrt subN, bool onlyCountsFlag)
{
  unsigned sz  = _refN * sizeof(float);			// for accu->count
  sz	      += _refN * sizeof(float);			// for accu->denCount

  if (onlyCountsFlag == false) {
    sz   += _refN * sizeof(float) * meanLen;		// for accu->rv
    sz   += _refN * sizeof(float) * _dimN;		// for accu->sumOsq
  }

  sz     *= subN;					// for subN

  							// total up size of accumulator info
  sz     += sizeof(short) + strlen(name.chars()) + 1;	// for name string
  sz     += 5 * sizeof(int);        			// for refN, dimN, subN, type,
							// and countsOnlyFlag

  sz     += sizeof(int);        			// for the marker

  return sz;
}

unsigned CodebookTrain::Accu::size(const String& name, bool onlyCountsFlag)
{
  unsigned sz = _size(name, _dimN, _subN, onlyCountsFlag);

  // cout << "Accu " << name << " : size = " << sz << endl;

  return sz;
}

bool CodebookTrain::Accu::zeroOccupancy()
{
  for (UnShrt isub = 0; isub < _subN; isub++)
    for (UnShrt iref = 0; iref < _refN; iref++)
      if (_count[isub][iref] != 0.0 || _denCount[isub][iref] != 0.0) return false;

  return true;
}


// ----- methods for class `CodebookSetTrain' -----
//
CodebookSetTrain::CodebookSetTrain(const String& descFile, FeatureSetPtr fs, const String& cbkFile,
				   double massThreshhold)
  : CodebookSetAdapt(/* descFile= */ "", fs)
{
  CodebookTrain::Accu::MinimumCount = massThreshhold;

  if (descFile == "") return;  

  freadAdd(descFile, ';', this);

  if (cbkFile == "") return;

  load(cbkFile);
}

void CodebookSetTrain::_addCbk(const String& name, UnShrt rfN, UnShrt dmN, CovType cvTp,
			       const String& featureName)
{
  CodebookTrainPtr cbk(new CodebookTrain(name, rfN, dmN, cvTp, featureName));

  _cblist.add(name, cbk);
}

void CodebookSetTrain::_addCbk(const String& name, UnShrt rfN, UnShrt dmN, CovType cvTp,
			       VectorFloatFeatureStreamPtr feat)
{
  CodebookTrainPtr ptr(new CodebookTrain(name, rfN, dmN, cvTp, feat));

  _cblist.add(name, ptr);
}

void CodebookSetTrain::allocAccus()
{
  for (_ConstIterator itr(_cblist); itr.more(); itr++) {
    CodebookTrainPtr& cbk(Cast<CodebookTrainPtr>(*itr));
    cbk->allocAccu();
  }
}

void CodebookSetTrain::
saveAccus(const String& fileName, float totalPr, unsigned totalT, bool onlyCountsFlag) const
{
  FILE* fp = fileOpen(fileName, "w");

  cout << "Saving codebook accumulators to '" << fileName << "'." << endl;

  write_float(fp, totalPr);
  write_int(fp, totalT);

  AccMap accMap(*this, onlyCountsFlag);
  accMap.write(fp);

  for (_ConstIterator itr(_cblist); itr.more(); itr++) {
    CodebookTrainPtr& cbk(Cast<CodebookTrainPtr>(*itr));

    if (cbk->accu()->zeroOccupancy()) continue;

    cbk->saveAccu(fp, onlyCountsFlag);
  }

  fileClose(fileName, fp);
}

void CodebookSetTrain::
loadAccus(const String& fileName, float& totalPr, unsigned& totalT, float factor,
	  unsigned nParts, unsigned part)
{
  FILE* fp = fileOpen(fileName, "r");

  cout << "Loading codebook accumulators from '" << fileName << "'" << endl;

  totalPr  = read_float(fp);
  totalT   = read_int(fp);

  AccMap accMap(fp);
  long int startOfAccs = ftell(fp);

  for (_Iterator itr(_cblist, nParts, part); itr.more(); itr++) {
    CodebookTrainPtr& cbk(Cast<CodebookTrainPtr>(*itr));
    long int current = accMap.state(cbk->name());

    if (current == AccMap::NotPresent) continue;

    current += startOfAccs;

    if (current != ftell(fp))
      fseek(fp, current, SEEK_SET);

    cbk->loadAccu(fp, factor);
  }

  fileClose(fileName, fp);
}

void CodebookSetTrain::loadAccus(const String& fileName)
{
  float	   totalPr;
  unsigned totalT;

  loadAccus(fileName, totalPr, totalT);
}

void CodebookSetTrain::zeroAccus(unsigned nParts, unsigned part)
{
  for (_Iterator itr(_cblist, nParts, part); itr.more(); itr++) {
    CodebookTrainPtr& cbk(Cast<CodebookTrainPtr>(*itr));
    cbk->zeroAccu();
  }
}

void CodebookSetTrain::update(int nParts, int part, bool verbose)
{
  for (_Iterator itr(_cblist, nParts, part); itr.more(); itr++) {
    CodebookTrainPtr& cb(Cast<CodebookTrainPtr>(*itr));
    cb->update(verbose);
  }
}

void CodebookSetTrain::updateMMI(int nParts, int part, double E)
{
  for (_Iterator itr(_cblist, nParts, part); itr.more(); itr++) {
    CodebookTrainPtr& cb(Cast<CodebookTrainPtr>(*itr));
    cb->updateMMI(E);
  }
}

void CodebookSetTrain::invertVariances(unsigned  nParts, unsigned part)
{
  for (_Iterator itr(_cblist, nParts, part); itr.more(); itr++) {
    CodebookTrainPtr& cbk(Cast<CodebookTrainPtr>(*itr));

    cbk->invertVariance();
  }
}

void CodebookSetTrain::fixGConsts(unsigned nParts, unsigned part)
{
  for (_Iterator itr(_cblist, nParts, part); itr.more(); itr++) {
    CodebookTrainPtr& cbk(Cast<CodebookTrainPtr>(*itr));

    for (CodebookTrain::Iterator gitr(cbk); gitr.more(); gitr++) {
      gitr.mix().fixGConst();
    }
  }
}

const float CodebookTrain::gblVarFloor   = 1.0E-04;
const float CodebookTrain::floorRatio    = 0.1;
const int   CodebookTrain::MinFloorCount = 10;

#if 0

void CodebookTrain::floorVariances(unsigned& ttlVarComps, unsigned& ttlFlooredVarComps)
{
  double    ttlCnt = 0.0;

  float* floor = new float[featLen()];
  for (unsigned dimX = 0; dimX < featLen(); dimX++)
    floor[dimX] = 0.0;

  // determine variance floor for each feature component
  for (unsigned refX = 0; refX < refN(); refX++) {
    float cnt  = fastAccu()->_count[0][refX];
    ttlCnt    += cnt;
    for (int dimX = 0; dimX < featLen(); dimX++)
      floor[dimX] += cnt * _cv[refX]->m.d[dimX];
  }

  if (ttlCnt < MinFloorCount) {
    printf("Only %g total counts instead of at least %d - not flooring.\n",
	   ttlCnt, MinFloorCount);
    delete[] floor;
    return;
  }

  for (unsigned dimX = 0; dimX < featLen(); dimX++) {
    floor[dimX] *= (floorRatio / ttlCnt);
    if (floor[dimX] < gblVarFloor) {
      printf("Warning: Variance component %d has value %g -- resetting to %g\n",
	     dimX, floor[dimX], gblVarFloor);
      floor[dimX] = gblVarFloor;
    }
  }

  // now floor variances
  for (unsigned refX = 0; refX < refN(); refX++) {
    for (unsigned dimX = 0; dimX < featLen(); dimX++) {
      ttlVarComps++;
      if ( _cv[refX]->m.d[dimX] < floor[dimX]) {
	_cv[refX]->m.d[dimX] = floor[dimX];
	ttlFlooredVarComps++;
      }
    }
  }

  delete[] floor;
}

#else

void CodebookTrain::floorVariances(unsigned& ttlVarComps, unsigned& ttlFlooredVarComps)
{
  // now floor variances
  for (unsigned refX = 0; refX < refN(); refX++) {
    for (unsigned dimX = 0; dimX < featLen(); dimX++) {
      ttlVarComps++;
      if (gsl_vector_float_get(_cv[refX], dimX) < gblVarFloor) {
	gsl_vector_float_set(_cv[refX], dimX, gblVarFloor);
	ttlFlooredVarComps++;
      }
    }
  }
}

#endif

void CodebookSetTrain::floorVariances(unsigned nParts, unsigned part)
{
  unsigned ttlVarComps        = 0;
  unsigned ttlFlooredVarComps = 0;

  for (_Iterator itr(_cblist, nParts, part); itr.more(); itr++) {
    CodebookTrainPtr& cbk(Cast<CodebookTrainPtr>(*itr));
    cbk->floorVariances(ttlVarComps, ttlFlooredVarComps);
  }

  printf("Floored %d of %d total variance components.\n",
	 ttlFlooredVarComps, ttlVarComps);  fflush(stdout);
}
