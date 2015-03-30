//
//                               Millennium
//                    Distant Speech Recognition System
//                                  (dsr)
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


#include "train/fsa.h"
#include "common/mach_ind_io.h"


// ----- methods for class `FeatureSpaceAdaptationFeature' -----
//
FeatureSpaceAdaptationFeature::
FeatureSpaceAdaptationFeature(VectorFloatFeatureStreamPtr& src,
			      unsigned maxAccu, unsigned maxTran, const String& nm)
  : VectorFloatFeatureStream(src->size(), nm), _src(src),
    _beta(new float[_maxAccu]), _count(new double[_maxAccu]),
    _z(new gsl_matrix*[_maxAccu]), _Gi(new gsl_matrix**[_maxAccu]), _w(new gsl_matrix*[_maxTran]),
    _dsXA(NULL), _dssN(0), _dimN(0), _featX(0), _addCount(NULL), _gammaX(NULL),
    _maxAccu(maxAccu), _maxTran(maxTran),
    _U(NULL), _V(NULL), _singularVals(NULL), _d(NULL), _p(NULL),
    _workspace(NULL), _permutation(NULL) { }

void FeatureSpaceAdaptationFeature::_initialize()
{
  unsigned refN = 0;
  _dimN = 0;
  _dssN = _dss->ndists();
  for (DistribSetBasic::ConstIterator itr(_dss); itr.more(); itr++) {
    const DistribBasicPtr& ptr(itr.dst());
    if (ptr->cbk()->refN() > refN) {
      refN  = ptr->cbk()->refN();
      _dimN = ptr->cbk()->featLen();
    }
  }

  _topN     =  1;
  _shift    =  1.0;

  _dsXA     = new bool[_dssN];
  _addCount = new float[refN];
  _gammaX   = new SortElement[refN];

  for (unsigned i = 0; i < _dssN; i++)
    _dsXA[i] = false;

  for (unsigned accuX = 0; accuX < _maxAccu; accuX++) {
    _count[accuX] =  0;
    _beta[accuX]  =  0.0;
  }

  for (unsigned tranX = 0; tranX < _maxTran; tranX++)
    _w[tranX] = gsl_matrix_alloc(_dimN, _dimN + 1);

  for (unsigned accuX = 0; accuX < _maxAccu; accuX++) {
    _z[accuX] = gsl_matrix_alloc(_dimN, _dimN + 1);
    _Gi[accuX] = new gsl_matrix*[_dimN];

    for (unsigned i = 0; i < _dimN; i++)
      _Gi[accuX][i] = gsl_matrix_alloc(_dimN + 1, _dimN + 1);
  }

  // set _w to identity
  for (unsigned tranX = 0; tranX < _maxTran; tranX++) {
    for (unsigned i=0; i < _dimN; i++) {
      gsl_matrix_set(_w[tranX], i, 0, 0.0);
      for (unsigned j = 0; j < _dimN;j++) {
	gsl_matrix_set(_w[tranX], i, j+1, ((i==j) ? 1.0 : 0.0));
      }
    }
  }

  // allocate working space for feature space transformation estimation
  _U            = new gsl_matrix*[_dimN];
  _V            = new gsl_matrix*[_dimN];
  _singularVals = new gsl_vector*[_dimN];
  for (unsigned i = 0; i < _dimN; i++) {
    _U[i]            = gsl_matrix_alloc(_dimN + 1, _dimN + 1);
    _V[i]            = gsl_matrix_alloc(_dimN + 1, _dimN + 1);
    _singularVals[i] = gsl_vector_alloc(_dimN + 1);
  }

  _d	         = gsl_vector_alloc(_dimN + 1);
  _p             = gsl_vector_alloc(_dimN + 1);
  _workspace     = gsl_vector_alloc(_dimN + 1);
  _workspaceCof  = gsl_vector_alloc(_dimN);

  _permutation   = gsl_permutation_alloc(_dimN);
  _cofactorU     = gsl_matrix_alloc(_dimN, _dimN);
  _cofactorV     = gsl_matrix_alloc(_dimN, _dimN);
  _cofactorS     = gsl_vector_alloc(_dimN);
  _inverseMatrix = gsl_matrix_alloc(_dimN, _dimN);
}

FeatureSpaceAdaptationFeature::~FeatureSpaceAdaptationFeature()
{ 
  for (unsigned tranX = 0; tranX < _maxTran;tranX++)
    if (_w[tranX]) gsl_matrix_free(_w[tranX]);

  for (unsigned accuX = 0; accuX < _maxAccu; accuX++) {
    if (_z[accuX])  gsl_matrix_free(_z[accuX]);
    if (_Gi[accuX]) {
      for (unsigned i = 0; i < _dimN; i++)
	gsl_matrix_free(_Gi[accuX][i]);
      delete[] (_Gi[accuX]);
    }
  }

  delete[] _dsXA;
  delete[] _addCount;
  delete[] _gammaX;

  delete[] _beta;
  delete[] _count;
  delete[] _z;
  delete[] _Gi;
  delete[] _w;

  for (unsigned i = 0; i < _dimN; i++) {
    gsl_matrix_free(_U[i]);
    gsl_matrix_free(_V[i]);
    gsl_vector_free(_singularVals[i]);
  }
  delete[] _U;
  delete[] _V;
  delete[] _singularVals;

  gsl_vector_free(_d);
  gsl_vector_free(_p);
  gsl_vector_free(_workspace);
  gsl_vector_free(_workspaceCof);

  gsl_permutation_free(_permutation);
  gsl_matrix_free(_cofactorU);
  gsl_matrix_free(_cofactorV);
  gsl_vector_free(_cofactorS);
  gsl_matrix_free(_inverseMatrix);
}

void FeatureSpaceAdaptationFeature::add(unsigned dsX)
{
  int               dimN = _w[0]->size1;
  const DistribBasicPtr&  dsP(_dss->find(dsX));
  const CodebookBasicPtr& cbP(dsP->cbk());
  
  if (dimN != cbP->featLen())
    throw jdimension_error("FeatureSpaceAdaptationFeature::add: %s has bad dimension. \n",
			   dsP->name().c_str());

  _dsXA[dsX] = true;
}

void FeatureSpaceAdaptationFeature::addAll()
{
  for (unsigned i = 0; i < _dssN; i++)
    _dsXA[i] = true;
}

int FeatureSpaceAdaptationFeature::compareSortElement(const void* x, const void* y)
{
  SortElement* xs = (SortElement*) x;
  SortElement* ys = (SortElement*) y;
  if      ( xs->_gamma > ys->_gamma) return -1;
  else if ( xs->_gamma < ys->_gamma) return  1;
  else                               return  0;
}

void FeatureSpaceAdaptationFeature::_accuOne(unsigned dsX, int frameX, unsigned accuX, float gamma)
{
  if (_dsXA[dsX] == false) return;

  _count[accuX]++;
  const DistribBasicPtr&  dsP(_dss->find(dsX));
  const CodebookBasicPtr& cbP(dsP->cbk());

  const float* pattern = _src->next(frameX)->data;
  cbP->_scoreOpt(frameX, dsP->val(), _addCount);
  for (unsigned refX = 0; refX < cbP->refN(); refX++) {
    _gammaX[refX]._gamma = _addCount[refX];
    _gammaX[refX]._refX  = refX;
  }
  qsort(_gammaX, cbP->refN(), sizeof(SortElement), compareSortElement);

  // normalize sum of all probs of all (topN) ref vectors (should be one)
  float countSum = 0.0;
  for (unsigned topX = 0; topX < _topN; topX++) {
    unsigned refX    = _gammaX[topX]._refX;
    countSum        += _addCount[refX]; 
  }
  for (unsigned topX = 0; topX < _topN; topX++) {
    unsigned refX    = _gammaX[topX]._refX;
    _addCount[refX] /= countSum;
  }

  // update total prob : beta =sum_s_t gamma(s,t)
  _beta[accuX] += gamma;

  for (unsigned topX = 0; topX < _topN; topX++) {
    unsigned refX = _gammaX[topX]._refX;

    for (unsigned i = 0; i < _dimN; i++) {
      double  c = gamma * _addCount[refX];

      c *= cbP->mean(refX, i) * cbP->invCov(refX, i);

      gsl_matrix_set(_z[accuX], i, 0, gsl_matrix_get(_z[accuX], i, 0) + c);
      for (unsigned j = 0; j < _dimN; j++)
	gsl_matrix_set(_z[accuX], i, j+1, gsl_matrix_get(_z[accuX], i, j+1) + c * pattern[j]);
    }

    for (unsigned i = 0; i < _dimN; i++) {
      double   c = gamma * _addCount[refX];

      c *= cbP->invCov(refX, i);

      gsl_matrix_set(_Gi[accuX][i], 0, 0, gsl_matrix_get(_Gi[accuX][i], 0, 0) + c);
      for (unsigned j = 0; j < _dimN; j++) {
	double d = c * pattern[j];
	gsl_matrix_set(_Gi[accuX][i], j+1, 0, gsl_matrix_get(_Gi[accuX][i], j+1, 0) + d);
	for (unsigned k = 0; k <= j; k++) {
	  gsl_matrix_set(_Gi[accuX][i], j+1, k+1, gsl_matrix_get(_Gi[accuX][i], j+1, k+1) + d * pattern[k]);
	}
      }
    }
  }
}

const gsl_vector_float* FeatureSpaceAdaptationFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (_dss.isNull())
    throw jconsistency_error("Must initialize the distribution set before using.");

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);

  const gsl_vector_float* srcVec = _src->next(_frameX + 1);
  _increment();

  unsigned tranX = 0;
  if (_shift == -1) {

    // translation only, no rotation
    for (unsigned dimX = 0; dimX < _dimN; dimX++)
      gsl_vector_float_set(_vector, dimX, gsl_matrix_get(_w[tranX], dimX, 0) + gsl_vector_float_get(srcVec, dimX));

  } else {

    // do all
    for (unsigned dimX = 0; dimX < _dimN; dimX++) {
      double sum = _shift * gsl_matrix_get(_w[tranX], dimX, 0);
      for (unsigned i = 0; i < _dimN; i++)
	sum += gsl_matrix_get(_w[tranX], dimX, i+1) * gsl_vector_float_get(srcVec, i);
      gsl_vector_float_set(_vector, dimX, float(sum));
    }

  }

  return _vector;
}

void FeatureSpaceAdaptationFeature::accumulate(DistribPathPtr& path, unsigned accuX, float factor)
{
  unsigned frameX = 0;
  for (DistribPath::Iterator itr(path); itr.more(); itr++) {
    unsigned idx;
    try {
      idx = itr.index();
    } catch (...) {
      // printf("Could not find distribution '%s'. Continuing ...\n", itr.name().c_str());
      continue;
    }
    _accuOne(idx, frameX++, accuX, factor);
  }

  cout << "Accumulated " << frameX << " frames." << endl;
}

void FeatureSpaceAdaptationFeature::accumulateLattice(LatticePtr& lat, unsigned accuX, float factor)
{
  _dss->resetCache();
  unsigned cnt = 0;
  for (Lattice::EdgeIterator itr(lat); itr.more(); itr++) {
    Lattice::EdgePtr& edge(itr.edge());

    unsigned distX = edge->input();

    if (distX == 0) continue;

    LogDouble gamma = edge->data().gamma();

    if (gamma > 10.0) continue;

    double postProb = factor * exp(-gamma);

    cnt++;
    for (int frameX = edge->data().start(); frameX <= edge->data().end(); frameX++)
      _accuOne(distX-1, frameX++, accuX, postProb);
  }
  printf("Finished accumulation for %d links.\n", cnt);  fflush(stdout);
}

void FeatureSpaceAdaptationFeature::_makeSymmetric(gsl_matrix* mat)
{
  for (unsigned j = 0; j <= _dimN; j++)
    for (unsigned k = 0; k < j; k++)
      gsl_matrix_set(mat, k, j, gsl_matrix_get(mat, j, k));
}

const double FeatureSpaceAdaptationFeature::MaxSingularValueRatio = 1.0e-07;

double FeatureSpaceAdaptationFeature::_calcCofactor(const gsl_matrix* w, unsigned i0, gsl_vector* p)
{
  for (unsigned i = 0; i < _dimN; i++)
    for (unsigned j = 0; j < _dimN; j++)
      gsl_matrix_set(_cofactorU, i, j, gsl_matrix_get(w, i, j+1));

  gsl_linalg_SV_decomp(_cofactorU, _cofactorV, _cofactorS, _workspaceCof);
  double largestSingularValue  = gsl_vector_get(_cofactorS, 0);
  double determinantForEating  = 1.0;
  double determinantForSelling = 1.0;
  for (unsigned colX = 0; colX < _dimN; colX++) {
    double sing = gsl_vector_get(_cofactorS, colX);
    determinantForSelling *= sing;
    if ((gsl_vector_get(_cofactorS, colX) / largestSingularValue) < MaxSingularValueRatio) {
      sing = 0.0;
    } else {
      determinantForEating *= sing;
      sing = 1.0 / sing;
    }
    for (unsigned rowX = 0; rowX < _dimN; rowX++)
      gsl_matrix_set(_cofactorV, rowX, colX, sing * gsl_matrix_get(_cofactorV, rowX, colX));
  }
  gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, _cofactorV, _cofactorU, 0.0, _inverseMatrix);

  gsl_vector_set(p, 0, 0.0);
  for (unsigned i = 0; i < _dimN; i++)
    gsl_vector_set(p, i + 1, determinantForEating * gsl_matrix_get(_inverseMatrix, i, i0));

  return determinantForSelling;
}

void FeatureSpaceAdaptationFeature::estimate(unsigned iterN, unsigned accuX, unsigned tranX)
{
  double beta2 = _beta[accuX];

  // compute singular value decompositions of 'Gi'
  for (unsigned i = 0; i < _dimN; i++) {
    gsl_matrix_memcpy(_U[i], _Gi[accuX][i]);
    _makeSymmetric(_U[i]);
    gsl_linalg_SV_decomp(_U[i], _V[i], _singularVals[i], _workspace);
  }

  for (unsigned iterX = 0; iterX < iterN; iterX++) {
    for (unsigned i = 0; i < _dimN; i++) {
      double determinant = _calcCofactor(_w[tranX], i, _p);

      // d[i] = p[i] * GINV[i]
      double largestSingularValue = gsl_vector_get(_singularVals[i], 0);
      gsl_blas_dgemv(CblasTrans, 1.0, _V[i], _p, 0.0, _workspace);
      for (unsigned n = 0; n <= _dimN; n++) {
	if ((gsl_vector_get(_singularVals[i], n) / largestSingularValue) >= MaxSingularValueRatio)
	  gsl_vector_set(_workspace, n, (gsl_vector_get(_workspace, n) / gsl_vector_get(_singularVals[i], n)));
	else
	  gsl_vector_set(_workspace, n, 0.0);
      }
      gsl_blas_dgemv(CblasNoTrans, 1.0, _U[i], _workspace, 0.0, _d);

      // term1 = p[i] * GINV[i] * p[i] = dTmp * p[i]
      double  term1;
      gsl_blas_ddot(_d, _p, &term1);
      
      // term2 = p[i] * GINV[i] * z[i] = dTmp * z[i]
      double  term2 = 0.0;
      for (unsigned j = 0; j <= _dimN; j++)
	term2 += gsl_vector_get(_d, j) * gsl_matrix_get(_z[accuX], i, j);
      
      // solve term1 * a^2 + term2 * a - beta = 0
      double term3 = term2 / (2 * term1);
      double term4 = sqrt(term3 * term3 + beta2 / term1);

      double alpha1 = -1.0 * term3 + term4;
      double alpha2 = -1.0 * term3 - term4;

      // select alpha which maximize likelihood
      double kullback1 = beta2 * log(fabs(alpha1 * term1 + term2)) - 0.5 * alpha1 * alpha1 * term1;
      double kullback2 = beta2 * log(fabs(alpha2 * term1 + term2)) - 0.5 * alpha2 * alpha2 * term1;
      double alpha     = (kullback2 > kullback1) ? alpha2 : alpha1;

      /*
      printf("Iteration %d : Dimension %d : Determinant %g : Kullback %g\n", iterX, i, determinant, max(kullback1, kullback2));
      */

      // reestimate w : w[i] = (alpha*p[i] + z[i]) * ginv[i])
      for (unsigned j = 0; j <= _dimN; j++)
	gsl_vector_set(_workspace, j, gsl_matrix_get(_z[accuX], i, j));
      gsl_blas_daxpy(alpha, _p, _workspace);

      gsl_blas_dgemv(CblasTrans, 1.0, _V[i], _workspace, 0.0, _p);
      for (unsigned n = 0; n <= _dimN; n++) {
	if ((gsl_vector_get(_singularVals[i], n) / largestSingularValue) >= MaxSingularValueRatio)
	  gsl_vector_set(_p, n, gsl_vector_get(_p, n) / gsl_vector_get(_singularVals[i], n));
	else
	  gsl_vector_set(_p, n, 0.0);
      }
      gsl_blas_dgemv(CblasNoTrans, 1.0, _U[i], _p, 0.0, _d);

      // copy w[i] into full matrix
      for (unsigned j = 0; j <= _dimN; j++)
	gsl_matrix_set(_w[tranX], i, j, gsl_vector_get(_d, j));
    }
  }
}

void FeatureSpaceAdaptationFeature::zeroAccu(unsigned accuX)
{
  if (accuX >= _maxAccu)
    throw jindex_error("FeatureSpaceAdaptationFeatureClearAccu: Bad accu index: %d should be less then %d.",
	  accuX, _maxAccu);

  _count[accuX] = 0.0;
  _beta[accuX]  = 0.0;
  gsl_matrix_set_zero(_z[accuX]);
  for (unsigned i = 0; i < _dimN;i++)
    gsl_matrix_set_zero(_Gi[accuX][i]);
}

void FeatureSpaceAdaptationFeature::scaleAccu(float scale, unsigned accuX)
{
  if (accuX >= _maxAccu)
    throw jindex_error("FeatureSpaceAdaptationFeatureClearAccu: Bad accu index: %d should be less then %d.",
		     accuX, _maxAccu);

  _count[accuX] *= scale;
  _beta[accuX]  *= scale;

  for (unsigned i = 0; i < _dimN; i++)
    for (unsigned j = 0; j <= _dimN; j++)
      gsl_matrix_set(_z[accuX], i, j, gsl_matrix_get(_z[accuX], i, j) * scale);

  for (unsigned i = 0; i < _dimN; i++)
    for (unsigned j = 0; j <= _dimN; j++)
      for (unsigned k = 0; k <= _dimN; k++)
	gsl_matrix_set(_Gi[accuX][i], j, k, gsl_matrix_get(_Gi[accuX][i], j, k) * scale);
}

void FeatureSpaceAdaptationFeature::clear(unsigned tranX)
{
  if (tranX >= _maxTran)
    throw jindex_error("FeatureSpaceAdaptationFeature::clear: Bad transformation index: %d should be less then %d.\n",
		       tranX, _maxTran);

  // Init saP->w = Identity
  for (unsigned i = 0; i < _dimN; i++) {
    // w[i][0] holds the shift
    gsl_matrix_set(_w[tranX], i, 0, 0.0);
    for (unsigned j = 0; j < _dimN; j++) {
      gsl_matrix_set(_w[tranX], i, j+1, ((i == j) ? 1.0 : 0.0));
    }
  }
}

float FeatureSpaceAdaptationFeature::compareTransform(unsigned tranX, unsigned tranY)
{
  if (tranX >= _maxTran)
    throw jindex_error("FeatureSpaceAdaptationFeature::compareTransform : Bad transformation index: %d should be less then %d.\n",
		       tranX, _maxTran);

  if (tranY >= _maxTran)
    throw jindex_error("FeatureSpaceAdaptationFeature::compareTransform : Bad transformation index: %d should be less then %d.\n",
		       tranY, _maxTran);

  // sum of squares
  double sum = 0.0;
  for (unsigned i = 0; i < _dimN; i++) {
    for (unsigned j = 0; j < _dimN;j++) {
      double tmp = gsl_matrix_get(_w[tranX], i, j) - gsl_matrix_get(_w[tranY], i, j);
      sum += tmp * tmp;
    }
  }

  return sum;
}

void FeatureSpaceAdaptationFeature::saveAccu(const String& name, unsigned accuX)
{
  float dummy;

  if (accuX >= _maxAccu)
    throw jindex_error("FeatureSpaceAdaptationFeature::writeAccu: Bad accu index: %d should be less then %d.\n",
	  accuX, _maxAccu);

  FILE* fp = fileOpen(name,"w");

  fwrite("SAS",sizeof(char),3,fp);

  write_int(fp, _dimN);
  write_float(fp, _count[accuX]);
  write_float(fp, _beta[accuX]);

  // write z
  for (unsigned i = 0; i < _dimN; i++)
    for (unsigned j = 0; j <= _dimN; j++)
      write_float(fp, float(gsl_matrix_get(_z[accuX], i, j)));

  // write g
  for (unsigned i = 0; i < _dimN; i++)
    for (unsigned j = 0; j <= _dimN;j++)
      for (unsigned k = 0; k <= _dimN; k++)
	write_float(fp, float(gsl_matrix_get(_Gi[accuX][i], j, k)));

  fileClose(name,fp);
}

void FeatureSpaceAdaptationFeature::loadAccu(const String& name, unsigned accuX, float factor)
{ 
  char  fmagic[3];

  if (accuX >= _maxAccu)
    throw jindex_error("FeatureSpaceAdaptationFeature::readAccu: Bad accu index: %d should be less then %d.",
	  accuX, _maxAccu);

  FILE* fp = fileOpen(name,"r");

  if ( fread (fmagic, sizeof(char), 3, fp) != 3 || strncmp("SAS",fmagic,3)) {
    fileClose(name,fp);
    throw jconsistency_error("FeatureSpaceAdaptationFeature::readAccu: Couldn't find magic SAS in file %s.",
			     name.c_str());
  }

  unsigned dimN = (unsigned) read_int(fp);

  if (dimN != _dimN) {
    fileClose(name,fp);
    throw jdimension_error("FeatureSpaceAdaptationFeature::readAccu: Dimension mismatch. Exptected dimN= %d, but got %d.",
	  _dimN, dimN);
  }
	
  // read count, beta
  _count[accuX] += factor * read_float(fp);
  float val = read_float(fp);
  _beta[accuX]  += double(factor) * double(val);

  // read z
  for (unsigned i = 0 ; i < _dimN; i++) {
    for (unsigned j = 0; j <= _dimN; j++) {
      val = read_float(fp);
      gsl_matrix_set(_z[accuX], i, j, gsl_matrix_get(_z[accuX], i, j) + double(factor) * double(val));
    }
  }

  // read g[i]
  for (unsigned i = 0; i < _dimN; i++) {
    for (unsigned j = 0 ; j <= _dimN; j++) {
      for (unsigned k = 0; k <= _dimN; k++) {
	val = read_float(fp);
	gsl_matrix_set(_Gi[accuX][i], j, k, gsl_matrix_get(_Gi[accuX][i], j, k) + double(factor) * double(val));
      }
    }
  }

  fileClose(name,fp);
}

void FeatureSpaceAdaptationFeature::save(const String& name, unsigned tranX)
{
  if (tranX >= _maxTran)
    throw jindex_error("FeatureSpaceAdaptationFeature::save: Bad transformation index: %d should be less then %d.\n",
		       tranX, _maxTran);

  FILE* fp = fileOpen(name,"w");

  fwrite("SAW",sizeof(char),3,fp);

  write_int(fp, _dimN);

  // write w
  for (unsigned i = 0; i < _dimN; i++)
    for (unsigned j = 0;j <= _dimN;j++)
      write_float(fp, float(gsl_matrix_get(_w[tranX], i, j)));

  fileClose(name,fp);
}

void FeatureSpaceAdaptationFeature::load(const String& name, unsigned tranX)
{
  char fmagic[3];

  if (tranX >= _maxTran)
    throw jindex_error("FeatureSpaceAdaptationFeature::load: Bad transformation index: %d should be less then %d.\n",
		       tranX, _maxTran);

  FILE* fp = fileOpen(name,"r");

  if ( fread (fmagic, sizeof(char), 3, fp) != 3 || strncmp("SAW",fmagic,3)) {
    fileClose(name,fp);
    throw jio_error("FeatureSpaceAdaptationFeature::load: Couldn't find magic SAW in file %s\n",
		    name.c_str());
  }

  unsigned dimN = (unsigned) read_int(fp);

  if (dimN != _dimN) {
    fileClose(name, fp);
    throw jdimension_error("FeatureSpaceAdaptationFeature::load: Dimension mismatch. Expected dimN= %d but got %d.\n",
			   _dimN, dimN);
  }

  // read w
  for (unsigned i = 0; i < _dimN; i++)
    for (unsigned j = 0; j <= _dimN;j++)
      gsl_matrix_set(_w[tranX], i, j, double(read_float(fp)));

  fileClose(name,fp);
}

void FeatureSpaceAdaptationFeature::addAccu(unsigned accuX, unsigned accuY, float factor)
{ 
  _count[accuX] += factor * _count[accuY];
  _beta[accuX]  += factor * _beta[accuY];

  for (unsigned i = 0; i < _dimN; i++)
    for (unsigned j = 0; j <= _dimN; j++)
      gsl_matrix_set(_z[accuX], i, j,  gsl_matrix_get(_z[accuX], i, j) + factor * gsl_matrix_get(_z[accuY], i, j));

  for (unsigned i = 0; i < _dimN;i++)
    for (unsigned j = 0; j <= _dimN;j++)
      for (unsigned k = 0; k <= _dimN;k++)
	gsl_matrix_set(_Gi[accuX][i], j, k, gsl_matrix_get(_Gi[accuX][i], j, k) + factor * gsl_matrix_get(_Gi[accuY][i], j, k));
}
