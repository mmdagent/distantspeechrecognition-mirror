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


#include <vector>
#include <algorithm>
#include "common/mach_ind_io.h"
#include "sat/sat.h"

#include <gsl/gsl_eigen.h>


// ----- methods for class `SATMean' -----
//
unsigned    SATMean::nUse         = 0;
bool	    SATMean::allocated    = false;
gsl_matrix* SATMean::temp1        = NULL;
gsl_matrix* SATMean::temp2        = NULL;
gsl_vector* SATMean::vec1         = NULL;
gsl_vector* SATMean::vec2         = NULL;

gsl_matrix* SATMean::U            = NULL;
gsl_matrix* SATMean::V            = NULL;
gsl_vector* SATMean::newMean      = NULL;
gsl_vector* SATMean::singularVals = NULL;
gsl_vector* SATMean::tempVec      = NULL;

gsl_matrix* SATMean::matCopy      = NULL;
gsl_vector* SATMean::vecCopy      = NULL;
gsl_vector* SATMean::vecProduct   = NULL;

gsl_vector* SATMean::_workSpace   = NULL;

void SATMean::_checkBlocks(UnShrt nblk, bool allocVarVec)
{
  if (_block != NULL) {
    if (nblk != _nBlocks)
      throw jdimension_error("Block sizes (%d vs %d) are not equivalent.",
			     nblk, _nBlocks);
    return;
  }

  if (_featLen % nblk != 0)
    throw jdimension_error("Transformed feature length %d is not compatible with %d blocks.",
			   _featLen, nblk);
  if (_origFeatLen % nblk != 0)
    throw jdimension_error("Original feature length %d is not compatible with %d blocks.",
			   _featLen, nblk);

  _blockLen = _featLen / nblk;     _origBlockLen = _origFeatLen / nblk;
  _block    = new BlockAcc[nblk];  _nBlocks = nblk;
  for (UnShrt nb = 0; nb < nblk; nb++) {
    gsl_matrix* mat = _block[nb]._mat = gsl_matrix_alloc(_origBlockLen, _origBlockLen);
    gsl_vector* vec = _block[nb]._vec = gsl_vector_alloc(_origBlockLen);

    gsl_matrix_set_zero(mat);  gsl_vector_set_zero(vec); _block[nb]._scalar = 0.0;

    if (allocVarVec) {
      gsl_vector* varVec = _block[nb]._varVec = gsl_vector_alloc(_blockLen);  gsl_vector_set_zero(varVec);
    }
  }

  if (allocated == true) return;

  printf("Allocating SAT working space (%d x %d) ... ",
	 _blockLen, _origBlockLen);  fflush(stdout);

  temp1        = gsl_matrix_alloc(_blockLen,     _origBlockLen);
  temp2        = gsl_matrix_alloc(_origBlockLen, _origBlockLen);

  vec1         = gsl_vector_alloc(_blockLen);
  vec2         = gsl_vector_alloc(_origBlockLen);

  U            = gsl_matrix_alloc(_origBlockLen, _origBlockLen);
  V            = gsl_matrix_alloc(_origBlockLen, _origBlockLen);

  newMean      = gsl_vector_alloc(_origBlockLen);
  singularVals = gsl_vector_alloc(_origBlockLen);
  tempVec      = gsl_vector_alloc(_origBlockLen);

  matCopy      = gsl_matrix_alloc(_origBlockLen, _origBlockLen);
  vecCopy      = gsl_vector_alloc(_origBlockLen);
  vecProduct   = gsl_vector_alloc(_origBlockLen);

  _workSpace   = gsl_vector_alloc(_origBlockLen);

  allocated = true;

  cout << "Done." << endl; 
}

SATMean::SATMean(UnShrt len, UnShrt orgLen)
  : _occ(0.0), _featLen(len), _origFeatLen(orgLen),
    _nBlocks(0), _blockLen(0), _origBlockLen(0), _block(NULL)
{
  nUse++;
}

SATMean::~SATMean()
{
  for (UnShrt nb = 0; nb < _nBlocks; nb++) {
    gsl_matrix_free(_block[nb]._mat);
    gsl_vector_free(_block[nb]._vec);

    if (_block[nb]._varVec) gsl_vector_free(_block[nb]._varVec);
  }
  delete[] _block;

  if (--nUse > 0 || allocated == false) return;

  cout << "De-allocating SAT working space ... "; 

  gsl_matrix_free(temp1);        	temp1        = NULL;
  gsl_matrix_free(temp2);        	temp2        = NULL;

  gsl_vector_free(vec1);         	vec1         = NULL;
  gsl_vector_free(vec2);         	vec2         = NULL;

  gsl_matrix_free(U);            	U            = NULL;
  gsl_matrix_free(V);            	V            = NULL;

  gsl_vector_free(newMean);      	newMean      = NULL;
  gsl_vector_free(singularVals); 	singularVals = NULL;
  gsl_vector_free(tempVec);		tempVec      = NULL;

  gsl_matrix_free(matCopy);  		matCopy      = NULL;
  gsl_vector_free(vecCopy);  		vecCopy      = NULL;
  gsl_vector_free(vecProduct);   	vecProduct   = NULL;
  gsl_vector_free(_workSpace);   	_workSpace   = NULL;

  allocated = false;

  cout << "Done." << endl;
}

void SATMean::_validate(gsl_matrix* mat, gsl_vector* vec, double scalar)
{
  for (UnShrt m = 0; m < _origBlockLen; m++)
    for (UnShrt n = 0; n < _origBlockLen; n++)
      if (isnan(gsl_matrix_get(mat, m, n)))
	throw jnumeric_error("Matrix element (%d,%d) is NaN.", m, n);

  for (UnShrt n = 0; n < _origBlockLen; n++)
    if (isnan(gsl_vector_get(vec, n)))
      throw jnumeric_error("Vector element (%d) is NaN.", n);

  if (isnan(scalar))
    throw jnumeric_error("Scalar is NaN.");
}

void SATMean::accumulate(const TransformBasePtr& transformer, float c_k,
			 const InverseVariance& invVar, const DNaturalVector& sumO,
			 bool addScalar)
{
  assert(transformer->featLen()    == _featLen);
  assert(transformer->orgFeatLen() == _origFeatLen);
  assert(sumO.featLen()            == _featLen);
  assert(invVar.featLen()          == _featLen);

  if (c_k == 0.0) return;

  const TransformMatrix& transformMatrix(transformer->transformMatrix());
  _checkBlocks(transformMatrix.nBlocks());
  for (UnShrt nb = 0; nb < _nBlocks; nb++) {
    UnShrt offsetIndex = nb * _blockLen;
    const gsl_matrix* xformMatrix = transformMatrix.matrix(nb);
    gsl_matrix*    _mat    = _block[nb]._mat;
    gsl_vector*    _vec    = _block[nb]._vec;
    double& _scalar = _block[nb]._scalar;

    // accumulate matrix sufficient statistic
    gsl_matrix_memcpy(temp1, xformMatrix);
    for (UnShrt m = 0; m < _blockLen; m++) {
      float inv = invVar(m + offsetIndex);
      for (UnShrt n = 0; n < _origBlockLen; n++)
	gsl_matrix_set(temp1, m, n, gsl_matrix_get(temp1, m, n) * inv);
    }
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, c_k, temp1, xformMatrix, 1.0, _mat);

    if (addScalar) {

      // accumulate vector sufficient statistic
      const gsl_vector* offset = transformMatrix.offset(nb);
      for (UnShrt m = 0; m < _blockLen; m++)
	gsl_vector_set(vec1, m,
		       sumO(m + offsetIndex) - (c_k * gsl_vector_get(offset, m)));
      gsl_blas_dgemv(CblasTrans, 1.0, temp1, vec1, 1.0, _vec);

      // accumulate scalar sufficient statistic
      for (UnShrt m = 0; m < _blockLen; m++) {
	double vElem = gsl_vector_get(vec1, m);
	_scalar += invVar(m + offsetIndex) * vElem * vElem / c_k;
      }
    }
    _validate(_mat, _vec, _scalar);
  }
  _occ += c_k;
}

double SATMean::fixup(NaturalVector satMean)
{
  assert(satMean.featLen() == _origFeatLen);

  double auxFunc = 0.0;		// copy over original mean
  for (UnShrt nb = 0; nb < _nBlocks; nb++) {
    UnShrt offset = nb * _origBlockLen;
    for (UnShrt n = 0; n < _origBlockLen; n++)
      gsl_vector_set(newMean, n, satMean(n + offset));

    const gsl_matrix* _mat    = _block[nb]._mat;
    const gsl_vector* _vec    = _block[nb]._vec;
    double     _scalar = _block[nb]._scalar;

    auxFunc += _auxFunc(_mat, _vec, _scalar, newMean);
  }

  return auxFunc;
}

void SATMean::zero()
{
  if (_block == NULL) return;

  for (UnShrt nb = 0; nb < _nBlocks; nb++) {
    gsl_matrix_set_zero(_block[nb]._mat);
    gsl_vector_set_zero(_block[nb]._vec);
    _block[nb]._scalar = 0.0;

    if (_block[nb]._varVec)
      gsl_vector_set_zero(_block[nb]._varVec);
  }
  _occ = 0.0;
}

const double SATMean::MaxSingularValueRatio = 1.0e-07;

void SATMean::_solve(gsl_matrix* _mat, gsl_vector* _vec, gsl_vector* _newMean, UpdateInfo* info)
{
  // use SVD to solve for new mean
  gsl_linalg_SV_decomp(_mat, V, singularVals, _workSpace);
  gsl_blas_dgemv(CblasTrans, 1.0, _mat, _vec, 0.0, tempVec);
  double largestValue = gsl_vector_get(singularVals, 0);
  for (UnShrt n = 0; n < _origBlockLen; n++) {
    double evalue  = gsl_vector_get(singularVals, n);
    if ((evalue / largestValue) >= MaxSingularValueRatio) {
      gsl_vector_set(tempVec, n, gsl_vector_get(tempVec, n) / evalue);
      if (info) info->ttlDimensionsUpdated++;
    } else
      gsl_vector_set(tempVec, n, 0.0);
  }
  gsl_blas_dgemv(CblasNoTrans, 1.0, V, tempVec, 0.0, _newMean);

  if (info) info->ttlDimensions = _origBlockLen;
}

double SATMean::
_auxFunc(const gsl_matrix* mat, const gsl_vector* vec, double scalar, const gsl_vector* mean)
{
  double ip1, ip2;

  gsl_blas_dgemv(CblasNoTrans, 1.0, mat, mean, 0.0, vecProduct);
  gsl_blas_ddot(vecProduct, mean, &ip1);
  gsl_blas_ddot(mean, vec, &ip2);

  return 0.5 * (ip1 + scalar) - ip2;
}

double SATMean::update(NaturalVector satMean, UpdateInfo* info, UnShrt* dl)
{
  assert(satMean.featLen() == _origFeatLen);

  if (_occ == 0.0) {      // is this really necessary?
    for (UnShrt nb = 0; nb < _nBlocks; nb++) {
      UnShrt offset = nb * _origBlockLen;
      for (UnShrt m = 0; m < _origBlockLen; m++) {
	satMean(m + offset) = 0.0;
      }
    }
    return 0.0;
  }

  double auxFuncSum = 0.0;
  for (UnShrt nb = 0; nb < _nBlocks; nb++) {
    UpdateInfo uinfo(/* ttlBlocks = */ 1, /* ttlBlocksUpdated = */ 1);
    gsl_matrix*   _mat    = _block[nb]._mat;
    gsl_vector*   _vec    = _block[nb]._vec;
    double _scalar = _block[nb]._scalar;

    gsl_matrix_memcpy(matCopy, _mat);	// copy matrix and vector components
    gsl_vector_memcpy(vecCopy, _vec);	// needed to calculate final value
					// of auxiliary function
    UnShrt offset = nb * _origBlockLen;
    for (UnShrt n = 0; n < _origBlockLen; n++)
      gsl_vector_set(newMean, n, satMean(n + offset));
    double auxFuncOld = _auxFunc(matCopy, vecCopy, _scalar, newMean);

    _solve(_mat,_vec,newMean,&uinfo);	// solve for optimal mean

    double auxFuncNew =			// make sure new mean is really better
      _auxFunc(matCopy, vecCopy, _scalar, newMean);

    if (auxFuncOld < auxFuncNew) {
      auxFuncSum += auxFuncOld;
      if (info) info->ttlBlocks++;
      continue;
    }
					// copy over optimal mean
    for (UnShrt n = 0; n < _origBlockLen; n++) {
      double q = gsl_vector_get(newMean, n);
      if (isnan(q)) {
	gsl_matrix_fprintf(stdout, matCopy, "%g");
	gsl_vector_fprintf(stdout, vecCopy, "%g");
	gsl_vector_fprintf(stdout, newMean, "%g");
	throw jnumeric_error("Degenerate value in mean component %d", n);
      }
      satMean(n + offset) = q;
    }
    auxFuncSum += auxFuncNew;
    if (info) (*info) += uinfo;
  }
  return auxFuncSum;
}


// ----- methods for class `SATMean::UpdateInfo' -----
//
SATMean::UpdateInfo::UpdateInfo() :
  ttlBlocks(0), ttlBlocksUpdated(0),
  ttlDimensions(0), ttlDimensionsUpdated(0) { }

SATMean::UpdateInfo::UpdateInfo(unsigned tb, unsigned tbu) :
  ttlBlocks(tb), ttlBlocksUpdated(tbu),
  ttlDimensions(0), ttlDimensionsUpdated(0) { }

void SATMean::UpdateInfo::zero()
{
  ttlBlocks = ttlBlocksUpdated = ttlDimensions = ttlDimensionsUpdated = 0;
}

void SATMean::UpdateInfo::operator+=(const UpdateInfo& ui)
{
  ttlBlocks            += ui.ttlBlocks;
  ttlBlocksUpdated     += ui.ttlBlocksUpdated;

  ttlDimensions        += ui.ttlDimensions;
  ttlDimensionsUpdated += ui.ttlDimensionsUpdated;
}

ostream& operator<<(ostream& os, const SATMean::UpdateInfo& ui)
{
  os << "Updated " << ui.ttlBlocksUpdated
     << " of "     << ui.ttlBlocks << " blocks." << endl;

  os << "Updated " << ui.ttlDimensionsUpdated
     << " of "     << ui.ttlDimensions << " dimensions." << endl;
  os << "Average dimensionality of sub-block = " << setw(4) <<
    (double) ui.ttlDimensionsUpdated / ui.ttlBlocksUpdated  << endl;

  return os;
}


// ----- methods for class `MDLSATMean' -----
//
gsl_eigen_symmv_workspace* MDLSATMean::_workSpace = NULL;
MDLSATMean::MDLSATMean(UnShrt len, UnShrt orgLen, double thr)
  : SATMean(len, orgLen), _lhoodThresh(thr)
{
  if (_workSpace == NULL)
    _workSpace = gsl_eigen_symmv_alloc(_origFeatLen);
}

MDLSATMean::~MDLSATMean() { }

double MDLSATMean::update(NaturalVector satMean, UpdateInfo* info, UnShrt* descLen)
{
  assert(satMean.featLen() == _origFeatLen);

  if (_occ == 0.0) {			// is this really necessary?
    for (UnShrt nb = 0; nb < _nBlocks; nb++) {
      UnShrt offset = nb * _origBlockLen;
      for (UnShrt m = 0; m < _origBlockLen; m++) {
	satMean(m + offset) = 0.0;
      }
    }
    return 0.0;
  }

  double auxFuncSum = 0.0;
  for (UnShrt nb = 0; nb < _nBlocks; nb++) {
    UpdateInfo uinfo(/* ttlBlocks = */ 1, /* ttlBlocksUpdated = */ 1);
    gsl_matrix*   _mat    = _block[nb]._mat;
    gsl_vector*   _vec    = _block[nb]._vec;
    double _scalar = _block[nb]._scalar;

    gsl_matrix_memcpy(matCopy, _mat);	// copy matrix and vector components
    gsl_vector_memcpy(vecCopy, _vec);	// needed to calculate final value
					// of auxiliary function

    UnShrt offset = nb * _origBlockLen;
    for (UnShrt n = 0; n < _origBlockLen; n++)
      gsl_vector_set(newMean, n, satMean(n + offset));

    double mdlPenalty = (descLen != NULL) ? (_lhoodThresh * descLen[nb]) : 0.0;
    double auxFuncOld =
      _auxFunc(matCopy, vecCopy, _scalar, newMean) + mdlPenalty;

    _solve(_mat,_vec,newMean,&uinfo);	// solve for optimal mean

    double auxFuncNew =			// make sure new mean is really better
      _auxFunc(matCopy, vecCopy, _scalar, newMean)
      + (_lhoodThresh * uinfo.ttlDimensionsUpdated);

    /*
    cout << endl << "Old aux. func. = " << auxFuncOld
	 << endl << "New aux. func. = " << auxFuncNew << endl;
    */

    if (auxFuncOld < auxFuncNew) {
      auxFuncSum += auxFuncOld;
      if (info) info->ttlBlocks++;
      continue;
    }

    if (descLen != NULL)		// copy over optimal mean
      descLen[nb] = uinfo.ttlDimensionsUpdated;
    for (UnShrt n = 0; n < _origBlockLen; n++) {
      double q = gsl_vector_get(newMean, n);
      if (isnan(q)) {
	gsl_matrix_fprintf(stdout, matCopy, "%g");
	gsl_vector_fprintf(stdout, vecCopy, "%g");
	gsl_vector_fprintf(stdout, newMean, "%g");
	throw jnumeric_error("Degenerate value in mean component %d", n);
      }
      satMean(n + offset) = q;
    }
    auxFuncSum += auxFuncNew;
    if (info) (*info) += uinfo;
  }

  return auxFuncSum;
}

void MDLSATMean::_checkBlocks(UnShrt nblk, bool allocVarVec)
{
  SATMean::_checkBlocks(nblk, allocVarVec);
}

void MDLSATMean::_solve(gsl_matrix* _mat, gsl_vector* _vec, gsl_vector* _newMean, UpdateInfo* info)
{
  gsl_eigen_symmv (_mat, singularVals, U, _workSpace);
  gsl_blas_dgemv(CblasTrans, 1.0, U, _vec, 0.0, tempVec);

  double largestValue = 0.0;
  for (UnShrt n = 0; n < _origBlockLen; n++)
    if (gsl_vector_get(singularVals, n) > largestValue)
      largestValue = gsl_vector_get(singularVals, n);

  for (UnShrt n = 0; n < _origBlockLen; n++) {
    double   evalue  = gsl_vector_get(singularVals, n);
    double   tempn   = gsl_vector_get(tempVec, n);
    double contrib = 0.5 * (tempn * tempn / evalue);
    if (contrib > _lhoodThresh && (evalue / largestValue) >= MaxSingularValueRatio) {
      gsl_vector_set(tempVec, n, tempn / evalue);
      if (info) info->ttlDimensionsUpdated++;
    } else {
      gsl_vector_set(tempVec, n, 0.0);
    }
  }

  gsl_blas_dgemv(CblasNoTrans, 1.0, U, tempVec, 0.0, _newMean);
  if (info) info->ttlDimensions = _origBlockLen;
}


// ----- methods for class `FastSAT' -----
//
FastSAT::FastSAT(UnShrt len, UnShrt orgLen, double thr)
  : MDLSATMean(len, orgLen, thr) { }

void FastSAT::accumulate(const TransformBasePtr& transformer, double c_k,
			 const InverseVariance& invVar)
{
  assert(transformer->featLen()    == _featLen);
  assert(transformer->orgFeatLen() == _origFeatLen);
  assert(invVar.featLen()          == _featLen);

  if (c_k == 0.0) return;

  const TransformMatrix& transformMatrix(transformer->transformMatrix());
  _checkBlocks(transformMatrix.nBlocks(), /* allocVarVec= */ true);
  for (UnShrt nb = 0; nb < _nBlocks; nb++) {
    UnShrt offsetIndex = nb * _blockLen;
    const gsl_matrix* xformMatrix = transformMatrix.matrix(nb);
    gsl_matrix* _mat = _block[nb]._mat;

    // accumulate matrix sufficient statistic
    gsl_matrix_memcpy(temp1, xformMatrix);
    for (UnShrt m = 0; m < _blockLen; m++) {
      float inv = invVar(m + offsetIndex);
      for (UnShrt n = 0; n < _origBlockLen; n++)
	gsl_matrix_set(temp1, m, n, gsl_matrix_get(temp1, m, n) * inv);
    }
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, c_k, temp1, xformMatrix, 1.0, _mat);
  }

  _occ += c_k;
}

void FastSAT::addFastAcc(const DNaturalVector& meanAcc, const DNaturalVector& varAcc,
			 const double* scalar, int nblks)
{
  _checkBlocks(nblks, /* allocVarVec= */ true);
  for (UnShrt nb = 0; nb < _nBlocks; nb++) {
    gsl_vector* _vec          = _block[nb]._vec;
    UnShrt offsetIndex = nb * _origBlockLen;
    for (UnShrt m = 0; m < _origBlockLen; m++)
      gsl_vector_set(_vec, m, gsl_vector_get(_vec, m) + meanAcc(m + offsetIndex));

    gsl_vector* varVec   = _block[nb]._varVec;
    offsetIndex   = nb * _blockLen;
    for (UnShrt m = 0; m < _blockLen; m++)
      gsl_vector_set(varVec, m, gsl_vector_get(varVec, m) + varAcc(m + offsetIndex));

    _block[nb]._scalar += scalar[nb];

    if (isnan(_block[nb]._scalar))
      throw jnumeric_error("Scalar is NaN.");
  }
}

double FastSAT::update(NaturalVector satMean, NaturalVector satVar,
		       UpdateInfo* info, UnShrt* dl)
{
  if (SATBase::MeanOnly) {
    for (UnShrt n = 0; n < _featLen; n++)
      satVar(n) = 1.0 / satVar(n);
  } else {
    _updateVariance(satVar);
  }

  return _updateMean(satMean, info, dl);
}

double FastSAT::_updateMean(NaturalVector satMean, UpdateInfo* info, UnShrt* dl)
{
  return MDLSATMean::update(satMean, info, dl);
}

void FastSAT::_updateVariance(NaturalVector satVar)
{
  for (UnShrt nb = 0; nb < _nBlocks; nb++) {
    UnShrt offsetIndex = nb * _blockLen;
    gsl_vector*   varVec      = _block[nb]._varVec;

    for (UnShrt m = 0; m < _blockLen; m++) {
      double var = gsl_vector_get(varVec, m) / _occ;

      if (isnan(var)) {
	printf("Warning: Variance component %d is NaN.\n", m);
	satVar(m + offsetIndex) = 1.0 / satVar(m + offsetIndex);
      } else {
	satVar(m + offsetIndex) = var;
      }
    }
  }
}


// ----- methods for class `SATMeanFullCovar' -----
//
void SATMeanFullCovar::accumulate(const TransformBasePtr& transformer, float c_k,
          const gsl_matrix* invCovar, const DNaturalVector& sumO)
{
  assert(transformer->featLen()    == _featLen);
  assert(transformer->orgFeatLen() == _origFeatLen);
  assert(sumO.featLen()            == _featLen);
  assert(invCovar->size1           == _featLen);

  if (c_k == 0.0) return;

  const TransformMatrix& transformMatrix(transformer->transformMatrix());

  _checkBlocks(transformMatrix.nBlocks());
  for (UnShrt nb = 0; nb < _nBlocks; nb++) {
    UnShrt offsetIndex = nb * _blockLen;
    const gsl_matrix* xformMatrix = transformMatrix.matrix(nb);
    gsl_matrix*    _mat    = _block[nb]._mat;
    gsl_vector*    _vec    = _block[nb]._vec;

				// accumulate matrix sufficient statistic
    gsl_blas_dgemm(CblasTrans,   CblasNoTrans, 1.0, xformMatrix, invCovar,
		   0.0, temp1);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, c_k, temp1, xformMatrix,
		   1.0, _mat);

    				// accumulate vector sufficient statistic
    const gsl_vector* offset = transformMatrix.offset(nb);
    for (UnShrt m = 0; m < _blockLen; m++)
      gsl_vector_set(vec1, m,
		     sumO(m+offsetIndex) - (c_k * gsl_vector_get(offset, m)));
    gsl_blas_dgemv(CblasNoTrans, 1.0, temp1, vec1, 1.0, _vec);
  }
  _occ += c_k;
}

void SATMeanFullCovar::dump(const String& fileName) const
{
  FILE* fp = fileOpen(fileName, "w");

  write_int(fp, _nBlocks);  write_int(fp, _blockLen);  write_int(fp, _origBlockLen);
  write_float(fp, _occ);
  
  for (UnShrt nb = 0; nb < _nBlocks; nb++) {
    gsl_matrix* _mat = _block[nb]._mat;
    gsl_vector* _vec = _block[nb]._vec;

    for (UnShrt i = 0; i < _origBlockLen; i++)
      for (UnShrt j = 0; j < _origBlockLen; j++)
	write_float(fp, gsl_matrix_get(_mat, i, j));
    
    for (UnShrt i = 0; i < _origBlockLen; i++)
      write_float(fp, gsl_vector_get(_vec, i));
  }

  fileClose(fileName, fp);
}

void SATMeanFullCovar::load(const String& fileName)
{
  FILE* fp = fileOpen(fileName, "r");

  int nblks  = read_int(fp);
  int blen   = read_int(fp);
  int oblen  = read_int(fp);

  if (nblks != _nBlocks)
    throw jdimension_error("No. of blocks (%d vs. %d) do not match.", nblks, _nBlocks);
  if (blen != _blockLen)
    throw jdimension_error("Block lengths (%d vs. %d) do not match.", blen, _blockLen);
  if (oblen != _origBlockLen)
    throw jdimension_error("Original block lengths (%d vs. %d) do not match.",
			   oblen, _origBlockLen);
  
  _occ += read_float(fp);

  for (UnShrt nb = 0; nb < _nBlocks; nb++) {
    gsl_matrix* _mat = _block[nb]._mat;
    gsl_vector* _vec = _block[nb]._vec;

    for (UnShrt i = 0; i < _origBlockLen; i++)
      for (UnShrt j = 0; j < _origBlockLen; j++)
	gsl_matrix_set(_mat, i, j,
		       gsl_matrix_get(_mat, i, j) + read_float(fp));
    
    for (UnShrt i = 0; i < _origBlockLen; i++)
      gsl_vector_set(_vec, i, gsl_vector_get(_vec, i) + read_float(fp));
  }

  fileClose(fileName, fp);
}


// ----- methods for class `SATVariance' -----
//
SATVariance::SATVariance(UnShrt len)
  : _featLen(len), _vec(gsl_vector_alloc(len)), _occ(0.0)
{
  gsl_vector_set_zero(_vec);
}

SATVariance::~SATVariance() { gsl_vector_free(_vec); }

void SATVariance::_validate(gsl_vector* vec, double occ)
{
  for (UnShrt m = 0; m < _featLen; m++)
    if (isnan(gsl_vector_get(_vec, m)))
      throw jnumeric_error("Variance component %d is NaN.", m);

  if (isnan(occ))
    throw jnumeric_error("Total frame count is NaN.");
}

void SATVariance::
accumulate(const TransformBasePtr& transformer, double c_k,
	   const NaturalVector& mean, const DNaturalVector& sumO,
	   const DNaturalVector& sumOsq)
{
  assert(transformer->featLen() == _featLen);
  assert(sumO.featLen()         == _featLen);
  assert(sumOsq.featLen()       == _featLen);

  if (c_k == 0.0) return;

  static NaturalVector trnsMean;
  if (trnsMean.featLen() == 0)
    trnsMean.resize(transformer->featLen(), transformer->nSubFeat());
  transformer->transform(mean, trnsMean);

  for (UnShrt m = 0; m < _featLen; m++) {
    double trn = trnsMean(m);
    double sqr = trn * trn;
    gsl_vector_set(_vec, m,
		   gsl_vector_get(_vec, m) + sumOsq(m) - 2.0 * trn * sumO(m) + c_k * sqr);
  }
  _occ += c_k;

  _validate(_vec, _occ);
}

double SATVariance::update(NaturalVector var)
{
  assert(var.featLen() == _featLen);

  if (_occ == 0.0) {
    for (UnShrt m = 0; m < _featLen; m++)
      var(m) = 0.0;
    return _occ;
  }

  for (UnShrt m = 0; m < _featLen; m++) {
    double newVar = gsl_vector_get(_vec, m) / _occ;

    if (isnan(newVar))
      throw jnumeric_error("updateSATVar: degenerate variance value; occ = %f", _occ);

    var(m) = newVar;
  }

  return _occ;
}

void SATVariance::fixup(NaturalVector var) {
  for (UnShrt n = 0; n < _featLen; n++)
    var(n) = 1.0 / var(n);
}

void SATVariance::zero() {
  gsl_vector_set_zero(_vec); _occ = 0.0;
}


#if 0

// ----- methods for class `SATMeanFullCovar' -----
//
SATMeanFullCovar::SATFullCovar(UnShrt len) :
  _featLen(len), _mat(gsl_matrix_alloc(_featLen, _featLen)), _occ(0.0) { }

SATMeanFullCovar::~SATFullCovar()
{
  gsl_matrix_free(_mat);
}

void SATMeanFullCovar::accumulate(const TransformBasePtr& transformer, float c_k,
				  const gsl_matrix* invVar, const DNaturalVector& sumO)
{
  assert(transformer->featLen()    == _featLen);
  assert(transformer->orgFeatLen() == _origFeatLen);
  assert(sumO.featLen()            == _featLen);
  assert(invVar->size1()           == _featLen);
  assert(invVar->size2()           == _featLen);

  if (c_k == 0.0) return;

  const TransformMatrix& transformMatrix(transformer->transformMatrix());

  _checkBlocks(transformMatrix.nBlocks());
  for (UnShrt nb = 0; nb < _nBlocks; nb++) {
    UnShrt offsetIndex = nb * _blockLen;
    const gsl_matrix* xformMatrix = transformMatrix.matrix(nb);
    gsl_matrix*    _mat    = _block[nb]._mat;
    gsl_vector*    _vec    = _block[nb]._vec;
    double& _scalar = _block[nb]._scalar;

    // accumulate matrix sufficient statistic
    gsl_matrix_memcpy(temp1, xformMatrix);
    for (UnShrt m = 0; m < _blockLen; m++) {
      float inv = invVar(m + offsetIndex);
      for (UnShrt n = 0; n < _origBlockLen; n++)
	gsl_matrix_set(temp1, m, n, gsl_matrix_get(temp1, m, n) * inv);
    }
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, c_k, temp1, xformMatrix,
		   1.0, _mat);
    
    // accumulate vector sufficient statistic
    const gsl_vector* offset = transformMatrix.offset(nb);
    for (UnShrt m = 0; m < _blockLen; m++)
      gsl_vector_set(vec1, m,
		     sumO(m + offsetIndex) - (c_k * gsl_vector_get(offset,m)));
    gsl_blas_dgemv(CblasTrans, 1.0, temp1, vec1, 1.0, _vec);

    // accumulate scalar sufficient statistic
    for (UnShrt m = 0; m < _blockLen; m++) {
      double vElem = gsl_vector_get(vec1, m);
      _scalar += invVar(m + offsetIndex) * vElem * vElem / c_k;
    }
    _validate(_mat, _vec, _scalar);
  }
  _occ += c_k;
}

#endif


// ----- methods for class `FastMMISAT' -----
//
unsigned FastMMISAT::TtlVarianceComponents         = 0;
unsigned FastMMISAT::TtlNegativeVarianceComponents = 0;

FastMMISAT::FastMMISAT(UnShrt len, UnShrt orgLen, double thr, bool addScalar)
  : FastSAT(len, orgLen, thr),
    _mmiMean(orgLen, 1), _difference(orgLen, 1), _transMean(len, 1), _addScalar(addScalar) { }

FastMMISAT::~FastMMISAT() { }

void FastMMISAT::accumulate(const TransformBasePtr& transformer, double ck, double dk,
			    const InverseVariance& invVar, const NaturalVector& mdlMean)
{
  if (ck == 0.0 && dk == 0.0) return;

  dk *= SATBase::MMISATMultiplier;

  // accumulate mean statistics
  FastSAT::accumulate(transformer, ck, invVar);

  // accumulate D´ contribution
  transformer->transform(mdlMean, _transMean);
  _transMean *= dk;
  SATMean::accumulate(transformer, dk, invVar, _transMean, _addScalar);

  // store speaker transformation for later
  _transList.add(SpkrTransPtr(new SpkrTrans(transformer, dk)));
}

void FastMMISAT::accumulateVk(const TransformBasePtr& transformer0, const TransformBasePtr& transformer,
			      double dk, const InverseVariance& invVar, const NaturalVector& mdlMean)
{
  if (dk == 0.0) return;

  dk *= SATBase::MMISATMultiplier;

  for (UnShrt nb = 0; nb < _nBlocks; nb++) {
    const gsl_vector*    bias         = transformer->transformMatrix().offset(nb);
    const gsl_matrix*    xformMatrix  = transformer->transformMatrix().matrix(nb);
          gsl_vector*    _vec         = _block[nb]._vec;
          double& _scalar      = _block[nb]._scalar;

    // accumulate vector term
    transformer0->transform(mdlMean, _transMean);
    for (UnShrt m = 0; m < _blockLen; m++)
      gsl_vector_set(vec1, m, (_transMean[m] - gsl_vector_get(bias, m)) * invVar[m]);
    gsl_blas_dgemv(CblasTrans, dk, xformMatrix, vec1, 1.0, _vec);

    // accumulate scalar term
    double sum = 0.0;
    for (UnShrt m = 0; m < _blockLen; m++)
      sum += gsl_vector_get(bias, m) * (gsl_vector_get(bias, m) - 2.0 * _transMean[m]) * invVar[m];
    _scalar += dk * sum;
  }
}

void FastMMISAT::_accumulateMMIVar(const NaturalVector& varAcc, double dk)
{
  for (UnShrt nb = 0; nb < _nBlocks; nb++) {
    UnShrt offsetIndex = nb * _blockLen;
    gsl_vector*   varVec      = _block[nb]._varVec;

    for (UnShrt m = 0; m < _blockLen; m++)
      gsl_vector_set(varVec, m,
		     gsl_vector_get(varVec, m) + dk * varAcc(m + offsetIndex));
  }
}

double FastMMISAT::update(NaturalVector satMean, NaturalVector satVar, Info* info,
			  UnShrt* dl)
{
  // solve for new mean
  _mmiMean = satMean;
  double auxFunc = _updateMean(_mmiMean, info, dl);

  if (SATBase::MeanOnly) {
    satMean = _mmiMean;
    for (UnShrt n = 0; n < _featLen; n++)
      satVar(n) = 1.0 / satVar(n);

    return auxFunc;
  }

  // solve for new variance; temporarily store in `_transMean'
  for (UnShrt n = 0; n < _origFeatLen; n++)
    _difference[n] = _mmiMean[n] - satMean[n];
  for (SpkrTransList::Iterator itr(_transList); itr.more(); itr++) {
    const SpkrTransPtr& spkTrans(itr.trans());

    spkTrans->trans()->transform(_difference, _transMean, /* useBias= */ false);
    _transMean.square();
    for (UnShrt n = 0; n < _featLen; n++)
      _transMean[n] += 1.0 / satVar[n];
    _accumulateMMIVar(_transMean, spkTrans->dk());
  }
  _updateVariance(_transMean);

  // copy new mean and variance, checking for negative variance components
  satMean = _mmiMean;
  for (UnShrt n = 0; n < _featLen; n++) {
    TtlVarianceComponents++;
    if (_transMean[n] > 0.0) {
      satVar[n] = _transMean[n];
    } else {
      satVar[n] = 0.0;
      TtlNegativeVarianceComponents++;
    }
  }

  return auxFunc;
}

void FastMMISAT::zeroNoNegativeComponents()
{
  TtlVarianceComponents = TtlNegativeVarianceComponents = 0;
}

void FastMMISAT::announceNoNegativeComponents()
{
  printf("There were %d negative components out of %d total.\n",
	 TtlNegativeVarianceComponents, TtlVarianceComponents);  fflush(stdout);
}


// ----- static variables `SATBase::BaseAcc' -----
//
unsigned SATBase::BaseAcc::_TotalGaussians;
unsigned SATBase::BaseAcc::_TotalUpdatedGaussians;


// ----- methods for class `SATBase::MeanAcc' -----
//
SATMean::UpdateInfo& SATBase::MeanAcc::updateInfo()
{
  static Info _updateInfo;
  return _updateInfo;
}

SATBase::MeanAcc::MeanAcc(GaussDensity pdf, UnShrt meanLen, double thr)
  : BaseAcc(pdf, meanLen),
    _satMean((thr == 0.0) ? new SATMean(_featLen, _origFeatLen) :
	     new MDLSATMean(_featLen, _origFeatLen, thr)) { }

SATBase::MeanAcc::~MeanAcc() { }

void SATBase::MeanAcc::announceAccumulate()
{
  cout << "Accumulating SAT mean statistics ... " << endl;
}

void SATBase::MeanAcc::announceAccumulateDone()
{
  cout << "Done" << endl;
}

void SATBase::MeanAcc::announceUpdate()
{
  _TotalGaussians = _TotalUpdatedGaussians = 0;

  cout << "Updating SAT means ... ";
  updateInfo().zero();
}

void SATBase::MeanAcc::announceUpdateDone()
{
  cout << "Done"       << endl
       << updateInfo() << endl;

  printf("Updated %d of %d total Gaussians.\n",
	 _TotalUpdatedGaussians, _TotalGaussians); fflush(stdout);
}

void SATBase::MeanAcc::accumulate(const TransformerTreePtr& tree)
{
  const TransformBasePtr& transformer(tree->transformer(regClass()));

  float  c_k = postProb();

  _satMean->accumulate(transformer, c_k, invVar(), sumO());

  _increment(c_k);
}

void SATBase::MeanAcc::update()
{
  extendMean(_origFeatLen);

  _TotalGaussians++;

  if (_occ() < UpdateThreshhold) return;

  _TotalUpdatedGaussians++;

  NaturalVector satMean(origMean());
  Info& update(updateInfo());
  _satMean->update(satMean, &update);
}

void SATBase::MeanAcc::zero() { _satMean->zero(); }


// ----- methods for class `SATBase::FastAcc' -----
//
SATMean::UpdateInfo& SATBase::FastAcc::updateInfo()
{
  static Info _updateInfo;
  return _updateInfo;
}

SATBase::FastAcc::FastAcc(GaussDensity pdf, UnShrt meanLen, double thr)
  : BaseAcc(pdf, meanLen), _fastSAT(_featLen, _origFeatLen, thr) { }

void SATBase::FastAcc::zero() { _fastSAT.zero(); }

void SATBase::FastAcc::announceAccumulate()
{
  cout << "Accumulating fast SAT statistics ... " << endl;
}

void SATBase::FastAcc::announceAccumulateDone()
{
  cout << "Done" << endl;
}

void SATBase::FastAcc::announceUpdate()
{
  _TotalGaussians = _TotalUpdatedGaussians = 0;

  cout << "Updating SAT means ... ";
  updateInfo().zero();
}

void SATBase::FastAcc::announceUpdateDone()
{
  cout << "Done"       << endl
       << updateInfo() << endl;

  printf("Updated %d of %d total Gaussians.\n",
	 _TotalUpdatedGaussians, _TotalGaussians); fflush(stdout);
}

void SATBase::FastAcc::accumulate(const TransformerTreePtr& tree)
{
  const TransformBasePtr& transformer(tree->transformer(regClass()));

  float  c_k = postProb();

  _fastSAT.accumulate(transformer, c_k, invVar());

  _increment(c_k);
}

void SATBase::FastAcc::update()
{
  extendMean(_origFeatLen);

  NaturalVector satMean(origMean()), satVar(var());
  UnShrt* dl = descLen();

  _TotalGaussians++;

  if (_occ() < UpdateThreshhold) {
    for (UnShrt n = 0; n < _featLen; n++)
      satVar(n) = 1.0 / satVar(n);

    return;
  }

  _TotalUpdatedGaussians++;

  Info& update(updateInfo());
  _fastSAT.update(satMean, satVar, &update, dl);
}

void SATBase::FastAcc::addFastAcc()
{
  _fastSAT.addFastAcc(fastMean(), fastVar(), fastScalar(), fastNBlks());
}


// ----- methods for class `SATBase::MMIAcc' -----
//
SATMean::UpdateInfo& SATBase::MMIAcc::updateInfo()
{
  static Info _updateInfo;
  return _updateInfo;
}

SATBase::MMIAcc::MMIAcc(GaussDensity pdf, UnShrt meanLen, double thr)
  : BaseAcc(pdf, meanLen), _mmiSAT(_featLen, _origFeatLen, thr) { }

SATBase::MMIAcc::~MMIAcc() { }

void SATBase::MMIAcc::announceAccumulate()
{
  cout << "Accumulating MMI mean and variance statistics ... " << endl;
}

void SATBase::MMIAcc::announceAccumulateDone()
{
  cout << "Done" << endl;
}

void SATBase::MMIAcc::announceUpdate()
{
  cout << "Updating MMI-SAT means and variances ... ";

  updateInfo().zero();
  FastMMISAT::zeroNoNegativeComponents();
}

void SATBase::MMIAcc::announceUpdateDone()
{
  cout << "Done"       << endl
       << updateInfo() << endl;

  printf("Updated %d of %d total Gaussians.\n",
	 _TotalUpdatedGaussians, _TotalGaussians); fflush(stdout);

  FastMMISAT::announceNoNegativeComponents();
}

void SATBase::MMIAcc::accumulate(const TransformerTreePtr& tree)
{
  const TransformBasePtr& transformer(tree->transformer(regClass()));

  _mmiSAT.accumulate(transformer, postProb(), denProb(), invVar(), origMean());

  _increment(numProb());
}

void SATBase::MMIAcc::update()
{
  extendMean(_origFeatLen);

  NaturalVector satMean(origMean()), satVar(var());
  UnShrt* dl = descLen();

  _TotalGaussians++;

  if (_occ() < UpdateThreshhold) {
    for (UnShrt n = 0; n < _featLen; n++)
      satVar(n) = 1.0 / satVar(n);

    return;
  }

  _TotalUpdatedGaussians++;

  Info& update(updateInfo());
  _mmiSAT.update(satMean, satVar, &update, dl);
}

void SATBase::MMIAcc::addFastAcc()
{
  _mmiSAT.addFastAcc(fastMean(), fastVar(), fastScalar(), fastNBlks());
}

void SATBase::MMIAcc::zero()
{
  _zero();
  _mmiSAT.zero();
}


// ----- methods for class `SATBase::VarianceAcc' -----
//
SATBase::VarianceAcc::VarianceAcc(GaussDensity pdf, UnShrt meanLen)
  : BaseAcc(pdf, meanLen), _satVar(_featLen) { }

void SATBase::VarianceAcc::announceAccumulate()
{
  cout << "Accumulating SAT variance statistics ... " << endl;
}

void SATBase::VarianceAcc::announceAccumulateDone()
{
  cout << "Done" << endl;
}

void SATBase::VarianceAcc::announceUpdate()
{
  _TotalGaussians = _TotalUpdatedGaussians = 0;

  cout << "Updating SAT variances ... ";
}

void SATBase::VarianceAcc::announceUpdateDone()
{
  cout << "Done" << endl;

  printf("Updated %d of %d total Gaussians.\n",
   _TotalUpdatedGaussians, _TotalGaussians); fflush(stdout);
}

void SATBase::VarianceAcc::accumulate(const TransformerTreePtr& tree)
{
  const TransformBasePtr& transformer(tree->transformer(regClass()));

  float  c_k = postProb();

  _satVar.accumulate(transformer, c_k, origMean(), sumO(), sumOsq());

  _increment(c_k);
}

void SATBase::VarianceAcc::update()
{
  NaturalVector satVar(var());

  _TotalGaussians++; 

  if (_occ() < UpdateThreshhold) { _satVar.fixup(satVar);  return; }

  _TotalUpdatedGaussians++;

  _satVar.update(satVar);
}

void SATBase::VarianceAcc::zero() { _satVar.zero(); _zero(); }


// ----- methods for class `ClassTotals' -----
//
void ClassTotals::zero() 
{
  for (_ClassTotalsIter itr = _classTotals.begin();
       itr != _classTotals.end(); itr++)
    (*itr).second = 0;
}
   
ostream& operator<<(ostream& os, const ClassTotals& cl)
{
  const ClassTotals::_ClassTotals& totals(cl._classTotals);
   
  os << endl << "Regression Class Totals:" << endl;
  for (ClassTotals::_ConstClassTotalsIter itr = totals.begin();
       itr != totals.end(); itr++)
    os << " Class "
       << setw(4) << (*itr).first  << " : "
       << setw(8) << (*itr).second << endl;

  os << endl;

  return os;
}


// ----- methods for class `SATBase::BaseAcc' -----
//
SATBase::BaseAcc::BaseAcc(GaussDensity pdf, UnShrt meanLen)
  : GaussDensity(pdf), _featLen(pdf.featLen()), _origFeatLen(meanLen),
    _occurences(0.0) { }


// ----- methods for class `SATBase::RegClassAcc' -----
//
UnShrt			SATBase::RegClassAcc::_firstLeaf = 0;
unsigned		SATBase::RegClassAcc::_ttlChanges;
unsigned		SATBase::RegClassAcc::_ttlUpdates;
unsigned		SATBase::RegClassAcc::_maxRegClasses = 0;

SATMean::UpdateInfo& SATBase::RegClassAcc::updateInfo()
{
  static Info _updateInfo;
  return _updateInfo;
}

ClassTotals& SATBase::RegClassAcc::classTotals()
{
  static ClassTotals _classTotals;
  return _classTotals;
}

SATBase::RegClassAcc::
RegClassAcc(GaussDensity pdf, UnShrt meanLen, double thr)
  : BaseAcc(pdf, meanLen), _lhoodThresh(thr) { }

void SATBase::RegClassAcc::announceAccumulate()
{
  cout << "Accumulating SAT mean and regression class statistics ... " << endl;
  _firstLeaf = 0;
}

void SATBase::RegClassAcc::announceAccumulateDone()
{
  cout << "Done" << endl;
}

void SATBase::RegClassAcc::announceUpdate()
{
  _TotalGaussians = _TotalUpdatedGaussians = 0;

  cout << "Updating SAT means and regression classes ... " << endl;

  _ttlChanges = _ttlUpdates = 0;
  classTotals().zero();  updateInfo().zero();
}

void SATBase::RegClassAcc::announceUpdateDone()
{
  cout << "Done" << endl << endl;
  cout << "Total Changes / Total Updates = ( "
       << _ttlChanges << " / " << _ttlUpdates << " )" << endl;

  cout << classTotals() << endl
       << updateInfo()  << endl;

  printf("Updated %d of %d total Gaussians.\n",
   _TotalUpdatedGaussians, _TotalGaussians); fflush(stdout);
}

void SATBase::RegClassAcc::accumulate(const TransformerTreePtr& tree)
{
  bool 	onlyLeaves = true;
  float	c_k        = postProb();

  for (TransformerTree::ConstIterator itr(tree, onlyLeaves); itr.more(); itr++) {
    UnShrt rClass = itr.regClass();

    if (_firstLeaf == 0) _firstLeaf = rClass;

    if (findFastAccX(rClass) == FastAccNotPresent) continue;

    const TransformBasePtr& transformer(itr.transformer());
    FastSATPtr& satMean(_fastSATList[rClass]);
    satMean->accumulate(transformer, c_k, invVar());
  }

  _increment(numProb());
}

// do not update the mean and variance; instead, using
// the current values, assign the best regression class
void SATBase::RegClassAcc::_fixup()
{
  NaturalVector satVar(var());
  for (UnShrt n = 0; n < _featLen; n++)
    satVar(n) = 1.0 / satVar(n);

  if (_occ() == 0.0) { _setClass(_firstLeaf);  return; }

  RCArray rcsarray;
  short  bestClass = -1;
  double bestScore = HUGE;
  NaturalVector satMean(origMean());
  for (FastSATList::Iterator itr(_fastSATList); itr.more(); itr++) {
    double tempScore = itr.acc()->fixup(satMean);

    rcsarray.push_back(RegClassScore(itr.regClass(), tempScore));

    if (tempScore < bestScore) {
      bestClass = itr.regClass();
      bestScore = tempScore;
    }
  }
  if (bestClass == -1)
    throw jconsistency_error("No valid regression classes");

  _setClass(rcsarray);
}

void SATBase::RegClassAcc::update()
{
  extendMean(_origFeatLen);

  _TotalGaussians++;

  UnShrt* dl = descLen();
  if (_occ() < UpdateThreshhold) { _fixup();  return; }

  _TotalUpdatedGaussians++;

  // solve for the best mean and variance for all tabulated
  // regression classes, keep the best of these
  bool deepCopy = true;
  NaturalVector satMean(origMean()), satVar(var());
  NaturalVector initMean(satMean, deepCopy), initVar(satVar, deepCopy);

  short  bestClass = -1;
  double bestScore = HUGE;
  Info   bestInfo;
  RCArray rcsarray;

  UnShrt descLen[_nSubFeat], tempDescLen[_nSubFeat];

  for (int blkX = 0; blkX < _nSubFeat; blkX++)
    descLen[blkX] = dl[blkX];

  for (FastSATList::Iterator itr(_fastSATList); itr.more(); itr++) {
    Info info;
    NaturalVector tempMean(initMean, deepCopy);
    NaturalVector tempVar(initVar,   deepCopy);

    for (int blkX = 0; blkX < _nSubFeat; blkX++)
      tempDescLen[blkX] = descLen[blkX];

    double tempScore = itr.acc()->update(tempMean, tempVar, &info, tempDescLen);

    rcsarray.push_back(RegClassScore(itr.regClass(), tempScore));
    if (tempScore < bestScore) {
      bestClass = itr.regClass();  bestScore = tempScore;  bestInfo = info;
      satMean   = tempMean;        satVar    = tempVar;

      for (int blkX = 0; blkX < _nSubFeat; blkX++)
	dl[blkX] = tempDescLen[blkX];
    }
  }
  if (bestClass == -1)
    throw jconsistency_error("No valid regression classes");

  updateInfo() += bestInfo;

  _setClass(rcsarray);
}

void SATBase::RegClassAcc::addFastAcc()
{
  for (int accX = 0; accX < noRegClasses(); accX++) {
    UnShrt rClass = regClass(accX);

    if (accX >= fastAccu()->subN())
      throw jindex_error("Accumulator index (%d) greater than number of accumulators (%d).",
			 accX, fastAccu()->subN());

    FastSATPtr& fsat(_fastSATList.find(rClass, _featLen, _origFeatLen, _lhoodThresh));
    fsat->addFastAcc(fastMean(accX), fastVar(accX), fastScalar(accX), fastNBlks());
  }
}

void SATBase::RegClassAcc::_setClass(short cl)
{
  if (!isAncestor(regClass(), cl))	// don't count as a change if current
    _ttlChanges++;			// class is an ancestor of new class

  setRegClass(cl);  classTotals()[cl]++;
  _ttlUpdates++;
}

// sort the regression classes according to their scores and
// retain only the top `_maxRegClasses'
//
void SATBase::RegClassAcc::_setClass(RCArray& rcsarray)
{
  sort(rcsarray.begin(), rcsarray.end(), LessThan());
  if ((_maxRegClasses > 0) && (rcsarray.size() > _maxRegClasses))
    rcsarray.erase(rcsarray.begin() + _maxRegClasses, rcsarray.end());

  UnShrt cl = rcsarray[0]._regClass;
  if (!isAncestor(regClass(), cl))	// don't count as a change if current
    _ttlChanges++;			// class is an ancestor of new class

  classTotals()[cl]++;  _ttlUpdates++;

  setRegClasses(rcsarray);
}

void SATBase::RegClassAcc::zero()
{
  for (FastSATList::Iterator itr(_fastSATList); itr.more(); itr++)
    itr.acc()->zero();
}

SATBase::RegClassAcc::FastSATList::~FastSATList()
{
  _fastSATList.erase(_fastSATList.begin(), _fastSATList.end());
}

FastSATPtr& SATBase::RegClassAcc::FastSATList::operator[](UnShrt rc)
{
  _FastSATListIter itr = _fastSATList.find(rc);
  if (itr == _fastSATList.end())
    throw jkey_error("Could not find mean accumulator for regression class %d.", rc);
  return (*itr).second;
}

FastSATPtr& SATBase::RegClassAcc::FastSATList::
find(UnShrt rc, UnShrt fl, UnShrt ol, double pl)
{
  _FastSATListIter itr = _fastSATList.find(rc);
  if (itr != _fastSATList.end()) return (*itr).second;

  FastSATPtr ptr(new FastSAT(fl, ol, pl));
  _fastSATList.insert(_ValueType(rc, ptr));

  itr = _fastSATList.find(rc);

  return (*itr).second;
}


// ----- methods for class `SATBase::MMIRegClassAcc' -----
//
UnShrt			SATBase::MMIRegClassAcc::_firstLeaf = 0;
unsigned		SATBase::MMIRegClassAcc::_ttlChanges;
unsigned		SATBase::MMIRegClassAcc::_ttlUpdates;
unsigned		SATBase::MMIRegClassAcc::_maxRegClasses = 0;

SATMean::UpdateInfo& SATBase::MMIRegClassAcc::updateInfo()
{
  static Info _updateInfo;
  return _updateInfo;
}

ClassTotals& SATBase::MMIRegClassAcc::classTotals()
{
  static ClassTotals _classTotals;
  return _classTotals;
}

SATBase::MMIRegClassAcc::
MMIRegClassAcc(GaussDensity pdf, UnShrt meanLen, double thr)
  : BaseAcc(pdf, meanLen), _lhoodThresh(thr) { }

void SATBase::MMIRegClassAcc::announceAccumulate()
{
  cout << "Accumulating SAT mean and regression class statistics ... " << endl;
  _firstLeaf = 0;
}

void SATBase::MMIRegClassAcc::announceAccumulateDone()
{
  cout << "Done" << endl;
}

void SATBase::MMIRegClassAcc::announceUpdate()
{
  _TotalGaussians = _TotalUpdatedGaussians = 0;

  cout << "Updating SAT means and regression classes ... " << endl;

  _ttlChanges = _ttlUpdates = 0;
  classTotals().zero();  updateInfo().zero();
}

void SATBase::MMIRegClassAcc::announceUpdateDone()
{
  cout << "Done" << endl << endl;
  cout << "Total Changes / Total Updates = ( "
       << _ttlChanges << " / " << _ttlUpdates << " )" << endl;

  cout << classTotals() << endl
       << updateInfo()  << endl;

  printf("Updated %d of %d total Gaussians.\n",
   _TotalUpdatedGaussians, _TotalGaussians); fflush(stdout);
}

void SATBase::MMIRegClassAcc::accumulate(const TransformerTreePtr& tree)
{
  bool 	onlyLeaves = true;
  float	c_k        = postProb();

  const TransformBasePtr transformer0(tree->transformer(regClass()));
  for (TransformerTree::ConstIterator itr(tree, onlyLeaves); itr.more(); itr++) {
    UnShrt rClass = itr.regClass();

    if (_firstLeaf == 0) _firstLeaf = rClass;

    if (findFastAccX(rClass) == FastAccNotPresent) continue;

    const TransformBasePtr transformer(itr.transformer());
    FastMMISATPtr satMean(_fastSATList[rClass]);
    satMean->accumulate(transformer, postProb(), denProb(), invVar(), origMean());
    satMean->accumulateVk(transformer0, transformer, denProb(), invVar(), origMean());
  }

  _increment(numProb());
}

// do not update the mean and variance; instead, using
// the current values, assign the best regression class
void SATBase::MMIRegClassAcc::_fixup()
{
  NaturalVector satVar(var());
  for (UnShrt n = 0; n < _featLen; n++)
    satVar(n) = 1.0 / satVar(n);

  if (_occ() == 0.0) { _setClass(_firstLeaf);  return; }

  RCArray rcsarray;
  short  bestClass = -1;
  double bestScore = HUGE;
  NaturalVector satMean(origMean());
  for (FastSATList::Iterator itr(_fastSATList); itr.more(); itr++) {
    double tempScore = itr.acc()->fixup(satMean);

    rcsarray.push_back(RegClassScore(itr.regClass(), tempScore));

    if (tempScore < bestScore) {
      bestClass = itr.regClass();
      bestScore = tempScore;
    }
  }
  if (bestClass == -1)
    throw jconsistency_error("No valid regression classes");

  _setClass(rcsarray);
}

void SATBase::MMIRegClassAcc::update()
{
  extendMean(_origFeatLen);

  _TotalGaussians++;

  UnShrt* dl = descLen();
  if (_occ() < UpdateThreshhold) { _fixup();  return; }

  _TotalUpdatedGaussians++;

  // solve for the best mean and variance for all tabulated
  // regression classes, keep the best of these
  bool deepCopy = true;
  NaturalVector satMean(origMean()), satVar(var());
  NaturalVector initMean(satMean, deepCopy), initVar(satVar, deepCopy);

  short  bestClass = -1;
  double bestScore = HUGE;
  Info   bestInfo;
  RCArray rcsarray;

  UnShrt descLen[_nSubFeat], tempDescLen[_nSubFeat];

  for (int blkX = 0; blkX < _nSubFeat; blkX++)
    descLen[blkX] = dl[blkX];

  for (FastSATList::Iterator itr(_fastSATList); itr.more(); itr++) {
    Info info;
    NaturalVector tempMean(initMean, deepCopy);
    NaturalVector tempVar(initVar,   deepCopy);

    for (int blkX = 0; blkX < _nSubFeat; blkX++)
      tempDescLen[blkX] = descLen[blkX];

    double tempScore = itr.acc()->update(tempMean, tempVar, &info, tempDescLen);

    rcsarray.push_back(RegClassScore(itr.regClass(), tempScore));
    if (tempScore < bestScore) {
      bestClass = itr.regClass();  bestScore = tempScore;  bestInfo = info;
      satMean   = tempMean;        satVar    = tempVar;

      for (int blkX = 0; blkX < _nSubFeat; blkX++)
	dl[blkX] = tempDescLen[blkX];
    }
  }
  if (bestClass == -1)
    throw jconsistency_error("No valid regression classes");

  updateInfo() += bestInfo;

  _setClass(rcsarray);
}

void SATBase::MMIRegClassAcc::addFastAcc()
{
  for (int accX = 0; accX < noRegClasses(); accX++) {
    UnShrt rClass = regClass(accX);

    if (accX >= fastAccu()->subN())
      throw jindex_error("Accumulator index (%d) greater than number of accumulators (%d).",
			 accX, fastAccu()->subN());

    FastMMISATPtr& fsat(_fastSATList.find(rClass, _featLen, _origFeatLen, _lhoodThresh));
    fsat->addFastAcc(fastMean(accX), fastVar(accX), fastScalar(accX), fastNBlks());
  }
}

void SATBase::MMIRegClassAcc::_setClass(short cl)
{
  if (!isAncestor(regClass(), cl))	// don't count as a change if current
    _ttlChanges++;			// class is an ancestor of new class

  setRegClass(cl);  classTotals()[cl]++;
  _ttlUpdates++;
}

// sort the regression classes according to their scores and
// retain only the top `_maxRegClasses'
//
void SATBase::MMIRegClassAcc::_setClass(RCArray& rcsarray)
{
  sort(rcsarray.begin(), rcsarray.end(), LessThan());
  if ((_maxRegClasses > 0) && (rcsarray.size() > _maxRegClasses))
    rcsarray.erase(rcsarray.begin() + _maxRegClasses, rcsarray.end());

  UnShrt cl = rcsarray[0]._regClass;
  if (!isAncestor(regClass(), cl))	// don't count as a change if current
    _ttlChanges++;			// class is an ancestor of new class

  classTotals()[cl]++;  _ttlUpdates++;

  setRegClasses(rcsarray);
}

void SATBase::MMIRegClassAcc::zero()
{
  for (FastSATList::Iterator itr(_fastSATList); itr.more(); itr++)
    itr.acc()->zero();
}

SATBase::MMIRegClassAcc::FastSATList::~FastSATList()
{
  _fastSATList.erase(_fastSATList.begin(), _fastSATList.end());
}

FastMMISATPtr& SATBase::MMIRegClassAcc::FastSATList::operator[](UnShrt rc)
{
  _FastSATListIter itr = _fastSATList.find(rc);
  if (itr == _fastSATList.end())
    throw jkey_error("Could not find mean accumulator for regression class %d.", rc);
  return (*itr).second;
}

FastMMISATPtr& SATBase::MMIRegClassAcc::FastSATList::
find(UnShrt rc, UnShrt fl, UnShrt ol, double pl)
{
  _FastSATListIter itr = _fastSATList.find(rc);
  if (itr != _fastSATList.end()) return (*itr).second;

  FastMMISATPtr ptr(new FastMMISAT(fl, ol, pl, /*addScalar=*/ false));
  _fastSATList.insert(_ValueType(rc, ptr));

  itr = _fastSATList.find(rc);

  return (*itr).second;
}


// ----- methods for class `SATBase' -----
//
double SATBase::LogLhoodThreshhold;
double SATBase::UpdateThreshhold;
double SATBase::MMISATMultiplier;
bool   SATBase::MeanOnly = false;

SATBase::SATBase(CodebookSetFastSATPtr& cbs, UnShrt meanLen,
		 double lhoodThreshhold, double massThreshhold, double multiplier,
		 unsigned nParts, unsigned part, bool meanOnly)
  : _cbs(cbs), _origFeatLen(_setFeatLen(meanLen)), _nParts(nParts), _part(part)
{
  if (massThreshhold == 0.0) massThreshhold = 10.0;

  LogLhoodThreshhold = lhoodThreshhold;
  UpdateThreshhold   = massThreshhold;
  MMISATMultiplier   = multiplier;
  MeanOnly           = meanOnly;

  cout << endl << "Using likelihood threshold of " << LogLhoodThreshhold << "." << endl;
  cout << endl << "Using update threshold of "     << UpdateThreshhold   << "." << endl;
  cout << endl << "Using MMI-SAT multiplier of "   << MMISATMultiplier   << "." << endl;

  cout << "Updating part " << part << " of " << nParts << endl;
}

SATBase::~SATBase() { }

UnShrt SATBase::_setFeatLen(UnShrt meanLen)
{
  if (meanLen == 0)
    meanLen = orgFeatLen();

  if (meanLen % nSubFeat() != 0)
    throw jdimension_error("Mean length %d is inconsistent with %d sub-features",
			   meanLen, nSubFeat());

  return meanLen;
}

void SATBase::zero()
{
  for (AccList::Iterator itr(_accList); itr.more(); itr++)
    itr.accu()->zero();
}

double SATBase::
estimate(const SpeakerListPtr& sList, const String& spkrAdaptParms, const String& meanVarsStats)
{
  int       totalT        = 0;
  LogDouble totalPr       = LogLhoodThreshhold * cbs()->descLength();
  int       orgSubFeatLen = _origFeatLen / cbs()->cepNSubFeat();

  cout << "Performing ML-SAT re-estimation." << endl;

  announceAccumulate();
  cbs()->zeroAccus(_nParts, _part);
  unsigned counter = 0;
  float    factor  = 1.0;
  printf("\n");  fflush(stdout);
  for (SpeakerList::Iterator itr(sList); itr.more(); itr++) {
    printf("    %4d. %s\n", ++counter, itr.spkr().c_str());  fflush(stdout);

    float tmpFlt;
    unsigned tmpInt;
    String accumFileName(meanVarsStats); accumFileName += itr.spkr(); accumFileName += ".cbs.accu"; 

    cbs()->loadAccus(accumFileName, tmpFlt, tmpInt, factor, _nParts, _part);

    totalPr += LogDouble(tmpFlt);
    totalT  += tmpInt;

    TransformerTreeList treeList(cbs(), spkrAdaptParms, orgSubFeatLen);

    const TransformerTreePtr tree(treeList.getTree(itr.spkr()));

    accumulate(tree);
    cbs()->zeroAccus(_nParts, _part);
  }
  announceAccumulateDone();
  update();

  return totalPr / totalT;
}

void SATBase::accumulate(const TransformerTreePtr& tree)
{
  for (AccList::Iterator itr(_accList); itr.more(); itr++)
    itr.accu()->accumulate(tree);
}

void SATBase::update()
{
  announceUpdate();

  for (AccList::Iterator itr(_accList); itr.more(); itr++)
    itr.accu()->update();

  announceUpdateDone();
}


// ----- methods for class `SATBase::AccList' -----
//
SATBase::AccList::~AccList()
{
  _list.erase(_list.begin(), _list.end());
}


// ----- methods for class `TransformerTreeList' -----
//
TransformerTreeList::TransformerTreeList(CodebookSetFastSATPtr& cb, const String& parmFile,
					 UnShrt orgSubFeatLen, const SpeakerList* spkList)
  : _cbs(cb), _parmFile(parmFile),
    _orgSubFeatLen((orgSubFeatLen == 0) ? _cbs->orgSubFeatLen() : orgSubFeatLen)
{
  if (spkList == NULL) return;

  for (SpeakerList::Iterator itr(*spkList); itr.more(); itr++) {
    String spkrLabel(itr.spkr());
    String fileName(_parmFile + "." +  spkrLabel);
    _treeList.insert(_ValueType(spkrLabel, TransformerTreePtr(new TransformerTree(_cbs, fileName, _orgSubFeatLen))));
  }
}

TransformerTreeList::~TransformerTreeList() { }

const TransformerTreePtr& TransformerTreeList::getTree(const String& spkrLabel)
{
  _TreeListIter itr = _treeList.find(spkrLabel);

  if (itr != _treeList.end()) return (*itr).second;

  String fileName(_parmFile + spkrLabel);

  _treeList.insert(_ValueType(spkrLabel, TransformerTreePtr(new TransformerTree(_cbs, fileName, _orgSubFeatLen))));

  itr = _treeList.find(spkrLabel);
  return (*itr).second;
}

const TransformerTreePtr& TransformerTreeList::getTree(const String& spkrLabel) const
{
  _TreeListIter itr = _treeList.find(spkrLabel);

  if (itr == _treeList.end())
    throw jkey_error("Could not find adaptation parameters for speaker %s.\n",
		     spkrLabel.chars());

  return (*itr).second;
}
