//
//                                Millennium
//                    Distant Speech Recognition System
//                                  (dsr)
//
//  Module:  asr.gaussian
//  Purpose: Basic acoustic likelihood computation.
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
#include "gaussian/codebookBasic.h"
#include "common/mach_ind_io.h"
#include <gsl/gsl_blas.h>

// ----- methods for class `CodebookBasic' -----
//
CodebookBasic::CodebookBasic(const String& nm, UnShrt rfN, UnShrt dmN,
			     CovType cvTp, VectorFloatFeatureStreamPtr feat)
  : _name(nm), _refN(rfN), _dimN(dmN), _orgDimN(dmN),
    _covType(cvTp), _rv(NULL), _cv(NULL), _determinant(NULL), _count(NULL),
    _regClass(NULL), _descLen(NULL), _nSubFeat(0), _feature(feat),
    _scale(1.0), _cache(NULL), _scoreFunc(&CodebookBasic::_scoreOpt)
{
  if (_feature.isNull() == false && _feature->size() != _dimN)
    throw jdimension_error("Feature and mean lengths (%d vs. %d) do not match.",
			   _feature->size(), _dimN);

  _allocRV();  _allocCovar();  _allocCount();
}

CodebookBasic::CodebookBasic(const String& nm, UnShrt rfN, UnShrt dmN,
			     CovType cvTp, const String& featureName)
  : _name(nm), _featureName(featureName), _refN(rfN), _dimN(dmN), _orgDimN(dmN),
    _covType(cvTp), _rv(NULL), _cv(NULL), _determinant(NULL), _count(NULL),
    _regClass(NULL), _descLen(NULL), _nSubFeat(0), _feature(NULL),
    _scale(1.0), _cache(NULL), _scoreFunc(&CodebookBasic::_scoreOpt)
{
  _allocRV();  _allocCovar();  _allocCount();
}

CodebookBasic::~CodebookBasic()
{
  // cout << "Deleting codebook " << name() << endl;
  _dealloc();
}

void CodebookBasic::setFeature(FeatureSetPtr& featureSet)
{
  _feature = featureSet->feature(_featureName);
}

void CodebookBasic::_dealloc()
{
  // printf("Deallocating codebook %s\n", _name.c_str());

  if (_rv) {
    gsl_matrix_float_free(_rv);  _rv = NULL;
  }
  if (_cv) {
    for (UnShrt refX = 0; refX < _refN; refX++)
      gsl_vector_float_free(_cv[refX]);
    delete[] _cv;  _cv = NULL;
  }

  delete[] _determinant;  _determinant = NULL;

  delete[] _count;  _count = NULL;

  if (_regClass) {
    for (UnShrt refX = 0; refX < _refN; refX++)
      free(_regClass[refX]);
    delete[] _regClass;  _regClass = NULL;
  }
  if (_descLen) {
    for (UnShrt refX = 0; refX < _refN; refX++)
      free(_descLen[refX]);
    delete[] _descLen;  _descLen = NULL;
  }
}

String CodebookBasic::puts() const
{
  static char buffer[500];

  sprintf(buffer, "Codebook '%s' : %d Dimensions : %d Gaussians",
	  _name.c_str(), _dimN, _refN);

  return String(buffer);
}

void CodebookBasic::setScoreAll(unsigned cacheN)
{
  if (cacheN > 0) _cache = new Cache(_refN, cacheN);
  _scoreFunc = &CodebookBasic::_scoreAll;
}

void CodebookBasic::applySTC(const gsl_matrix_float* trans)
{
  gsl_matrix_float* rv = gsl_matrix_float_alloc(_rv->size1, _rv->size2);
  gsl_blas_sgemm(CblasNoTrans, CblasTrans, 1.0, _rv, trans, 0.0, rv);

  gsl_matrix_float_memcpy(_rv, rv);
  gsl_matrix_float_free(rv);
}

void CodebookBasic::_allocCount()
{
  if (_count != NULL) delete[] _count;

  _count = new float[_refN];

  if (_count == NULL)
    throw j_error("Allocation error for codebook counts.");
  
  for (UnShrt i = 0; i < _refN; i++) _count[i] = 0.0;
}

void CodebookBasic::_allocRV()
{
  // printf("Allocating means for codebook %s\n", _name.c_str());

  if (_rv != NULL && (_refN != _rv->size1 || _orgDimN != _rv->size2)) {
    gsl_matrix_float_free(_rv);  _rv = NULL;
  }

  if (_rv == NULL) {
      _rv = gsl_matrix_float_alloc(_refN, _orgDimN);
      gsl_matrix_float_set_zero(_rv);
  }

  if (_rv == NULL)
    throw j_error("Allocation error for codebook.");
}

void CodebookBasic::_allocCovar()
{
  // printf("Allocating covariances for codebook %s\n", _name.c_str());

  if (_cv == NULL) {
    _cv = new gsl_vector_float*[_refN];

    for (UnShrt refX = 0; refX < _refN; refX++) _cv[refX] = NULL;
  }

  for (UnShrt refX = 0; refX < _refN; refX++) {
    if (_cv[refX] != NULL && _cv[refX]->size != _dimN) {
      gsl_vector_float_free(_cv[refX]); _cv[refX] = NULL;
    }
    if (_cv[refX] == NULL) {
      _cv[refX] = gsl_vector_float_alloc(_dimN);
      for (UnShrt dimX = 0; dimX < _dimN; dimX++)
	  gsl_vector_float_set(_cv[refX], dimX, 1.0);
    }
  }
  _pi = log(2.0 * M_PI) * _dimN;

  if (_determinant == NULL) {
    _determinant = new float[_refN];
    for (UnShrt dimX = 0; dimX < _refN; dimX++)
	_determinant[dimX] = 0.0;
  }
}

void CodebookBasic::_allocRegClass()
{
  if (_regClass != NULL) return;

  _regClass = new UnShrt*[_refN];

  for (UnShrt i = 0; i < _refN; i++)
    _regClass[i] = NULL;
}

void CodebookBasic::_allocDescLength()
{
  if (_descLen == NULL) {
    _descLen = new UnShrt*[_refN];

    for (UnShrt i = 0; i < _refN; i++)
      _descLen[i] = NULL;
  }

  for (UnShrt i = 0; i < _refN; i++)
    _descLen[i] = 
      (UnShrt*) realloc(_descLen[i], _nSubFeat * sizeof(UnShrt));
}


void CodebookBasic::_allocCb()
{
  if (_rv && _cv && _count) return;

  _allocRV();
  _allocCount();
  _allocCovar();
}

void CodebookBasic::setRegClasses(UnShrt c)
{
  _allocRegClass();

  UnShrt size = 1;
  for (UnShrt refX = 0; refX < _refN; refX++) {
    _regClass[refX] = (UnShrt*) realloc(_regClass[refX], (size + 1) * sizeof(UnShrt));
    _regClass[refX][0] = 1;
    _regClass[refX][1] = c;
  }  
}

const int CodebookBasic::CodebookMagic = 64207531;
const int CodebookBasic::MarkerMagic   = 123456789;

// dumpMarker: dump a marker into file fp
void CodebookBasic::_dumpMarker(FILE* fp)
{
  write_int(fp, MarkerMagic);
}

// _checkMarker: check if file fp has a marker next
void CodebookBasic::_checkMarker(FILE* fp)
{
  if (read_int(fp) != MarkerMagic)
    throw j_error("CheckMarker: Marker expected in codebook file.");
}

void CodebookBasic::_loadCovariance(unsigned refX, FILE* fp)
{
  for (unsigned i = 0; i < _dimN; i++)
    gsl_vector_float_set(_cv[refX], i, read_float(fp));
  _determinant[refX] = read_float(fp);
}

void CodebookBasic::_saveCovariance(unsigned refX, FILE* fp) const
{
  for (unsigned i = 0; i < _dimN; i++)
    write_float(fp, gsl_vector_float_get(_cv[refX], i));
  write_float(fp, _determinant[refX]);
}


static const int MaxCpn = 60;

void CodebookBasic::load(FILE* fp, bool readName)
{
  if (fp==NULL) throw j_error("NULL file pointer");

  static const unsigned MaxNameLength = 200;
  static char name[MaxNameLength];
  if (readName)
    read_string(fp, name);

  _refN               = read_int(fp);
  _dimN               = read_int(fp);
  _orgDimN            = read_int(fp);
  _nSubFeat           = read_int(fp);
  int ctype           = read_int(fp);
  _covType            = (ctype == -1) ? COV_DIAGONAL : (CovType) ctype;

  int regClassPresent = read_int(fp);
  int descLenPresent  = read_int(fp);

  _allocCount();  _allocRV();  _allocCovar();

  for (UnShrt refX = 0; refX < _refN; refX++) {
    _count[refX] = read_float(fp);
    for (UnShrt dimX = 0; dimX < _orgDimN; dimX++)
      gsl_matrix_float_set(_rv, refX, dimX, read_float(fp));
    _loadCovariance(refX, fp);
  }
  _pi = log(2.0 * M_PI) * _dimN;

  // read regression class assigments of Gaussians
  if (regClassPresent) {
    _allocRegClass();
    for (UnShrt refX = 0; refX < _refN; refX++) {
      UnShrt noRegClass = (UnShrt) read_short(fp);
      _regClass[refX] =
	(UnShrt*) realloc(_regClass[refX], sizeof(UnShrt) * (noRegClass + 1));
      _regClass[refX][0] = noRegClass;
      for (UnShrt clsX = 1; clsX <= noRegClass; clsX++)
	_regClass[refX][clsX] = (UnShrt) read_short(fp);
    }
  }

  // read description lengths for each Gaussian component
  if (descLenPresent) {
    _allocDescLength();
    for (UnShrt refX = 0; refX < _refN; refX++)
      for (UnShrt blkX = 0; blkX < _nSubFeat; blkX++)
	_descLen[refX][blkX] = (UnShrt) read_short(fp);
  }

  _checkMarker(fp);  
}

void CodebookBasic::loadOld(FILE* fp)
{
  if (fp==NULL) throw j_error("NULL file pointer");

  int l_refN = read_int(fp);		// read and compare size of codebook

  if (l_refN != ((int) refN()))
    throw jdimension_error("Size of %s is %d, expected %d (not loaded!)\n",
			   name().chars(), l_refN, ((int) refN()));

  int l_dimN = read_int(fp);		// read and compare size of means
  if (l_dimN != ((int) featLen()))
    throw jdimension_error("Dimension of %s is %d, expected %d (not loaded!)\n",
			   name().chars(), l_dimN, ((int) featLen()));

  int l_cov = read_int(fp);		// read and compare covariance type

  for (UnShrt iref = 0; iref < refN(); iref++) {
      if (l_cov==(-1)) _count[iref] = read_float(fp);
	          else _count[iref] = 0;

      					// read the reference vector
      for (UnShrt idim = 0; idim < featLen(); idim++) 	gsl_matrix_float_set(_rv, iref, idim, read_float(fp));

					// read covariance matrix

					// no uniform codebook covar type,
					// then read the type from file
      int thisCov = l_cov==(-1) ? read_int(fp) : l_cov;	

					// not the same type as in memory,
					// exit with error
      if (thisCov != COV_DIAGONAL)
	throw jio_error("Wrong covariance type.");

      _loadCovariance(iref, fp);	// load the covariance

      _pi=(float)log(2.0*M_PI)* (float) l_dimN;
  }
}

void CodebookBasic::save(FILE *fp, bool janusFormat) const
{
  bool regClassPresent = (_regClass == NULL) ? false : true;
  bool descLenPresent  = (_descLen  == NULL) ? false : true;

  write_string(fp, _name.chars());
  write_int(fp,    _refN);
  write_int(fp,    _dimN);

  if (janusFormat == false) {
    write_int(fp,    _orgDimN);
    write_int(fp,    _nSubFeat);
  }

  write_int(fp,    (int) _covType);

  if (janusFormat == false) {
    write_int(fp,    regClassPresent);
    write_int(fp,    descLenPresent);
  }

  // write mean vectors and covariance matrices of all Gaussians
  for (UnShrt refX = 0; refX < _refN; refX++) {

    if (janusFormat == false)
      write_float(fp, _count[refX]);

    for (UnShrt dimX = 0; dimX < _rv->size2; dimX++)
      write_float(fp, gsl_matrix_float_get(_rv, refX, dimX));

    _saveCovariance(refX, fp);
  }

  if (janusFormat) return;

  // write regression class assignments of Gaussians
  if (regClassPresent) {
    for (UnShrt refX = 0; refX < _refN; refX++) {
      UnShrt noRegClass = _regClass[refX][0];
      write_short(fp, noRegClass);
      for (UnShrt clsX = 1; clsX <= noRegClass; clsX++)
	write_short(fp, (short) _regClass[refX][clsX]);
    }
  }

  // write description lengths of Gaussians
  if (descLenPresent) {
    for (UnShrt refX = 0; refX < _refN; refX++)
      for (UnShrt blkX = 0; blkX < _nSubFeat; blkX++)
	write_short(fp, (short) _descLen[refX][blkX]);
  }

  _dumpMarker(fp);
}

void CodebookBasic::write(FILE *fp) const
{
  // printf("Writing description for codebook %s\n", _name.c_str());

  fprintf(fp, "%-25s%-20s%-10d%-3d%-10s\n", _name.c_str(), _feature->name().c_str(), _refN, _dimN, "DIAGONAL");
}

void CodebookBasic::resetCache()
{
  _frameX = _minDistIdx = -1;
  _score = 0.0;

  if (_cache.isNull() == false) _cache->reset();
}

void CodebookBasic::resetFeature()
{
  _feature->reset();
}

UnShrt CodebookBasic::RefMax = 256;
CBX*   CodebookBasic::tmpI = NULL;
int    CodebookBasic::tmpN = 0;

float CodebookBasic::_scoreOpt(int frameX, const float* val, float* addCount)
{
  if (frameX == _frameX) {
    if (addCount) {
      for (unsigned refX = 0; refX < _refN; refX++)
	addCount[refX] = 0.0;
      addCount[_minDistIdx] = 1.0;
    }
    return _score;
  }

  int dimN_4  = _dimN / 4;

#ifdef __ICC
  float     sa[256]; /* NOTE: This is hardcoded, because otherwise ICC would
			not VECTORIZE the loop which is very bad for speed! */
#endif

#ifdef __ICC
  int            minDistIdx, bestI;
  float          minDistSum;
#else
  /*fast access*/
  register int   minDistIdx, bestI;
  register float minDistSum;
#endif

  // allocate and initialize tmp index array
  if (tmpN < RefMax) {
    tmpN = RefMax;
    tmpI = (CBX*) malloc (RefMax * sizeof(CBX));
    for (UnShrt i = 0; i < RefMax; i++) tmpI[i] = i;
  }
 
  float* pattern = _feature->next(frameX)->data;

  // get bbi tree if available
  int  bbiN;
  CBX* bbilist;

  bbiN    = refN();
  bbilist = tmpI;

  // Compute all the Mahalanobis distances for the subset of Gaussians
  minDistSum = 1E20;
  minDistIdx = 0;
  bestI      = 0;

  for (int i = 0; i < bbiN; i++) {     
    int     refX     =  bbilist[i], dimX;
#ifdef __ICC
    float   distSum  =  _pi + _cv[refX]->det;
    int     dimS     =  8, j, dimG;
    float   *restrict pt      =  pattern;
    float   *restrict rv      =  _rv->data[refX * _rv->size2];
    float   *restrict cv      =  _cv[refX]->m.d;
#else
    register float   distSum  =  _pi + _determinant[refX];
    float   *pt      =  pattern;
    float   *rv      =  _rv->data + (refX * _rv->size2);
    float   *cv      =  _cv[refX]->data;
#endif

    /* If we wanted to use Intel Performance Primitives (which we don't want to do):
    Ipp32f* pSrc      = (Ipp32f*) pattern;
    Ipp32f* pSrcMean  = (Ipp32f*) cbP->rv->matPA[refX];
    Ipp32f* pSrcVar   = (Ipp32f*) cbP->cv[refX]->m.d;
    Ipp32f* pResult   = (Ipp32f*) &distSum;
    IppStatus     st  = ippsMahDistSingle_32f (pSrc, pSrcMean, pSrcVar, dimN, pResult);
    distSum          += cbP->pi + cbP->cv[refX]->det; */

#ifdef __ICC
    // This works best with the Intel compiler
    for (int j = 0, dimG = (dimN < 16) ? dimN : 16; j < dimG; j++)
      sa[j] = rv[j] - pt[j], distSum += sa[j]*sa[j]*cv[j];
    for (dimX = 16, dimG = 24; distSum < minDistSum && dimX < dimN; dimX += dimS, dimG += dimS) {
      if (dimG > dimN) dimG = dimN;
      for (j = dimX; j < dimG; j++)
	sa[j] = rv[j] - pt[j], distSum += sa[j]*sa[j]*cv[j];
    }
#else
    // Original version (register: distSum & fast access, no restrict)
    //
    for (dimX = 0; dimX < dimN_4; dimX++) {
      register float diff0;
      if (distSum > minDistSum) break;
      diff0 = *rv++ - *pt++;
      distSum += diff0*diff0*(*cv++);
      diff0 = *rv++ - *pt++;
      distSum += diff0*diff0*(*cv++);
      diff0 = *rv++ - *pt++;
      distSum += diff0*diff0*(*cv++);
      diff0 = *rv++ - *pt++;
      distSum += diff0*diff0*(*cv++);
    }
    if (dimX == dimN_4) {
      for (dimX = 4*dimN_4; dimX < _dimN; dimX++) {
	register float diff0 = *rv++ - *pt++;
	distSum += diff0*diff0*(*cv++);
      }
    }
#endif

    if (distSum < minDistSum) {
      minDistSum = distSum;
      minDistIdx = refX;
      bestI      = i;
    }
  }

  _score  = 0.5 * (minDistSum + 2 * val[minDistIdx]);
  _frameX = frameX;  _minDistIdx = minDistIdx;

  if (_scale != 1.0) _score *= _scale;

  if (addCount) {
    for (unsigned refX = 0; refX < _refN; refX++)
      addCount[refX] = 0.0;
    addCount[minDistIdx] = 1.0;
  }

  // assume a fully continuous system
  return _score;
}

// simplified log-likelihood calculation
LogLhoodIndexPtr CodebookBasic::logLhood(const gsl_vector* frame, float* val) const
{
  int dimN_4  = _dimN / 4;

  static float* pattern = NULL;
  if (pattern == NULL)
    pattern = new float[featLen()];
  for (int idim = 0; idim < featLen(); idim++)
    pattern[idim] = gsl_vector_get(frame, idim);

  // Compute all the Mahalanobis distances for all Gaussians
  register int   minDistIdx = 0;
  register float minDistSum = 1.0E20;

  for (int refX = 0; refX < refN(); refX++) {
    register float   distSum  =  _pi + _determinant[refX];
    const float* pt  =  pattern;
    const float* rv  =  _rv->data + (refX * _rv->size2);
    const float* cv  =  _cv[refX]->data;
    int dimX;

    for (dimX = 0; dimX < dimN_4; dimX++) {
      register float diff0;
      if (distSum > minDistSum) break;
      diff0 = *rv++ - *pt++;
      distSum += diff0*diff0*(*cv++);
      diff0 = *rv++ - *pt++;
      distSum += diff0*diff0*(*cv++);
      diff0 = *rv++ - *pt++;
      distSum += diff0*diff0*(*cv++);
      diff0 = *rv++ - *pt++;
      distSum += diff0*diff0*(*cv++);
    }
    if (dimX == dimN_4) {
      for (dimX = 4*dimN_4; dimX < _dimN; dimX++) {
	register float diff0 = *rv++ - *pt++;
	distSum += diff0*diff0*(*cv++);
      }
    }

    if (distSum < minDistSum) {
      minDistSum = distSum;
      minDistIdx = refX;
    }
  }

  // assume a fully continuous system
  minDistSum *= 0.5;
  if (val != NULL)
    minDistSum += val[minDistIdx];

  return LogLhoodIndexPtr(new LogLhoodIndex(minDistSum, minDistIdx));
}

void CodebookBasic::copyMeanVariance(const CodebookBasicPtr& cb)
{
  /*
  if (_cv) {
    for (UnShrt refX = 0; refX < _refN; refX++)
      fcvFree(_cv[refX]);
    delete[] _cv;
  }
  */

  if (_refN != cb->_refN || _dimN != cb->_dimN)
    _dealloc();

  _refN = cb->_refN;  _dimN = cb->_dimN;  _orgDimN = cb->_orgDimN;  _covType = cb->_covType;

  _allocCount();  _allocRV();  _allocCovar();

  gsl_matrix_float_memcpy(_rv, cb->_rv);
  for (UnShrt refX = 0; refX < _refN; refX++)
    gsl_vector_float_memcpy(_cv[refX], cb->_cv[refX]);
  memcpy(_determinant, cb->_determinant, _refN * sizeof(float));
}

float CodebookBasic::ttlCounts() const
{
  float cnt = 0.0;
  for (unsigned refX = 0; refX < _refN; refX++)
    cnt += _count[refX];

  return cnt;
}

const double CodebookBasic::DefaultExptValue = -100.0;

float CodebookBasic::_scoreAll(int frameX, const float* val, float* addCount)
{
  double minlogdist;
  UnShrt rbufX = frameX % _cache->_distCacheN;

  // check, if cache is useable
  bool update = false;
  if (_cache->_distFrameX[rbufX] != frameX) {
    _cache->_distFrameX[rbufX] = frameX;
    update = true;
  }
  float* logdistCache = _cache->_logdist[rbufX];

  // set a pointer to the corresponding speech frame
  const gsl_vector_float* pattern = _feature->next(frameX);
  if (pattern == NULL)
    throw jconsistency_error("Can't get frame %d.\n", frameX);

  // If the Gaussians of the actual codebook are not in the
  // cache, we will have to compute them now
  if (update) {

    // Compute all the Mahalanobis distances for the Gaussians
    minlogdist = 1E20;

    switch (_covType) {

      case COV_NO:

        for (UnShrt refX = 0; refX < _refN; refX++) {
          const float* pt       = pattern->data;
	        float* rv       = _rv->data + (refX * _rv->size2);
                double distSum  = _pi;
          for (UnShrt dimX = 0; dimX < _dimN; dimX++) {
            register double diff0 = *rv++ - *pt++;
            distSum += diff0*diff0;
          }
          logdistCache[refX] = 0.5*distSum;
          if (logdistCache[refX] < minlogdist)
            minlogdist = logdistCache[refX];
        }
        break;

      case COV_RADIAL:

        for (UnShrt refX = 0; refX < _refN; refX++) {
          const float* pt      =  pattern->data;
	        float* rv      =  _rv->data + (refX * _rv->size2);
                float  cv      =  _cv[refX]->data[0];
                double distSum =  0.0;
          for (UnShrt dimX = 0; dimX < _dimN; dimX++) {
            register double diff0 = *rv++ - *pt++;
            distSum += diff0*diff0;
          }
          distSum = (_pi + _determinant[refX]) + cv * distSum;
          logdistCache[refX] = 0.5*distSum;
          if (logdistCache[refX] < minlogdist)
            minlogdist = logdistCache[refX];
        }
        break;

      case COV_DIAGONAL:

        for (UnShrt refX = 0; refX < _refN; refX++) {
	  const float* pt      = pattern->data;
	        float* rv      = _rv->data + (refX * _rv->size2);
                float* cv      = _cv[refX]->data;
	        double distSum = _pi + _determinant[refX];
          for (UnShrt dimX = 0; dimX < _dimN; dimX++) {
	    register double diff0 = *rv++ - *pt++;
	    distSum += diff0*diff0*(*cv++);
          }

          logdistCache[refX] = 0.5*distSum;
          if (logdistCache[refX] < minlogdist)
            minlogdist = logdistCache[refX];
        }
        break;

      default:
	throw jtype_error("Cannot determine covariance type.");
        break;
    }

    logdistCache[_refN] = minlogdist;

  } else

    // Gaussians are cached, so we only need the
    // value of the minimum Mahalanobis distance
    // which we have already computed previously
    minlogdist = logdistCache[_refN];

  // Add up the exponentiated Mahalanobis distances
  // in the cache, and logarithmize the sum to get
  // the score. In case of one single vector, we
  // do not need to exponentiate
  if (_refN == 1) {

    if (addCount != NULL) addCount[0] = 1.0;
    return (_scale * (logdistCache[0] + val[0]));

  }

  double score = 0.0;
  if (addCount == NULL) {
    for (UnShrt refX = 0; refX < _refN; refX++) {
      double dist = _scale * (minlogdist - logdistCache[refX]);
      if (dist > DefaultExptValue) score += exp(dist - _scale * val[refX]);
    }
  } else {
    for (UnShrt refX = 0; refX < _refN; refX++) {
      double dist =  _scale * (minlogdist - logdistCache[refX]);
      if (dist > DefaultExptValue)
	score += (addCount[refX] = exp(dist - _scale * val[refX]));
      else
	addCount[refX] = 0.0;
    }
  }

  return (float) (_scale * minlogdist - log(score));
}


// ----- methods for class `CodebookSetBasic' -----
//
CodebookSetBasic::CodebookSetBasic(const String& descFile, FeatureSetPtr& fs,
				   const String& cbkFile )
  : _cblist("Codebook Set"), _featureSet(fs)
{  
  if (descFile == "") return;

  freadAdd(descFile, ';', this);

  if (cbkFile == "") return;

  load(cbkFile);
}

CodebookSetBasic::~CodebookSetBasic() { }

void CodebookSetBasic::setFeatures(FeatureSetPtr& featureSet)
{
  for (_Iterator itr(_cblist); itr.more(); itr++)
    (*itr)->setFeature(featureSet);
}

static void _splitList(const String& s, list<String>& line)
{
  String::size_type pos = 0;
  for (unsigned i = 0; i < 5; i++) {
    String::size_type spacePos = s.find_first_of(' ', pos);
    String subString(s.substr(pos, spacePos - pos));
    line.push_back(subString);
    pos = spacePos + 1;
    while (s[pos] == ' ') pos++;
  }
}

void CodebookSetBasic::__add(const String& s)
{
  list<String> line;
  // splitList(s, line);
  _splitList(s, line);

  if (line.size() != 5)
    throw jparse_error("Could not parse codebook description line:\n %s", s.c_str());

  list<String>::const_iterator itr = line.begin();

  String  name = (*itr);	        itr++;
  String  feat = (*itr);                itr++;
  int     rfN  = atoi((*itr).c_str());	itr++;
  int     dmN  = atoi((*itr).c_str());  itr++;
  // CovType cvTp = cvStrToType((*itr).c_str());
  CovType cvTp = COV_DIAGONAL;

  // printf("name='%s' feat='%s' rfN=%d dmN=%d\n", name.c_str(), feat.c_str(), rfN, dmN);

  if (_featureSet.isNull())
    _addCbk(name, rfN, dmN, cvTp, feat);
  else
    _addCbk(name, rfN, dmN, cvTp, _featureSet->feature(feat));
}

void CodebookSetBasic::_addCbk(const String& name, UnShrt rfN, UnShrt dmN, CovType cvTp,
			       const String& featureName)
{
  CodebookBasicPtr cbk(new CodebookBasic(name, rfN, dmN, cvTp, featureName));

  _cblist.add(name, cbk);
}

void CodebookSetBasic::_addCbk(const String& name, UnShrt rfN, UnShrt dmN, CovType cvTp,
			       VectorFloatFeatureStreamPtr feat)
{
  CodebookBasicPtr cbk(new CodebookBasic(name, rfN, dmN, cvTp, feat));

  _cblist.add(name, cbk);
}

void CodebookSetBasic::setNSubFeat(UnShrt nsub)
{
  _nSubFeat = nsub;
  for (_Iterator itr(_cblist); itr.more(); itr++)
    (*itr)->setSubFeatN(nsub);
}

void CodebookSetBasic::setSubFeatLen(UnShrt len)
{
  _subFeatLen = len;
}

void CodebookSetBasic::setFeatLen(UnShrt len)
{
  _featLen = len;
}

void CodebookSetBasic::setOrgFeatLen(UnShrt len)
{
  _orgFeatLen = len;
}

void CodebookSetBasic::setOrgSubFeatLen(UnShrt len)
{
  _orgSubFeatLen = len;
}

void CodebookSetBasic::setScoreAll(unsigned cacheN)
{
  for (_Iterator itr(_cblist); itr.more(); itr++)
    (*itr)->setScoreAll(cacheN);
}

void CodebookSetBasic::applySTC(const gsl_matrix_float* trans)
{
  for (_Iterator itr(_cblist); itr.more(); itr++)
    (*itr)->applySTC(trans);
}

void CodebookSetBasic::setScale(float scale)
{
  for (_Iterator itr(_cblist); itr.more(); itr++)
    (*itr)->setScale(scale);
}

void CodebookSetBasic::resetCache()
{
  for (_Iterator itr(_cblist); itr.more(); itr++)
    (*itr)->resetCache();
}

void CodebookSetBasic::resetFeature()
{
  for (_Iterator itr(_cblist); itr.more(); itr++)
    (*itr)->resetFeature();
}

const unsigned CodebookSetBasic::MaxNameLength = 200;
const int      CodebookSetBasic::CodebookMagic = 64207531;

void CodebookSetBasic::load(const String& filename)
{
  if (filename == "")
    throw j_error("Codebook set file name is null.");

  FILE* fp = fileOpen(filename, "r");

  if (fp == NULL)
    throw jio_error("Could not open codebook file %s.", filename.chars());

  printf("Loading codebook set from file %s.\n", filename.chars());
  fflush(stdout);

  int magic = read_int(fp);

  static char name[MaxNameLength];

  if (magic == CodebookMagic) {

    int l_cb0 = read_int(fp);
    int l_cbN = read_int(fp);

    for (int icb = l_cb0; icb < l_cbN; icb++) {
      read_string(fp, name);
      CodebookBasicPtr cb = find(name);
      cb->load(fp);
    }

  } else {

    int l_cbN = magic;
    int mode  = 1;

    if (l_cbN < 0) {	// compression mode
      l_cbN *= -1;
      mode   = -1;
    }

    if (l_cbN != ((int) ncbks()))
      throw jio_error("Found %d codebooks, expected %d.",
		      l_cbN, ((int) ncbks()));

    if (mode == 1) {
      for (int icb = 0; icb < l_cbN; icb++) {
	read_string(fp, name);
	CodebookBasicPtr cb = find(name);
	cb->loadOld(fp);
      }
    }
    if (mode == -1)
      throw jio_error("Mode not supported.");
  }

  fileClose(filename, fp);
}

void CodebookSetBasic::save(const String& filename, bool janusFormat, int nParts, int part)
{
  FILE* fp = fileOpen(filename, "w");

  if (fp == NULL)
    throw jio_error("Could not open codebook file %s.", filename.chars());

  unsigned ttlCbks = 0;
  for (_ConstIterator itr(_cblist, nParts, part); itr.more(); itr++)
    ttlCbks++;
  
  if (janusFormat == false) {
    write_int(fp, CodebookMagic);
    write_int(fp, 0);
  }

  write_int(fp, ttlCbks);
  for (_ConstIterator itr(_cblist, nParts, part); itr.more(); itr++)
    (*itr)->save(fp, janusFormat);

  fileClose(filename, fp);
}

void CodebookSetBasic::write(const String& fileName, const String& time) const
{
  FILE* fp = fileOpen(fileName, "w");

  fprintf(fp, "; -------------------------------------------------------\n");
  fprintf(fp, ";  Name            : %s\n", "cbk");
  fprintf(fp, ";  Type            : %s\n", "CodebookSet");
  fprintf(fp, ";  Number of Items : %d\n", _cblist.size());
  fprintf(fp, ";  Date            : %s\n", time.c_str());
  fprintf(fp, "; -------------------------------------------------------\n");

  for (_ConstIterator itr(_cblist); itr.more(); itr++)
    (*itr)->write(fp);

  fileClose(fileName, fp);
}

void CodebookSetBasic::setRegClasses(UnShrt c)
{
  for (_Iterator itr(_cblist); itr.more(); itr++)
    (*itr)->setRegClasses(c);
}


// ----- methods for class `CodebookBasic::Cache' -----
//
const short  CodebookBasic::Cache::EmptyIndicator    = (-1);
const UnShrt CodebookBasic::Cache::DefaultDistCacheN = 100;

CodebookBasic::Cache::Cache(UnShrt refN, UnShrt distCacheN)
  :_refN(refN), _distCacheN(distCacheN)
{
  _distFrameX = new int[_distCacheN];
  _logdist    = new float*[_distCacheN];

  // fill the frame index array with 'EmptyIndicator'
  for (UnShrt i = 0; i < _distCacheN; i++)
    _distFrameX[i] = EmptyIndicator;

  // 'logdist' array has refX+1 elements; 'scale' is the last element
  for (UnShrt i = 0; i < _distCacheN; i++)
    _logdist[i]   = new float[_refN + 1];

  reset();
}

CodebookBasic::Cache::~Cache()
{
  for (UnShrt i = 0; i < _distCacheN; i++)
    delete[] _logdist[i];

  delete[] _distFrameX;
  delete[] _logdist;
}

void CodebookBasic::Cache::reset()
{
  for (UnShrt i = 0; i < _distCacheN; i++)
    _distFrameX[i] = EmptyIndicator;
}
