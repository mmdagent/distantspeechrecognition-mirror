//
//                               Millennium
//                   Automatic Speech Recognition System
//                                  (asr)
//
//  Module:  asr.adapt
//  Purpose: Maximum likelihood model space adaptation. 
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

#include <ctype.h>
#include <iostream>
#include "adapt/transform.h"
#include "common/jexception.h"
#include "matrix/gslmatrix.h"

#include <gsl/gsl_blas.h>

static       int Trace     = 0x0000;
static const int Top       = 0x0001;
static const int Sequences = 0x0002;


// ----- methods and friends of class `LaurentSeries' -----
//
LaurentSeries::LaurentSeries() : _seqLength(DefaultSeriesLength)
{
  int ttlSeqLength = (2 * _seqLength) - 1;
  _seq  = new double[ttlSeqLength];
  _seq += last();
}

LaurentSeries::~LaurentSeries()
{
  _seq -= last();
  delete[] _seq;
}

LaurentSeries& LaurentSeries::operator=(const LaurentSeries& rhs)
{
  assert(len() == rhs.len());
  int lst = last();
  for (int i = -lst; i <= lst; i++)
    _seq[i] = rhs._seq[i];
  return *this;
}

LaurentSeries& LaurentSeries::operator=(double val)
{
  int lst = last();
  for (int i = -lst; i <= lst; i++)
    _seq[i] = val;
  return *this;
}

void multAdd(double factor, const LaurentSeries& seq, LaurentSeries& out)
{
  assert(seq.len() == out.len());

  int last = seq.last();
  for (int i = -last; i <= last; i++)
    out[i] += factor * seq[i];
}

void shift(LaurentSeries& seq)
{
  int last = seq.last();
  for (int i = last; i > -last; i--)
    seq[i] = seq[i-1];
}

void CauchyProduct(const LaurentSeries& seq1, const LaurentSeries& seq2,
		   LaurentSeries& out)
{
  assert(seq1.len() == seq2.len() && seq1.len() == out.len());

  int last = seq1.last();
  for (int i = -last; i <= last; i++) {
    double sum = 0.0;

    int lower = max(-last, i-last);
    int upper = min( last, i+last);
    for (int j = lower; j <= upper; j++)
      sum += seq1[j] * seq2[i - j];

    out[i] = sum;
  }
}

ostream& operator<<(ostream& os, const LaurentSeries& seq)
{
  for (int j = -10; j <= 10; j++)
    os << " " << setw(4) << j << " : "
       << setprecision(16) << setw(16) << seq[j] << endl;
  return os;
}

AdaptationType getAdaptType(const char* ctype)
{
  if (ctype == NULL) return SLAPT;

  String type(ctype); 
  if (type == "NONE" || type == "None")
    return NoAdaptation;
  if (type == "CONFORMAL" || type == "Conformal" || type == "RAPT")
    return RAPT;
  if (type == "SLAPT")
    return SLAPT;
  if (type == "MLLR")
    return MLLR;
  if (type == "STC")
    return STC;
  if (type == "LDA")
    return LDA;

  throw j_error("Unrecognized bias type %s", type.chars());
  return Unspecified;
}

bool isAncestor(unsigned ancestor, unsigned child)
{
  if (ancestor == 0)
    throw jindex_error("Ancestor index is zero.");
  if (child == 0)
    throw jindex_error("Child index zero.");

  for (; child >= ancestor; child /= 2)
    if (child == ancestor)
      return true;

  return false;
}


// ----- methods for class `CoeffSequence' -----
//
CoeffSequence::CoeffSequence(UnShrt sz)
  : _coeff(sz) { }

CoeffSequence::CoeffSequence(const NaturalVector& v)
  : _coeff(v.featLen())
{
  UnShrt flen = v.featLen();
  for (UnShrt i = 0; i < flen; i++)
    _coeff[i] = v[i];
}

CoeffSequence::CoeffSequence(const String& fileName)
{
  FILE* fp = fileOpen(fileName, "r");

  double value;
  while(fscanf(fp, "%lf", &value) != EOF)
    _coeff.push_back(value);
  fileClose(fileName, fp);
}

CoeffSequence& CoeffSequence::operator=(double v)
{
  resize(1);

  _coeff[0] = v;

  return *this;
}

CoeffSequence& CoeffSequence::operator=(const NaturalVector& v)
{
  UnShrt flen = v.featLen();  resize(flen);

  for (UnShrt i = 0; i < flen; i++)
    _coeff[i] = v[i];

  return *this;
}

void CoeffSequence::zero()
{
  UnShrt len = _coeff.size();
  for (UnShrt n = 0; n < len; n++)
    _coeff[n] = 0.0;
}


// ----- methods for class `ParamBase' -----
//
static const unsigned MaxSymbolLength  =   16;
static const unsigned MaxUttLineLength = 2048;

const char* ParamBase::_symMap[]
= { "RegClass", "RAPTParam", "SLAPTParam", "MLLRParam", "STCParam", "LDAParam", "BiasOffset",
    "EndClass", "AdaptType", "EOF", "" };

const char* ParamBase::_adaptMap[]
= { "RAPT", "SLAPT", "MLLR", "STC", "LDA", "" };

ParamBase::ParamBase()
  : _regClass(0) { }

ParamBase::ParamBase(unsigned rc, unsigned b_sz)
  : _regClass(rc), _bias(b_sz) { }

ParamBase::ParamBase(unsigned rc, const NaturalVector& bs)
  : _regClass(rc), _bias(bs) { }

ParamBase::Symbol ParamBase::_getSymbol(FILE* fp)
{
  int c;
  static char buf[MaxSymbolLength];

  while (isspace(c=getc(fp)));     // Look for symbol
  if (c == EOF) return EOFSymbol;
  if (c != '<') throw j_error("Symbol expected");
  unsigned i = 0;
  while ((c=getc(fp)) != '>' && i < (MaxSymbolLength-1))
    buf[i++] = c;
  buf[i] = '\0';
  if (c != '>') throw j_error("_getSymbol: > missing in symbol");

  while (isspace(c=getc(fp)));
  if (c != EOF) ungetc(c, fp);

  for (int s = RegClass; s <= EOFSymbol; s++)
     if (strcmp(_symMap[s], buf) == 0)
       return (Symbol) s;

  throw j_error("Invalid symbol %s", buf);
  return NullSymbol;
}

AdaptationType ParamBase::_getAdaptType(FILE* fp)
{
  int c;
  static char buf[MaxSymbolLength];

  while (isspace(c=getc(fp)));
  if (c == EOF)
    throw j_error("No adaptation type specified.");
  ungetc(c, fp);

  unsigned i = 0;
  while (!isspace(c = getc(fp)) && i < (MaxSymbolLength-1))
     buf[i++] = c;
  buf[i] = '\0';
  if (i == (MaxSymbolLength-1))
    throw j_error("Adaptation type %s is not valid.", buf);

  for (int as = RAPT; as <= MLLR; as++)
    if (strcmp(_adaptMap[as], buf) == 0)
  return (AdaptationType) as;

  throw j_error("Invalid adaptation type %s", buf);
  return Unspecified;
}

void ParamBase::_putAdaptType(FILE* fp) const
{
  _putSymbol(fp, AdaptType);
  fprintf(fp, " %s\n", _adaptMap[_type()]);
}

void ParamBase::_putSymbol(FILE* fp, Symbol sym) const
{
  if (sym < RegClass || sym > NullSymbol)
    throw j_error("Symbol %d is unknown", sym);

   fprintf(fp, "<%s>", _symMap[sym]);
}

void ParamBase::_getRegClass(FILE* fp)
{
  fscanf(fp, "%u", &_regClass);
}

void ParamBase::_getBiasOffset(FILE* fp)
{
  int nParams;
  fscanf(fp, "%d", &nParams);
   
  _bias.resize(nParams);
  for (int i = 0; i < nParams; i++) {
    double par;
    fscanf(fp,"%lf", &par);
    _bias[i] = par;
  }
}

void ParamBase::_putRegClass(FILE* fp) const
{
  unsigned rClass = (regClass() == 0) ? 1 : regClass();

  _putSymbol(fp, RegClass);
  fprintf(fp, " %d\n", rClass);
}

void ParamBase::_putBiasOffset(FILE* fp) const
{
  unsigned len = _bias.size();
  if (len == 0) return;
   
  _putSymbol(fp, BiasOffset);
  fprintf(fp, " %d\n", len);
  for (unsigned i = 0; i < len; i++)
    fprintf(fp, " %g", _bias[i]);
  fprintf(fp, "\n");
}

void ParamBase::_putEndClass(FILE* fp) const
{
  _putSymbol(fp, EndClass);
  fprintf(fp, "\n");
}

// virtual constructor for `ParamBase' hierarchy
ParamBasePtr ParamBase::param(const String& fileName, const String& spkr)
{
  String paramFile(fileName); paramFile += "."; paramFile += spkr;
  FILE* fp = fileOpen(paramFile, "r");

  ParamBasePtr par = param(fp);
  fileClose(paramFile, fp);

  return par;
}

ParamBasePtr ParamBase::param(FILE* fp)
{
  int c;
  if (fp == NULL || (c = getc(fp)) == EOF) return NULL;

  ungetc(c, fp);

  if (_getSymbol(fp) != AdaptType)
    throw jtype_error("Unable to determine adaptation type.");

  AdaptationType type = _getAdaptType(fp);

  switch (type) {
  case RAPT:
    return RAPTParamPtr(new RAPTParam(fp));
  case SLAPT:
    return SLAPTParamPtr(new SLAPTParam(fp));
  case MLLR:
    return MLLRParamPtr(new MLLRParam(fp));
  default:
    throw jtype_error("Unknown adaptation type.");
  }

  return ParamBasePtr(NULL);
}

void ParamBase::write(FILE* fp) const
{
  _putAdaptType(fp);
  _putRegClass(fp);
  _putParams(fp);
  _putBiasOffset(fp);
  _putEndClass(fp);
}

void ParamBase::_readParams(FILE* fp)
{
  Symbol sym;

  while ((sym = _getSymbol(fp)) != EndClass) {
    switch (sym) {
    case AdaptType:
      if (_getAdaptType(fp) != _type())
	throw jparameter_error("Attempted to read parameters of wrong type.");  break;
    case RegClass:
      _getRegClass(fp);  break;
    case RAPTParams:
      if (_type() == MLLR)
	throw jparameter_error("Attempted to read MLLR parameters.");
      if (_type() == SLAPT)
	throw jparameter_error("Attempted to read SLAPT parameters.");
      _getParams(fp);  break;
    case SLAPTParams:
      if (_type() == MLLR)
	throw jparameter_error("Attempted to read MLLR parameters.");
      if (_type() == RAPT)
	throw jparameter_error("Attempted to read RAPT parameters.");
      _getParams(fp);  break;
    case MLLRParams:
      if (_type() == RAPT)
	throw jparameter_error("Attempted to read RAPT parameters.");
      _getParams(fp);  break;
    case BiasOffset:
      _getBiasOffset(fp);  break;
    default:
      throw jparameter_error("Unexpected symbol %s in conformal parameter file.", _symMap[sym]);
    }
  }
}


// ----- methods for class `APTParamBase' -----
//
APTParamBase::APTParamBase(unsigned c_sz, unsigned b_sz, unsigned rc)
  : ParamBase(rc, b_sz), _conf(c_sz) { }

APTParamBase::APTParamBase(double alpha, const NaturalVector& bias, unsigned rc)
  : ParamBase(rc, bias), _conf(1)
{
  _conf[0] = alpha;
}

APTParamBase::APTParamBase(const CoeffSequence& seq, const NaturalVector& bias,
         unsigned rc)
  : ParamBase(rc, bias), _conf(seq)
{
  UnShrt flen = _bias.size();
  for (UnShrt i = 0; i < flen; i++)
    _bias[i] = bias[i];
}

void APTParamBase::_getParams(FILE* fp)
{
  int nParams;
  fscanf(fp, "%d", &nParams);

  _conf.resize(nParams);
  for (int i = 0; i < nParams; i++) {
    double par;
    fscanf(fp,"%lf", &par);
    _conf[i] = par;
  }
}


// ----- methods for class `RAPTParam' -----
//
RAPTParam::RAPTParam(FILE* fp)
{
  _readParams(fp);
}

RAPTParam::RAPTParam(const String& paramFile)
{
  FILE* fp = fileOpen(paramFile, "r");

  _readParams(fp);
  fileClose(paramFile, fp);
}

// virtual constructor for `TransformBase'
//
const TransformBasePtr RAPTParam::
transformer(UnShrt sbFtSz, UnShrt nSbFt, UnShrt orgSbFtSz,
	    int trace, const String& ldaFile, UnShrt ldaFeatLen) const
{
  if (size() == 1)
    return TransformBasePtr(new BLTTransformer(*this, sbFtSz, nSbFt, orgSbFtSz,
					       trace, ldaFile, ldaFeatLen));
  else
    return TransformBasePtr(new RAPTTransformer(*this, sbFtSz, nSbFt, orgSbFtSz,
						trace, ldaFile, ldaFeatLen));
}

void RAPTParam::unitCirclePlot(const String& fileName) const
{
  APTTransformerBase* trans;
  if (size() == 1)
    trans = new BLTTransformer(*this);
  else
    trans = new RAPTTransformer(*this);

  static char indexString[10];
  sprintf(indexString, "%04d", regClass());

  String newFileName(fileName); newFileName += "."; newFileName += indexString;
  trans->unitCirclePlot(newFileName);

  delete trans;
}

void RAPTParam::_putParams(FILE* fp) const
{
  _putType(fp);

  fprintf(fp, " %d\n", _conf.size());
  fprintf(fp, " %g", _conf[0]);
  for (unsigned i = 1; i < _conf.size(); i++) {
    double out = (i % 2 == 0) ? fabs(_conf[i]) : _conf[i];
    fprintf(fp, " %g", out);
  }
  fprintf(fp, "\n");
}


// ----- methods for class `SLAPTParam' -----
//
SLAPTParam::SLAPTParam(FILE* fp)
{
  _readParams(fp);
}

SLAPTParam::SLAPTParam(const String& paramFile)
{
  FILE* fp = fileOpen(paramFile, "r");

  _readParams(fp);
  fileClose(paramFile, fp);
}

// virtual constructor for `TransformBase'
//
const TransformBasePtr SLAPTParam::
transformer(UnShrt sbFtSz, UnShrt nSbFt, UnShrt orgSbFtSz,
	    int trace, const String& ldaFile, UnShrt ldaFeatLen) const
{
  return TransformBasePtr(new SLAPTTransformer(*this, sbFtSz, nSbFt, orgSbFtSz,
					       trace, ldaFile, ldaFeatLen));
}

void SLAPTParam::unitCirclePlot(const String& fileName) const
{
  APTTransformerBase* trans = new SLAPTTransformer(*this);

  static char indexString[10];
  sprintf(indexString, "%04d", regClass());

  String newFileName(fileName); newFileName += "."; newFileName += indexString;
  trans->unitCirclePlot(newFileName);

  delete trans;
}

void SLAPTParam::_putParams(FILE* fp) const
{
  _putType(fp);

  fprintf(fp, " %d\n", _conf.size());
  for (unsigned i = 0; i < _conf.size(); i++)
    fprintf(fp, " %g", _conf[i]);

  fprintf(fp, "\n");
}


// ----- methods for class `MLLRParam' -----
//
MLLRParam::MLLRParam(FILE* fp)
  : _size(0), _matrix(NULL)
{
  if (fp != NULL)
    _readParams(fp);
}

MLLRParam::MLLRParam(const MLLRParam& m)
  : ParamBase(m), _size(m._size),
    _matrix(gsl_matrix_alloc(m._matrix->size1, m._matrix->size2))
{
  gsl_matrix_memcpy(_matrix, m._matrix);
}

MLLRParam::~MLLRParam()
{
  if (_matrix)
    gsl_matrix_free(_matrix);
}

void MLLRParam::_putParams(FILE* fp) const
{
  _putSymbol(fp, MLLRParams); 
  fprintf(fp, " %d\n", _size);

  for (UnShrt m = 0; m < _size; m++) {
    for (UnShrt n = 0; n < _size; n++) {
      fprintf(fp, " %g", gsl_matrix_get(_matrix, m, n));
    }
    fprintf(fp, "\n");
  }
}

void MLLRParam::_getParams(FILE* fp)
{
  fscanf(fp, "%hu", &_size);

  if (_matrix == NULL) {
    _matrix = gsl_matrix_alloc(_size, _size);
  } else if (_matrix->size1 != _size || _matrix->size2 != _size) {
    gsl_matrix_free(_matrix);
    _matrix = gsl_matrix_alloc(_size, _size);
  }

  for (UnShrt m = 0; m < _size; m++) {
    for (UnShrt n = 0; n < _size; n++) {
      double elem;
      fscanf(fp,"%lf", &elem);
      gsl_matrix_set(_matrix, m, n, elem);
    }
  }
}

MLLRParam& MLLRParam::operator=(const MLLRParam& m)
{
  _size   = m._size;
  if (_matrix && (_matrix->size1 != _size || _matrix->size2 != _size)) {
    gsl_matrix_free(_matrix);
    _matrix = gsl_matrix_alloc(_size, _size);
  }
  gsl_matrix_memcpy(_matrix, m._matrix);
  _bias   = m._bias;

  return *this;
}

MLLRParam& MLLRParam::operator=(const gsl_matrix* mat)
{
  if ((mat->size1+1) != mat->size2)
    throw jdimension_error("Matrix dimensions (%d vs. %d) do not match.",
			   mat->size1, mat->size2);

  if (_matrix == NULL) {
    _size = mat->size1;
    _matrix = gsl_matrix_alloc(_size, _size);
  } else if (_size != mat->size1) {
    _size = mat->size1;
    gsl_matrix_free(_matrix);
    _matrix = gsl_matrix_alloc(_size, _size);
  }

  if (_bias.isZero())
    _bias.resize(_size);
  else if (_bias.size() != _size)
    throw jdimension_error("Bias sizes (%d vs. %d) do not match.", _bias.size(), _size);

  for (UnShrt m = 0; m < _size; m++) {
    _bias[m] = gsl_matrix_get(mat, m, _size);
    for (UnShrt n = 0; n < _size; n++) {
      gsl_matrix_set(_matrix, m, n, gsl_matrix_get(mat, m, n));
    }
  }

  return *this;
}

// virtual constructor for `TransformBase'
//
const TransformBasePtr MLLRParam::
transformer(UnShrt sbFtSz, UnShrt nSbFt, UnShrt orgSbFtSz,
	    int trace, const String& ldaFile, UnShrt ldaFeatLen) const
{
  if (sbFtSz != orgSbFtSz)
    throw jdimension_error("Sub-features lengths (%d vs. %d) are not equivalent.", sbFtSz, orgSbFtSz);

  return TransformBasePtr(new MLLRTransformer(*this, sbFtSz, nSbFt));
}


// ----- methods for class `STCParam' -----
//
STCParam::STCParam(FILE* fp)
  : _size(0), _matrix(NULL)
{
  if (fp != NULL)
    _readParams(fp);
}

STCParam::STCParam(const STCParam& m)
  : ParamBase(m), _size(m._size),
    _matrix(gsl_matrix_alloc(m._matrix->size1, m._matrix->size2))
{
  gsl_matrix_memcpy(_matrix, m._matrix);
}

STCParam::~STCParam()
{
  if (_matrix)
    gsl_matrix_free(_matrix);
}

void STCParam::_putParams(FILE* fp) const
{
  _putSymbol(fp, STCParams); 
  fprintf(fp, " %d\n", _size);

  for (UnShrt m = 0; m < _size; m++) {
    for (UnShrt n = 0; n < _size; n++) {
      fprintf(fp, " %g", gsl_matrix_get(_matrix, m, n));
    }
    fprintf(fp, "\n");
  }
}

void STCParam::_getParams(FILE* fp)
{
  fscanf(fp, "%hu", &_size);

  if (_matrix == NULL) {
    _matrix = gsl_matrix_alloc(_size, _size);
  } else if (_matrix->size1 != _size || _matrix->size2 != _size) {
    gsl_matrix_free(_matrix);
    _matrix = gsl_matrix_alloc(_size, _size);
  }

  for (UnShrt m = 0; m < _size; m++) {
    for (UnShrt n = 0; n < _size; n++) {
      double elem;
      fscanf(fp,"%lf", &elem);
      gsl_matrix_set(_matrix, m, n, elem);
    }
  }
}

STCParam& STCParam::operator=(const STCParam& m)
{
  _size   = m._size;
  if (_matrix && (_matrix->size1 != _size || _matrix->size2 != _size)) {
    gsl_matrix_free(_matrix);
    _matrix = gsl_matrix_alloc(_size, _size);
  }
  gsl_matrix_memcpy(_matrix, m._matrix);
  _bias   = m._bias;

  return *this;
}

STCParam& STCParam::operator=(const gsl_matrix* mat)
{
  if ((mat->size1+1) != mat->size2)
    throw jdimension_error("Matrix dimensions (%d vs. %d) do not match.",
			   mat->size1, mat->size2);

  if (_matrix == NULL) {
    _size = mat->size1;
    _matrix = gsl_matrix_alloc(_size, _size);
  } else if (_size != mat->size1) {
    _size = mat->size1;
    gsl_matrix_free(_matrix);
    _matrix = gsl_matrix_alloc(_size, _size);
  }

  if (_bias.isZero())
    _bias.resize(_size);
  else if (_bias.size() != _size)
    throw jdimension_error("Bias sizes (%d vs. %d) do not match.", _bias.size(), _size);

  for (UnShrt m = 0; m < _size; m++) {
    _bias[m] = gsl_matrix_get(mat, m, _size);
    for (UnShrt n = 0; n < _size; n++) {
      gsl_matrix_set(_matrix, m, n, gsl_matrix_get(mat, m, n));
    }
  }

  return *this;
}

// virtual constructor for `TransformBase'
//
const TransformBasePtr STCParam::
transformer(UnShrt sbFtSz, UnShrt nSbFt, UnShrt orgSbFtSz,
	    int trace, const String& ldaFile, UnShrt ldaFeatLen) const
{
  /*
  if (sbFtSz != orgSbFtSz)
    throw jdimension_error("Sub-features lengths (%d vs. %d) are not equivalent.", sbFtSz, orgSbFtSz);
  */

  throw jconsistency_error("Must finish method.");

  return TransformBasePtr(NULL);
}


// ----- methods for class `ParamTree' -----
//
ParamTree::
ParamTree(const String& fileName)
  : _type(Unspecified)
{
  read(fileName);
}

ParamTree::ParamTree(const ParamTree& tree)
  : _type(tree._type)
{
  for (ParamMapConstIter itr = tree._map.begin(); itr != tree._map.end(); itr++) {
    const ParamBasePtr& oldPar((*itr).second);
    _map.insert(ValueType(oldPar->regClass(), oldPar->copy()));
  }
}

ParamTree::~ParamTree()
{
  clear();
}

void ParamTree::clear()
{
  _type = Unspecified;

  _map.erase(_map.begin(), _map.end());
}

ParamTree& ParamTree::operator=(const ParamTree& tree)
{
  _type = tree._type;

  clear();

  for (ParamMapConstIter itr = tree._map.begin(); itr != tree._map.end(); itr++) {
    const ParamBasePtr& oldPar((*itr).second);
    _map.insert(ValueType(oldPar->regClass(), oldPar->copy()));
  }

  return *this;
}

void ParamTree::write(const String& fileName) const
{
  cout << "Writing speaker parameters to " << fileName << endl << endl;

  FILE* fp = fileOpen(fileName, "w");

  for (ConstIterator itr(*this); itr.more(); itr++)
    itr.par()->write(fp);
  fileClose(fileName, fp);
}

ParamBasePtr& ParamTree::find(unsigned initRC, bool useAncestor)
{
  if (initRC == 0)
    throw jindex_error("Regression class index is zero");

  ParamMapIter itr = _map.find(initRC);
  if (itr != _map.end())
    return (*itr).second;

  if (useAncestor == false)
    throw jindex_error("No parameters for index %d.", initRC);

  unsigned reg = initRC / 2;
  while (reg > 0 &&  (itr = _map.find(reg)) == _map.end())
    reg /= 2;
  
  if (reg == 0)
    throw jindex_error("No ancestor for node %d.", initRC);

  ParamBasePtr par(((*itr).second)->copy());

  par->_setClass(initRC);  _map.insert(ValueType(initRC, par));
  
  itr = _map.find(initRC);
  return (*itr).second;
}

APTParamBasePtr& ParamTree::
findAPT(unsigned initRC, AdaptationType typ, bool useAncestor)
{
  if (_type == Unspecified)
    _type = typ;
  else if (_type != typ)
    throw jconsistency_error("Tree not instantiated with %s parameters.",
			     ((typ == RAPT) ? "RAPT" :  "SLAPT"));

  if (initRC == 0)
    throw jindex_error("Regression class index is zero");

  ParamMapIter itr = _map.find(initRC);
  if (itr != _map.end())
    return Cast<APTParamBasePtr>((*itr).second);

  if (useAncestor == false)
    throw jindex_error("Could not find parameters for regression class %d.", initRC);

  unsigned reg = initRC / 2;
  while (reg > 0 &&  (itr = _map.find(reg)) == _map.end())
    reg /= 2;

  APTParamBasePtr par(NULL);
  if (reg == 0)
    par = (_type == RAPT) ?
      (APTParamBase*) new RAPTParam() : (APTParamBase*) new SLAPTParam();
  else
    par = (APTParamBase*) ((*itr).second)->copy();

  par->_setClass(initRC);  _map.insert(ValueType(initRC, par));

  itr = _map.find(initRC);
  return Cast<APTParamBasePtr>((*itr).second);
}

MLLRParamPtr& ParamTree::findMLLR(unsigned initRC, bool useAncestor)
{
  if (_type == Unspecified)
    _type = MLLR;
  else if (_type != MLLR)
    throw jconsistency_error("Tree not instantiated with MLLR parameters.");

  if (initRC == 0)
    throw jindex_error("Regression class index is zero");

  ParamMapIter itr = _map.find(initRC);
  if (itr != _map.end())
    return Cast<MLLRParamPtr>((*itr).second);

  if (useAncestor == false)
    throw jindex_error("Could not find parameters for regression class %d.", initRC);

  unsigned reg = initRC / 2;
  while (reg > 0 &&  (itr = _map.find(reg)) == _map.end())
    reg /= 2;
  
  MLLRParamPtr par((reg == 0) ? new MLLRParam() : (MLLRParam*) ((*itr).second)->copy());

  par->_setClass(initRC);  _map.insert(ValueType(initRC, par));
  
  itr = _map.find(initRC);
  return Cast<MLLRParamPtr>((*itr).second);
}

void ParamTree::read(const String& fileName)
{
  clear();

  if (fileName == "") return;

  cout << "Reading speaker parameters from " << fileName << endl << endl;

  FILE* fp = fileOpen(fileName, "r");

  ParamBasePtr par(ParamBase::param(fp));
  while (par.isNull() == false) {
    _map.insert(ValueType(par->regClass(), par));
    if (_type == Unspecified)
      _type = par->_type();
    else if (_type != par->_type())
      throw jconsistency_error("Incorrect adaptation type for regression class %d.", par->regClass());
    par = ParamBase::param(fp);
  }
  if (_type == Unspecified)
    throw jconsistency_error("Unable to determine adaptation type.");

  fileClose(fileName, fp);
}

void ParamTree::applySTC(const String& stcFile)
{
  if (_type != MLLR)
    throw jconsistency_error("'ParamTree::applySTC' only defined for MLLR adaptation.");

  gsl_matrix* stcMatrix     = NULL;
  gsl_matrix* scratchMatrix = NULL;
  gsl_vector* biasVector    = NULL;
  gsl_vector* scratchVector = NULL;
  for (ParamMap::iterator itr = _map.begin(); itr != _map.end(); itr++) {
    MLLRParamPtr& mllr(Cast<MLLRParamPtr>((*itr).second));
    gsl_matrix* mllrMatrix = (gsl_matrix*) mllr->matrix();

    if (stcMatrix == NULL) {
      stcMatrix     = gsl_matrix_alloc(mllrMatrix->size1, mllrMatrix->size1);
      scratchMatrix = gsl_matrix_alloc(mllrMatrix->size1, mllrMatrix->size2);

      biasVector    = gsl_vector_alloc(mllrMatrix->size1);
      scratchVector = gsl_vector_alloc(mllrMatrix->size1);

      FILE* fp = fileOpen(stcFile, "r");
      gsl_matrix_fread(fp, stcMatrix);
      fileClose(stcFile, fp);
    }

    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, stcMatrix, mllrMatrix, 0.0, scratchMatrix);
    gsl_matrix_memcpy(mllrMatrix, scratchMatrix);

    for (unsigned i = 0; i < mllrMatrix->size1; i++)
      gsl_vector_set(scratchVector, i, mllr->bias(i));
    gsl_blas_dgemv(CblasNoTrans, 1.0, stcMatrix, scratchVector, 0.0, biasVector);
    for (unsigned i = 0; i < mllrMatrix->size1; i++)
      mllr->_bias[i] = gsl_vector_get(biasVector, i);
  }

  if (stcMatrix != NULL) {
    gsl_matrix_free(stcMatrix);
    gsl_matrix_free(scratchMatrix);

    gsl_vector_free(biasVector);
    gsl_vector_free(scratchVector);
  }
}


// ----- methods for class `TransformMatrix' -----
//
const UnShrt TransformMatrix::MaxSubBlocks;

TransformMatrix::TransformMatrix()
  : _nBlocks(0), _matrix(NULL), _offset(NULL) { }

void TransformMatrix::
_initialize(UnShrt sbFtSz, UnShrt nSubFt, UnShrt orgSbFtSz)
{
  if (_matrix != NULL)
    throw jconsistency_error("Transform matrix has already been initialized.");

  _nBlocks = nSubFt;
  _matrix  = new gsl_matrix*[nSubFt];
  _offset  = new gsl_vector*[nSubFt];

  for (UnShrt blk = 0; blk < nSubFt; blk++) {
    _matrix[blk] = gsl_matrix_alloc(sbFtSz, orgSbFtSz);  gsl_matrix_set_zero(_matrix[blk]);
    _offset[blk] = gsl_vector_alloc(sbFtSz);             gsl_vector_set_zero(_offset[blk]);
  }
}

TransformMatrix::~TransformMatrix()
{
  for (UnShrt blk = 0; blk < _nBlocks; blk++) {
    gsl_matrix_free(_matrix[blk]); gsl_vector_free(_offset[blk]);
  }
  delete[] _matrix;  delete[] _offset;
}


// ----- methods for class `TransformBase' -----
//
TransformBase::TransformBase()
  : _subFeatLen(0), _nSubFeat(0), _featLen(0), _orgSubFeatLen(0) { }

TransformBase::
TransformBase(UnShrt sbFtSz, UnShrt nSubFt, UnShrt orgSbFtSz, int trace)
  : _subFeatLen(sbFtSz), _nSubFeat(nSubFt), _featLen(_subFeatLen*_nSubFeat),
    _orgSubFeatLen((orgSbFtSz == 0) ? sbFtSz : orgSbFtSz),
    _transFeat(_featLen, _nSubFeat)
{
  Trace = trace;

  if (sbFtSz == 0 || nSubFt == 0)
    throw jconsistency_error("Feature size is incorrectly initialized.");
}

TransformBase::
TransformBase(const ParamBase& par,
	      UnShrt sbFtSz, UnShrt nSubFt, UnShrt orgSbFtSz, int trace)
  : _subFeatLen(sbFtSz), _nSubFeat(nSubFt), _featLen(_subFeatLen*_nSubFeat),
    _orgSubFeatLen((orgSbFtSz == 0) ? sbFtSz : orgSbFtSz),
    _transFeat(_featLen, _nSubFeat)
{
  Trace = trace;

  if (sbFtSz == 0 || nSubFt == 0)
    throw jconsistency_error("Feature size is incorrectly initialized.");

  unsigned biasSize = par.biasSize();

  if (biasSize == 0) return;

  int nSub = (biasSize < _subFeatLen) ? 1 : (biasSize / _subFeatLen);

  _bias.resize(biasSize, nSub);
  for (unsigned n = 0; n < biasSize; n++)
    _bias(n) = par.bias(n);
}

TransformBase::~TransformBase() { }

void TransformBase::transform(CodebookAdapt::GaussDensity& mix) const
{
  mix.allocMean();

  transform(mix.origMean(), _transFeat);
  mix.mean() = _transFeat;
}


// ----- methods for class `APTTransformerBase' -----
//
int			APTTransformerBase::_cnt            = 0;
LaurentSeries*		APTTransformerBase::_qmn            = NULL;
gsl_matrix*			APTTransformerBase::_cepTransMatrix = NULL;
gsl_matrix_float*	APTTransformerBase::_lda            = NULL;

APTTransformerBase::APTTransformerBase(UnShrt nParams)
  : NoAllPassParams(nParams), params(NoAllPassParams), p(NoAllPassParams),
    _transMatrix(NULL),
    _cepSubFeatLen(0), _cepOrgSubFeatLen(0), _cepNSubFeat(0)
{
  _cnt++;
}

APTTransformerBase::
APTTransformerBase(UnShrt sbFtSz, UnShrt nSubFt, UnShrt orgSbFtSz,
		   UnShrt noParams, int trace, const String& ldaFile, UnShrt ldaFtSize)
  : TransformBase((ldaFile == "") ? sbFtSz    : ldaFtSize,
		  (ldaFile == "") ? nSubFt    : 1,
		  (ldaFile == "") ? orgSbFtSz : (orgSbFtSz * nSubFt), trace),
    NoAllPassParams(noParams), params(NoAllPassParams), p(NoAllPassParams),
    _cepSubFeatLen(sbFtSz),
    _cepOrgSubFeatLen((orgSbFtSz == 0) ? sbFtSz : orgSbFtSz),
    _cepNSubFeat(nSubFt)
{
  _transMatrix      = gsl_matrix_alloc(_subFeatLen, _orgSubFeatLen);

  if (_qmn == NULL) {

    cout << "Allocating `_qmn' and `_cepTransMatrix'." << endl;

    _qmn            = new LaurentSeries[_cepOrgSubFeatLen];
    _cepTransMatrix = gsl_matrix_alloc(_cepSubFeatLen, _cepOrgSubFeatLen);

    if (ldaFile != "") {
      _lda = gsl_matrix_float_alloc(cepFeatLen(), cepFeatLen());
      _lda = gsl_matrix_float_load(_lda, ldaFile, /* old= */ true);
      _lda = gsl_matrix_float_resize(_lda, featLen(), cepFeatLen());
    }
  }

  _cnt++;
}

APTTransformerBase::
APTTransformerBase(const ParamBase& par,
		   UnShrt sbFtSz, UnShrt nSubFt, UnShrt orgSbFtSz, int trace,
		   const String& ldaFile, UnShrt ldaFtSize)
  : TransformBase(par,
		  (ldaFile == "") ? sbFtSz    : ldaFtSize,
		  (ldaFile == "") ? nSubFt    : 1,
		  (ldaFile == "") ? orgSbFtSz : (orgSbFtSz * nSubFt), trace),
    NoAllPassParams(par.size()), params(NoAllPassParams), p(NoAllPassParams),
    _cepSubFeatLen(sbFtSz),
    _cepOrgSubFeatLen((orgSbFtSz == 0) ? sbFtSz : orgSbFtSz),
    _cepNSubFeat(nSubFt)
{
  _transMatrix    = gsl_matrix_alloc(_subFeatLen, _orgSubFeatLen);

  if (_qmn == NULL) {
    cout << "Allocating `_qmn' and `_cepTransMatrix' with dimensions "
	 << _cepSubFeatLen << " x " <<_cepOrgSubFeatLen << "." << endl;

    _qmn            = new LaurentSeries[_cepOrgSubFeatLen];
    _cepTransMatrix = gsl_matrix_alloc(_cepSubFeatLen, _cepOrgSubFeatLen);

    if (ldaFile != "") {

      // load the LDA matrix
      _lda = gsl_matrix_float_alloc(cepFeatLen(), cepFeatLen());
      if (ldaFile == "Identity" || ldaFile == "identity" || ldaFile == "IDENTITY") {
	gsl_matrix_float_set_zero(_lda);
	for (unsigned  j = 0; j < cepFeatLen(); j++)
	  gsl_matrix_float_set(_lda, j, j, 1.0);
      } else {
	gsl_matrix_float_load(_lda, ldaFile, /* old= */ true);
      }
      gsl_matrix_float_resize(_lda, featLen(), cepFeatLen());
    }
  }

  _cnt++;

  if (_lda == NULL) return;

  // resize the bias if necessary
  unsigned bSize = _bias.featLen();
  if (bSize != 0 && bSize != featLen()) {
    if (bSize == cepFeatLen() || bSize == cepSubFeatLen()) {
      NaturalVector oBias(_bias, /* deepCopy= */ true);
      _bias.resize(featLen(), /* nsub= */ 1);

      for (UnShrt i = 0; i < featLen(); i++) {
	double  sum = 0.0;
	for (UnShrt j = 0; j < bSize; j++)
	  sum += gsl_matrix_float_get(_lda, i, j) * oBias(j);
	_bias(i) = sum;
      }
    } else {
      throw jconsistency_error("Bias size (%d) is inappropriate.", bSize);
    }
  }
}

APTTransformerBase::~APTTransformerBase()
{
  gsl_matrix_free(_transMatrix);

  if (--_cnt == 0) {
    cout << "De-allocating `_qmn' and `_cepTransMatrix'." << endl;

    delete[] _qmn;                     _qmn            = NULL;
    gsl_matrix_free(_cepTransMatrix);  _cepTransMatrix = NULL;

    if (_lda != NULL) { gsl_matrix_float_free(_lda); _lda = NULL; }
  }
}

Complex APTTransformerBase::
_calcMappedPoint(const Complex& z, const LaurentSeries& _seq)
{
  Complex sum(0.0, 0.0), sum_inv(0.0, 0.0), z_inv(1.0 / z);

  for (int i = _seq.len()-1; i > 0; i--) {
    sum_inv = _seq[-i] + z_inv * sum_inv;
    sum     = _seq[+i] + z     * sum;
  }

  return (sum_inv*z_inv) + (sum*z) + _seq[0];
}

void APTTransformerBase::_calcTransMatrix(const LaurentSeries& sequence)
{
  // 0th sequence is unit sample sequence
  _qmn[0]    = 0.0;
  _qmn[0][0] = 1.0;

  // 1st sequence given by series of coefficients
  _qmn[1] = sequence;

  for (int i = 2; i < _cepOrgSubFeatLen; i++)
    CauchyProduct(_qmn[1], _qmn[i-1], _qmn[i]);

  if (Trace & Sequences)
    for (int l = 0; l < _cepOrgSubFeatLen; l++)
      cout << "Coefficients of Q^" << l << "(z):" << endl
	   << _qmn[l] << endl;

  gsl_matrix_set_zero(_cepTransMatrix);
  gsl_matrix_set(_cepTransMatrix, 0, 0, 1.0);
  for (int n = 0; n < _cepSubFeatLen; n++) {
    for (int m = 1; m < _cepOrgSubFeatLen; m++) {
      double el = _qmn[m][n] + _qmn[m][-n];

      if (n == 0) el /= 2.0;

      gsl_matrix_set(_cepTransMatrix, n, m, el);
    }
  }

  _ldaMatrix(_cepTransMatrix, _transMatrix);
}

void APTTransformerBase::_ldaMatrix(const gsl_matrix* cepMatrix, gsl_matrix* ldaMatrix)
{
  if (_lda == NULL) { gsl_matrix_memcpy(ldaMatrix, cepMatrix);  return; }

  gsl_matrix_set_zero(ldaMatrix);
  for (UnShrt i = 0; i < _featLen; i++) {
    for (UnShrt isub = 0; isub < _cepNSubFeat; isub++) {

      UnShrt offset     = isub * _cepSubFeatLen;
      UnShrt origOffset = isub * _cepOrgSubFeatLen;

      for (UnShrt j = 0; j < _cepOrgSubFeatLen; j++) {
	double sum = 0.0;

	for (UnShrt m = 0; m < _cepSubFeatLen; m++)
	  sum += gsl_matrix_float_get(_lda, i, m + offset) * gsl_matrix_get(cepMatrix, m, j);

	gsl_matrix_set(ldaMatrix, i, j + origOffset, sum);
      }
    }
  }
}

const int APTTransformerBase::NoPlotPoints = 128;

void APTTransformerBase::
_unitCirclePlot(FILE* fp, const LaurentSeries& coeffs)
{
  for (int i = 0; i <= NoPlotPoints; i++) {
    double frac  = ((double) i) / NoPlotPoints;
    double omega = frac * M_PI;
    Complex z(cos(omega), sin(omega));
    Complex m(_calcMappedPoint(z, coeffs));
    double newArg = arg(m);
    if (newArg < 0.0)
      newArg += 2.0 * M_PI;
    newArg /= M_PI;
    fprintf(fp, "%12.8f %12.8f\n", frac, newArg);
  }
}

// intended as an aid to consistency checking
//
void APTTransformerBase::_printSum(const LaurentSeries& seq)
{
  double sum = 0.0;
  for (int i = 1-seq.len(); i < seq.len(); i++)
    sum += seq[i];
  cout << "Sum = " << sum << endl;
}

void APTTransformerBase::
transform(const NaturalVector& from, NaturalVector& to, bool useBias) const
{
  assert(to.subFeatLen() == _subFeatLen);

  for (int isub = 0; isub < _nSubFeat; isub++) {
    for (int n = 0; n < _subFeatLen; n++) {
      double sum = 0.0;

      if ((from.nSubFeat() == to.nSubFeat()) && (from.nSubFeat() == _nSubFeat)) {

	// Normal Case: _transMatrix is only as large as a single cepstral sub-feature
	const UnShrt olen = min(from.subFeatLen(), _orgSubFeatLen);
	for (int m = 0; m < olen; m++)
	  sum += gsl_matrix_get(_transMatrix, n, m) * from(isub, m);

      } else if (from.featLen() == orgFeatLen()) {

	// Normal LDA Case: _transMatrix is as large as entire original mean
	const UnShrt olen = from.featLen();
	for (int m = 0; m < olen; m++)
	  sum += gsl_matrix_get(_transMatrix, n, m) * from(m);

      } else {

	// Special LDA Case: _transMatrix is larger than original mean due to extension
	const UnShrt olen = orgFeatLen();
	NaturalVector fromExtended(olen, from.nSubFeat());  fromExtended = from;
	for (int m = 0; m < olen; m++)
	  sum += gsl_matrix_get(_transMatrix, n, m) * fromExtended(m);

      }

      to(isub, n) = sum;
    }
  }

  unsigned bSize = _bias.featLen();
  if (bSize == 0 || useBias == false) return;

  for (unsigned n = 0; n < bSize; n++)
    to(n) += _bias(n);
}

void APTTransformerBase::_initTransformMatrix()
{
  TransformMatrix::_initialize(_subFeatLen, _nSubFeat, _orgSubFeatLen);

  for (UnShrt s = 0; s < _nSubFeat; s++) {
    for (UnShrt m = 0; m < _subFeatLen; m++) {
      for (UnShrt n = 0; n < _orgSubFeatLen; n++) {
	gsl_matrix_set((gsl_matrix*) matrix(s), m, n, gsl_matrix_get(_transMatrix, m, n));
      }
    }
  }

  if (_bias.featLen() == 0) return;

  if (_bias.subFeatLen() != _subFeatLen)
    throw jconsistency_error("Bias size %d is incompatible with feature length %d.",
			     _bias.subFeatLen(), _subFeatLen);

  UnShrt nBiasSubFeat = _bias.nSubFeat();
  for (UnShrt s = 0; s < nBiasSubFeat; s++) {
    for (UnShrt n = 0; n < _subFeatLen; n++) {
      gsl_vector_set((gsl_vector*) offset(s), n, _bias(s, n));
    }
  }
}


// ----- methods for class `BLTTransformer' -----
//
BLTTransformer::BLTTransformer(const RAPTParam& par)
  : _alpha(par[0])
{
  bilinear(_alpha, _laurentCoeffs);
}

BLTTransformer::
BLTTransformer(UnShrt sbFtSz, UnShrt nSubFt, UnShrt orgSbFtSz)
  : TransformBase(sbFtSz, nSubFt, orgSbFtSz),
    APTTransformerBase(sbFtSz, nSubFt, orgSbFtSz), _alpha(0.0) { }

BLTTransformer::
BLTTransformer(const RAPTParam& par,
	       UnShrt sbFtSz, UnShrt nSubFt, UnShrt orgSbFtSz,
	       int trace, const  String& ldaFile, UnShrt ldaFeatLen)
  : TransformBase(par,
		  (ldaFile == "") ? sbFtSz    : ldaFeatLen,
		  (ldaFile == "") ? nSubFt    : 1,
		  (ldaFile == "") ? orgSbFtSz : (orgSbFtSz * nSubFt), trace),
    APTTransformerBase(par, sbFtSz, nSubFt, orgSbFtSz, trace, ldaFile, ldaFeatLen),
    _alpha(par[0])
{
  calcTransMatrix(_alpha);
  _initTransformMatrix();
}

BLTTransformer::
BLTTransformer(double al, UnShrt sbFtSz, UnShrt nSubFt, UnShrt orgSbFtSz)
  : TransformBase(sbFtSz, nSubFt, orgSbFtSz),
    APTTransformerBase(sbFtSz, nSubFt, orgSbFtSz),
    _alpha(al)
{
  calcTransMatrix(_alpha);
  _initTransformMatrix();
}

void BLTTransformer::bilinear(double mu, LaurentSeries& sequence)
{
  if (fabs(mu) >= 1.0)
    throw jparameter_error("Mu value (%g) is not < 1.0", mu);

  sequence = 0.0;
  
  sequence[0] = -mu;				// coefficient for z^0
  sequence[1] = (1.0 - mu * mu);		// coefficient for z^1
  for (int i = 2; i < sequence.len(); i++)	// coefficient for z^n
    sequence[i] = mu * sequence[i-1];

  if (Trace & Sequences)
    cout << "Coefficients of A(z):" << endl << sequence << endl;
}

void BLTTransformer::
_calcSequence(const CoeffSequence& _params, LaurentSeries& _sequence)
{
  bilinear(_params[0], _sequence);
}

void BLTTransformer::calcTransMatrix(double mu)
{
  bilinear(mu, _laurentCoeffs);
  _calcTransMatrix(_laurentCoeffs);
}

void BLTTransformer::unitCirclePlot(const String& fileName)
{
  if(fileName == "") return;
  
  FILE* fp = fileOpen(fileName, "w");

  bilinear(_alpha, _laurentCoeffs);
  _unitCirclePlot(fp, _laurentCoeffs);
  fileClose(fileName, fp);
}


// ----- methods for class `RAPTTransformer' -----
//
UnShrt RAPTTransformer::_setNoComplexPairs(UnShrt nParams) const
{
  if (nParams != 0 && nParams != NoAllPassParams)
    throw j_error("Inconsistent number of APT parameters (%d vs. %d).",
		  nParams, NoAllPassParams);

  if ((NoAllPassParams - 1) % 4 != 0)
    throw j_error("Incorrect number of APT parameters (%d).", NoAllPassParams);

  return (NoAllPassParams - 1) / 4;
}

RAPTTransformer::RAPTTransformer()
  : NoComplexPairs(_setNoComplexPairs()) { }

RAPTTransformer::
RAPTTransformer(const RAPTParam& par)
  : APTTransformerBase(par.size()),
    NoComplexPairs(_setNoComplexPairs(par.size()))
{
  for (unsigned i = 0; i < NoAllPassParams; i++)
    p[i] = params[i] = par[i];

  _calcSequence(params, _laurentCoeffs);
}

RAPTTransformer::
RAPTTransformer(UnShrt sbFtSz, UnShrt nSubFt, UnShrt orgSbFtSz, UnShrt noParams)
  : TransformBase(sbFtSz, nSubFt, orgSbFtSz),
    APTTransformerBase(sbFtSz, nSubFt, orgSbFtSz, noParams),
    BLTTransformer(sbFtSz, nSubFt),
    NoComplexPairs(_setNoComplexPairs()) { }

RAPTTransformer::
RAPTTransformer(const RAPTParam& par, UnShrt sbFtSz, UnShrt nSubFt, UnShrt orgSbFtSz,
		int trace, const String& ldaFile, UnShrt ldaFeatLen)
  :  TransformBase(par,
		   (ldaFile == "") ? sbFtSz    : ldaFeatLen,
		   (ldaFile == "") ? nSubFt    : 1,
		   (ldaFile == "") ? orgSbFtSz : (orgSbFtSz * nSubFt), trace),
    APTTransformerBase(par, sbFtSz, nSubFt, orgSbFtSz, trace, ldaFile, ldaFeatLen),
    BLTTransformer(sbFtSz, nSubFt),
    NoComplexPairs(_setNoComplexPairs(par.size()))
{
  for (unsigned i = 0; i < NoAllPassParams; i++)
    params[i] = par[i];

  _calcSequence(params, _laurentCoeffs);
  _calcTransMatrix(_laurentCoeffs);
  _initTransformMatrix();
}

void RAPTTransformer::unitCirclePlot(const String& fileName)
{
  if(fileName == "") return;
  
  FILE* fp = fileOpen(fileName, "w");

  _calcSequence(p, _laurentCoeffs);
  _unitCirclePlot(fp, _laurentCoeffs);
  fileClose(fileName, fp);
}

RAPTTransformer::SmallTheta RAPTTransformer::isThetaSmall(double& _theta)
{
  if (fabs(_theta) < DefaultSmallTheta) {
    return ThetaZero;
  } else if (fabs(_theta - M_PI) < DefaultSmallTheta) {
    _theta -= M_PI;
    return ThetaPi;
  } else if (fabs(_theta + M_PI) < DefaultSmallTheta) {
    _theta += M_PI;
    return ThetaPi;
  }

  return NotSmall;
}

bool RAPTTransformer::
allPassSmallTheta(double _rho, double _theta, LaurentSeries& _sequence,
		  bool _invFlag)
{
  SmallTheta smallTheta = isThetaSmall(_theta);

  if (smallTheta == NotSmall) return false;

  double rho2   = _rho * _rho;
  double rho4   = rho2 * rho2;
  double theta2 = _theta * _theta;
  double theta4 = theta2 * theta2;
  double theta6 = theta4 * theta2;

  // coefficients of z^n for all n >= 2
  double rho_n_2 = 1.0;
  for (int n = 2; n < _sequence.len(); n++) {
    int     n2 = n  * n;
    int     n3 = n2 * n;
    int     n4 = n2 * n2;
    int     n6 = n4 * n2;
    int    index = (_invFlag) ? -n : +n;
    double term  =
      (- (-15120 + (2520*theta2 - 42*(3*n2+6*n-4)*theta4
		    +(3*n4+12*n3-24*n+16)*theta6)*n*(n+2))
       * (n+1) * rho4 / 15120.0
       + (-15120 + 2520*(n2+2)*theta2 - 42*(3*n4+20*n2-8)*theta4
	  +(3*n6+42*n4-52*n2+32)*theta6) * n * rho2 / 7560.0
       - (-15120 + (2520*theta2 - 42*(3*n2-6*n-4)*theta4
		    + (3*n4-12*n3+24*n+16)*theta6)*(n-2)*n) * (n-1) / 15120.0)
      * rho_n_2;

    _sequence[index] = 
      (smallTheta == ThetaZero || (n%2) == 0) ? term : -term;
    rho_n_2 *= _rho;
  }

  if (Trace & Sequences)
    cout << "Coefficients of (small) "
   << ((_invFlag) ? "G(z):" : "B(z):") << endl
   << _sequence << endl;

  return true;
}

void RAPTTransformer::
allPass(double _rho, double _theta, LaurentSeries& _sequence, bool _invFlag)
{
  if (_rho <= 0.0 || _rho >= 1.0)
    throw j_error("Rho value (%g) is inappropriate", _rho);

  _sequence = 0.0;
  
  double rho2     = _rho * _rho;
  double rho4     = rho2 * rho2;
  double cosTheta = cos(_theta);
  double sinTheta = sin(_theta);

  _sequence[0] = rho2;				// coeff. of z^0

  int index = (_invFlag) ? -1 : +1;
  _sequence[index] =				// coeff. of z^1
    2.0 * _rho * (rho2 - 1.0) * cosTheta;

  if (allPassSmallTheta(_rho, _theta, _sequence, _invFlag)) return;

  double rho_n_2 = 1.0;
  for (int n = 2; n < _sequence.len(); n++) {  // coeff. of z^n; n >= 2

    index = (_invFlag) ? - n : + n;
    _sequence[index] = (rho_n_2 / sinTheta)
      * (rho4 * sin(_theta * (n + 1)) - 2.0 * rho2 * cosTheta * sin(_theta * n)
	 + sin(_theta * (n - 1)));

    rho_n_2 *= _rho;
  }

  if (Trace & Sequences)
    cout << "Coefficients of "
	 << ((_invFlag) ? "G(z):" : "B(z):") << endl
	 << _sequence << endl;
}

void RAPTTransformer::
_calcSequence(const CoeffSequence& _params, LaurentSeries& _sequence)
{
  double alpha  = _params[0];    // always unpack in the same order

  bilinear(alpha, _sequence);

  for (unsigned ipair = 0; ipair < NoComplexPairs; ipair++) {
    double betaReal  = _params[(4*ipair) + 1];
    double betaImag  = _params[(4*ipair) + 2];
    double gammaReal = _params[(4*ipair) + 3];
    double gammaImag = _params[(4*ipair) + 4];
    double _rho, _theta, _r, _omega;
    rect2polar(betaReal,  betaImag,  _rho, _theta);
    rect2polar(gammaReal, gammaImag, _r,   _omega);

    _temp1 = _sequence;
    allPass(_rho, _theta, _temp2);
    CauchyProduct(_temp1, _temp2, _temp3);
    allPass(_r, _omega, _temp1, /* inverseFlag= */ true);
    CauchyProduct(_temp3, _temp1, _sequence);
  }

  if (Trace & Sequences)
    cout << "Coefficients of Q(z):" << endl
	 << _sequence << endl;
}


// ----- methods for class `SLAPTTransformer' -----
//
const UnShrt  SLAPTTransformer::NoColumns = 50;

void SLAPTTransformer::_initialize() const
{
  _fmn = new LaurentSeries[NoColumns];

  if(eCoefficients().size() > 0) return;

  eCoefficients().resize(NoColumns);
  eCoefficients()[0] = eCoefficients()[1] = 1.0;
  for (UnShrt i = 2; i < NoColumns; i++)
    eCoefficients()[i] = eCoefficients()[i-1] / i;
}

SLAPTTransformer::SLAPTTransformer()
{
  _initialize();
}

SLAPTTransformer::SLAPTTransformer(const SLAPTParam& par)
  : APTTransformerBase(par.size())
{
  assert(par.size() >= 1);

  _initialize();

  for (unsigned i = 0; i < NoAllPassParams; i++)
    p[i] = params[i] = par[i];

  _calcSequence(params, _laurentCoeffs);
}

SLAPTTransformer::
SLAPTTransformer(UnShrt sbFtSz, UnShrt nSubFt, UnShrt noParams, UnShrt orgSbFtSz)
  : TransformBase(sbFtSz, nSubFt, orgSbFtSz),
    APTTransformerBase(sbFtSz, nSubFt, orgSbFtSz, noParams)
{
  _initialize();
}

SLAPTTransformer::
SLAPTTransformer(const SLAPTParam& par,
		 UnShrt sbFtSz, UnShrt nSubFt, UnShrt orgSbFtSz,
		 int trace, const String& ldaFile, UnShrt ldaFeatLen)
  : TransformBase(par,
		  (ldaFile == "") ? sbFtSz    : ldaFeatLen,
		  (ldaFile == "") ? nSubFt    : 1,
		  (ldaFile == "") ? orgSbFtSz : (orgSbFtSz * nSubFt), trace),
    APTTransformerBase(par, sbFtSz, nSubFt, orgSbFtSz, trace, ldaFile, ldaFeatLen)
{
  assert(par.size() >= 1);

  _initialize();

  for (unsigned i = 0; i < NoAllPassParams; i++)
    params[i] = par[i];

  _calcSequence(params, _laurentCoeffs);
  _calcTransMatrix(_laurentCoeffs);
  _initTransformMatrix();
}

SLAPTTransformer::~SLAPTTransformer()
{
  delete[] _fmn;
}

void SLAPTTransformer::unitCirclePlot(const String& fileName)
{
  if (fileName == "") return;
  
  FILE* fp = fileOpen(fileName, "w");

  _calcSequence(p, _laurentCoeffs);
  _unitCirclePlot(fp, _laurentCoeffs);
  fileClose(fileName, fp);
}

void SLAPTTransformer::
_calcSequence(const CoeffSequence& _params, LaurentSeries& _sequence)
{
  // 0th sequence is unit sample sequence
  _fmn[0]    = 0.0;
  _fmn[0][0] = 1.0;

  // 1st sequence given by series of coefficients
  _fmn[1] = 0.0;
  for (UnShrt i = 1; i <= NoAllPassParams; i++) {
    _fmn[1][i]  =  M_PI_2 * _params[i-1];
    _fmn[1][-i] = -M_PI_2 * _params[i-1];
  }

  for (UnShrt i = 2; i < NoColumns; i++)
    CauchyProduct(_fmn[1], _fmn[i-1], _fmn[i]);

  _sequence = 0.0;
  for (UnShrt i = 0; i < NoColumns; i++)
    multAdd(_eCoeff(i), _fmn[i], _sequence);
  shift(_sequence);

  if (Trace & Sequences)
    cout << "Coefficients of Q(z):" << endl
	 << _sequence << endl;
}


// ----- methods for class `MLLRTransformer' -----
//
MLLRTransformer::
MLLRTransformer(UnShrt sbFtSz, UnShrt nSubFt)
  : TransformBase(sbFtSz, nSubFt, sbFtSz) { }

MLLRTransformer::
MLLRTransformer(const MLLRParam& par, UnShrt sbFtSz, UnShrt nSubFt)
  : TransformBase(par, sbFtSz, nSubFt, sbFtSz)
{
  TransformMatrix::_initialize(_featLen, /* nSubFt= */ 1, _featLen);

  UnShrt len = par.size();
  if (len != sbFtSz * nSubFt)
    throw jdimension_error("Sizes (%d vs. %d) do not match.", par.size(), sbFtSz * nSubFt);

  if (_bias.featLen() != _featLen)
    throw jdimension_error("Feature lengths (%d vs. %d) do not match.", _bias.featLen(), _featLen);

  for (UnShrt m = 0; m < _featLen; m++) {
    gsl_vector_set((gsl_vector*) offset(), m, _bias[m]);
    for (UnShrt n = 0; n < _featLen; n++)
      gsl_matrix_set((gsl_matrix*) matrix(), m, n, gsl_matrix_get(par.matrix(), m, n));
  }
}

void MLLRTransformer::
transform(const NaturalVector& from, NaturalVector& to, bool useBias) const
{
  assert(from.featLen() == _featLen);
  assert(to.featLen()   == _featLen);
  assert(_orgSubFeatLen == _subFeatLen);

  for (UnShrt m = 0; m < _featLen; m++) {
    float comp = (useBias) ? gsl_vector_get(offset(), m) : 0.0;

    for (UnShrt n = 0; n < _featLen; n++)
      comp += gsl_matrix_get(matrix(), m, n) * from[n];

    to[m] = comp;
  }
}


// ----- methods for class `STCTransformer' -----
//
STCTransformer::
STCTransformer(VectorFloatFeatureStreamPtr& src, UnShrt sbFtSz, UnShrt nSubFt, const String& nm)
  : VectorFloatFeatureStream(sbFtSz * nSubFt, nm),
    TransformBase(sbFtSz, nSubFt, sbFtSz), _src(src)
{
  TransformMatrix::_initialize(_subFeatLen, _nSubFeat, _orgSubFeatLen);
}

STCTransformer::
STCTransformer(VectorFloatFeatureStreamPtr& src, const STCParam& par, UnShrt sbFtSz, UnShrt nSubFt, const String& nm)
  : VectorFloatFeatureStream(sbFtSz * nSubFt, nm),
    TransformBase(par, sbFtSz, nSubFt, sbFtSz), _src(src)
{
  TransformMatrix::_initialize(_subFeatLen, _nSubFeat, _orgSubFeatLen);

  UnShrt len = par.size();
  if (len != sbFtSz * nSubFt)
    throw jdimension_error("Sizes (%d vs. %d) do not match.", par.size(), sbFtSz * nSubFt);

  if (_bias.featLen() != _featLen)
    throw jdimension_error("Feature lengths (%d vs. %d) do not match.", _bias.featLen(), _featLen);

  for (UnShrt m = 0; m < _featLen; m++) {
    gsl_vector_set((gsl_vector*) offset(), m, _bias[m]);
    for (UnShrt n = 0; n < _featLen; n++)
      gsl_matrix_set((gsl_matrix*) matrix(), m, n, gsl_matrix_get(par.matrix(), m, n));
  }
}

void STCTransformer::
transform(const NaturalVector& from, NaturalVector& to, bool useBias) const
{
  assert(from.featLen() == _featLen);
  assert(to.featLen()   == _featLen);
  assert(_orgSubFeatLen == _subFeatLen);

  for (UnShrt m = 0; m < _featLen; m++) {
    float comp = (useBias) ? gsl_vector_get(offset(), m) : 0.0;

    for (UnShrt n = 0; n < _featLen; n++)
      comp += gsl_matrix_get(matrix(), m, n) * from[n];

    to[m] = comp;
  }
}

const gsl_vector_float* STCTransformer::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d",
		       name().c_str(), frameX - 1, _frameX);

  const gsl_vector_float* srcVec = _src->next(frameX + 1);
  _increment();

  _transform(srcVec, _vector);

  return _vector;
}

void STCTransformer::save(const String& fileName)
{
  FILE* fp = fileOpen(fileName, "w");
  gsl_matrix_fwrite(fp, matrix());
  fileClose(fileName, fp);
}

void STCTransformer::load(const String& fileName)
{
  FILE* fp = fileOpen(fileName, "r");
  gsl_matrix_fread(fp, (gsl_matrix*) matrix());
  fileClose(fileName, fp);
}

void STCTransformer::_transform(const gsl_vector_float* srcVec, gsl_vector_float* transVec)
{
  if (transVec == NULL) transVec = _vector;

  for (unsigned iblk = 0; iblk < _nSubFeat; iblk++) {
    unsigned offset = iblk * _subFeatLen;
    for (unsigned i = 0; i < _subFeatLen; i++) {
      double sum = 0.0;
      for (unsigned j = 0; j < _subFeatLen; j++)
	sum += gsl_vector_float_get(srcVec, offset + j) * gsl_matrix_get(matrix(), i, j);
      gsl_vector_float_set(transVec, i + offset, sum);
    }
  }
}


// ----- methods for class `LDATransformer' -----
//
LDATransformer::
LDATransformer(VectorFloatFeatureStreamPtr& src, UnShrt sbFtSz, UnShrt nSubFt, const String& nm)
  : VectorFloatFeatureStream(sbFtSz * nSubFt, nm),
    TransformBase(sbFtSz, nSubFt, src->size() / nSubFt), _src(src)
{
  TransformMatrix::_initialize(_src->size() / nSubFt, nSubFt, _src->size() / nSubFt);

  if (_src->size() % nSubFt != 0)
    throw jdimension_error("Source size (%d) and number of subfeatures (%d) are incompatible.",
			   _src->size(), nSubFt);

  printf("LDA Feature Input Size  = %d\n", _src->size());
  printf("LDA Feature Output Size = %d\n", size());  fflush(stdout);
}

LDATransformer::
LDATransformer(VectorFloatFeatureStreamPtr& src, const LDAParam& par, UnShrt sbFtSz, UnShrt nSubFt, const String& nm)
  : VectorFloatFeatureStream(sbFtSz * nSubFt, nm),
    TransformBase(par, sbFtSz, nSubFt, sbFtSz), _src(src)
{
  TransformMatrix::_initialize(_featLen, /* nSubFt= */ 1, _featLen);

  UnShrt len = par.size();
  if (len != sbFtSz * nSubFt)
    throw jdimension_error("Sizes (%d vs. %d) do not match.", par.size(), sbFtSz * nSubFt);

  if (_bias.featLen() != _featLen)
    throw jdimension_error("Feature lengths (%d vs. %d) do not match.", _bias.featLen(), _featLen);

  for (UnShrt m = 0; m < _featLen; m++) {
    gsl_vector_set((gsl_vector*) offset(), m, _bias[m]);
    for (UnShrt n = 0; n < _featLen; n++)
      gsl_matrix_set((gsl_matrix*) matrix(), m, n, gsl_matrix_get(par.matrix(), m, n));
  }
}

void LDATransformer::
transform(const NaturalVector& from, NaturalVector& to, bool useBias) const
{
  assert(from.featLen() == _featLen);
  assert(to.featLen()   == _featLen);
  assert(_orgSubFeatLen == _subFeatLen);

  for (UnShrt m = 0; m < _featLen; m++) {
    float comp = (useBias) ? gsl_vector_get(offset(), m) : 0.0;

    for (UnShrt n = 0; n < _featLen; n++)
      comp += gsl_matrix_get(matrix(), m, n) * from[n];

    to[m] = comp;
  }
}

const gsl_vector_float* LDATransformer::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d",
		       name().c_str(), frameX - 1, _frameX);

  const gsl_vector_float* srcVec = _src->next(frameX + 1);
  _increment();

  for (unsigned iblk = 0; iblk < _nSubFeat; iblk++) {
    unsigned offset = iblk * _orgSubFeatLen;
    for (unsigned i = 0; i < _subFeatLen; i++) {
      double sum = 0.0;
      for (unsigned j = 0; j < _orgSubFeatLen; j++)
	sum += gsl_vector_float_get(srcVec, offset + j) * gsl_matrix_get(matrix(iblk), i, j);
      gsl_vector_float_set(_vector, i + offset, sum);
    }
  }

  return _vector;
}

void LDATransformer::save(const String& fileName) const
{
  FILE* fp = fileOpen(fileName, "w");
#if 0
  gsl_matrix_fwrite(fp, matrix());
#else
  // ----- !!!!HACK!!!! -----
  gsl_matrix_float* temp = gsl_matrix_float_alloc(matrix()->size1, matrix()->size2);
  for (unsigned rowX = 0; rowX < matrix()->size1; rowX++)
    for (unsigned colX = 0; colX < matrix()->size2; colX++)
      gsl_matrix_float_set(temp, rowX, colX, gsl_matrix_get(matrix(), rowX, colX));
  gsl_matrix_float_fwrite(fp, temp);
  gsl_matrix_float_free(temp);
#endif

  fileClose(fileName, fp);
}

void LDATransformer::load(const String& fileName)
{
  FILE* fp = fileOpen(fileName, "r");
  gsl_matrix_fread(fp, (gsl_matrix*) matrix());
  fileClose(fileName, fp);
}


// ----- methods for class `SpeakerList' -----
//
SpeakerList::SpeakerList(const String& spkrList)
{
  FILE* fp = fileOpen(spkrList, "r");

  const int MaxListLength = 128;
  static char buffer[MaxListLength];

  while(fgets(buffer, MaxListLength, fp))
    _slist.push_back(strtok(buffer, "\n"));
}


// ----- methods for implementation base class `BaseTree' -----
//
BaseTree::BaseTree(CodebookSetAdaptPtr& cbs)
  : _codebookSet(cbs)
{
  // _codebookSet->setRegClasses();
}

BaseTree::~BaseTree()
{
  _nodeList.erase(_nodeList.begin(), _nodeList.end());
}

bool BaseTree::_nodePresent(UnShrt idx) const
{
  return _nodeList.find(idx) != _nodeList.end();  
}

void BaseTree::_setNode(UnShrt idx, NodePtr& n)
{
  if (idx == 0)
    throw jindex_error("Regression class index must be >= 1");

  if (_nodeList.find(idx) != _nodeList.end())
    throw jconsistency_error("Node %d already exists.", idx);

  _nodeList.insert(_ValueType(idx, n));
}

BaseTree::NodePtr& BaseTree::node(unsigned idx, bool useAncestor)
{
  if (idx == 0)
    throw jindex_error("Regression class index must be >= 1");

  _NodeListIter itr = _nodeList.find(idx);
  if (itr != _nodeList.end())
    return (*itr).second;

  if (useAncestor == false)
    throw jkey_error("Could not find node %d.", idx);

  while (idx > 1) {
    idx /= 2;
    if ((itr = _nodeList.find(idx)) != _nodeList.end())
      return (*itr).second;
  }

  throw jkey_error("Could not find node %d.", idx);
}

const BaseTree::NodePtr& BaseTree::node(unsigned idx, bool useAncestor) const
{
  if (idx == 0)
    throw jindex_error("Regression class index must be >= 1");

  _NodeListConstIter itr = _nodeList.find(idx);
  if (itr != _nodeList.end()) 
    return (*itr).second;

  if (useAncestor == false)
    throw jkey_error("Could not find node %d.", idx);

  while (idx > 1) {
    idx /= 2;
    if ((itr = _nodeList.find(idx)) != _nodeList.end())
      return (*itr).second;
  }

  throw jkey_error("Could not find node %d.", idx);
}

BaseTree::NodePtr& BaseTree::_leftChild(const NodePtr& p)
{
  return node(2 * p->index());
}

BaseTree::NodePtr& BaseTree::_rightChild(const NodePtr& p)
{
  return node(2 * p->index() + 1);
}

void BaseTree::_setNodeTypes()
{
  for (_NodeListIter itr = _nodeList.begin(); itr != _nodeList.end(); itr++) {
    NodePtr& node((*itr).second);
    node->_setType(Leaf);
  }

  for (_NodeListConstIter itr = _nodeList.begin(); itr != _nodeList.end(); itr++) {
    const NodePtr& node((*itr).second);
    if (node->type() != Leaf) continue;
    int index = (*itr).first;
    for (index = index / 2; index > 0; index /= 2) {
      _NodeListIter itr = _nodeList.find(index);
      if (itr == _nodeList.end()) continue;
      NodePtr& node((*itr).second);
      node->_setType(Internal);
    }
  }
}

void BaseTree::_validateNodeTypes() const
{
  for (_NodeListConstIter itr = _nodeList.begin(); itr != _nodeList.end(); itr++) {
    const NodePtr& node((*itr).second);
    if (node->type() != Leaf) continue;
    int index = (*itr).first;
    int leftIndex = 2*index, rightIndex = 2*index+1;

    if (_nodeList.find(leftIndex) != _nodeList.end())
      printf("Warning: Putative leaf %d has left child.\n", index);
      // throw jconsistency_error("Putative leaf %d has left child.", index);
    if (_nodeList.find(rightIndex) != _nodeList.end())
      printf("Warning: Putative leaf %d has right child.\n", index);
      // throw jconsistency_error("Putative leaf %d has right child.", index);
  }
}


// ----- methods for class `TransformerTree' -----
//
TransformerTree::TransformerTree(CodebookSetAdaptPtr& cb, ParamTreePtr& paramTree)
  : BaseTree(cb), _paramTree(paramTree) { }

TransformerTree::TransformerTree(CodebookSetAdaptPtr& cb, const String& parmFile,
				 UnShrt orgSubFeatLen)
  : BaseTree(cb), _paramTree(new ParamTree(parmFile))
{
  for (ParamTree::ConstIterator itr(_paramTree); itr.more(); itr++) {
    int rClass = itr.regClass();
    NodePtr n(new Node(*this, rClass, itr.par(), orgSubFeatLen));
    _setNode(rClass, n);
  }
  _setNodeTypes();
}

TransformerTree::TransformerTree(CodebookSetAdaptPtr& cb, const ParamTreePtr& pt,
				 UnShrt orgSubFeatLen)
  : BaseTree(cb), _paramTree(pt)
{
  for (ParamTree::ConstIterator itr(_paramTree); itr.more(); itr++) {
    int rClass = itr.regClass();
    NodePtr n(new Node(*this, rClass, itr.par(), orgSubFeatLen));
    _setNode(rClass, n);
  }
  _setNodeTypes();
}

TransformerTree& TransformerTree::transform()
{
  cout << endl << "Transforming Means" << endl;

  for (CodebookSetAdapt::GaussianIterator itr(cbs()); itr.more(); itr++) {
    CodebookAdapt::GaussDensity mix(itr.mix());
    int rClass = mix.regClass();
    NodePtr& nextNode(node(rClass, /* useAncestor= */ true));
    if (nextNode->type() != Leaf)
      Warning("Node for regression class %d is not a leaf", rClass);
    nextNode->transform(mix);
  }

  return *this;
}

const TransformBasePtr& TransformerTree::transformer(UnShrt regClass) const
{
  const NodePtr& n(node(regClass, /* useAncestor= */ false));

  if(n->type() != Leaf)
    throw jconsistency_error("Node for regression class %d is not a leaf", regClass);

  return n->transformer();
}


// ----- methods for class `TransformerTree::Node' -----
//
TransformerTree::Node::Node(TransformerTree& tr, unsigned idx, const ParamBasePtr& par,
			    UnShrt cepOrgSubFeatLen)
  : BaseTree::Node(tr, idx),
    _trans(par->transformer(tr.cepSubFeatLen(), tr.cepNSubFeat(),
			    (cepOrgSubFeatLen == 0) ? tr.cepOrgSubFeatLen() : cepOrgSubFeatLen,
			    /* trace= */ Top, tr.ldaFile(), tr.featLen())) { }

TransformerTree::Node::Node(TransformerTree& tr, unsigned idx, NodeType type, TransformBase* transformer)
  : BaseTree::Node(tr, idx, type), _trans(transformer) { }

TransformerTree::Node::~Node() { }

void TransformerTree::Node::transform(CodebookAdapt::GaussDensity& mix)
{
  _trans->transform(mix);
}

void adaptCbk(CodebookSetAdaptPtr& cbs, const ParamTreePtr& pt, UnShrt orgSubFeatLen)
{
  TransformerTree tree(cbs, pt, orgSubFeatLen);
  tree.transform();
}
