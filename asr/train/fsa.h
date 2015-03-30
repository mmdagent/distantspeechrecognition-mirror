//                              -*- C++ -*-
//
//                              Millennium
//                   Distant Speech Recognition System
//                                 (dsr)
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


#ifndef _fsa_h_
#define _fsa_h_

#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

#include "gaussian/codebookBasic.h"
#include "gaussian/distribBasic.h"
#include "path/distribPath.h"
#include "lattice/lattice.h"


// ----- definition for class `FeatureSpaceAdaptationFeature' -----
//
class FeatureSpaceAdaptationFeature : public VectorFloatFeatureStream {
  class SortElement;
  static int compareSortElement(const void* x, const void* y);

 public:
  FeatureSpaceAdaptationFeature(VectorFloatFeatureStreamPtr& src,
				unsigned maxAccu = 1, unsigned maxTran = 1,
				const String& nm = "FeatureSpaceAdaptationFeature");
  ~FeatureSpaceAdaptationFeature();

  // accessor methods
  unsigned topN() const { return _topN; }
  void	   setTopN(unsigned topN) { _topN = topN; }
  double   count(unsigned accuX = 0) const { return _count[accuX]; }

  float    shift() const { return _shift; }
  void     setShift(float shift) { _shift = shift; }
  void	   distribSet(DistribSetBasicPtr& dssP) { _dss = dssP; _initialize(); }

  // feature hierarchy methods 
  const gsl_vector_float* next(int frameX = -5);
  virtual void reset() { _src->reset(); VectorFloatFeatureStream::reset(); }

  // accumulation methods
  void	accumulate(DistribPathPtr& path, unsigned accuX = 0, float factor = 1.0);
  void	accumulateLattice(LatticePtr& lat, unsigned accuX = 0, float factor = 1.0);

  void	zeroAccu(unsigned accuX = 0);
  void	scaleAccu(float scale, unsigned accuX = 0);
  void	addAccu(unsigned accuX, unsigned accuY, float factor = 1.0);

  // estimation methods
  void	add(unsigned dsX);
  void	addAll();
  void	estimate(unsigned iterN = 10, unsigned accuX = 0, unsigned tranX = 0);

  void	clear(unsigned tranX = 0);
  float compareTransform(unsigned tranX, unsigned tranY);

  // read and write methods
  void	saveAccu(const String& name, unsigned accuX = 0);
  void	loadAccu(const String& name, unsigned accuX = 0, float factor = 1.0);

  void	save(const String& name, unsigned tranX = 0);
  void	load(const String& name, unsigned tranX = 0);

 private:
  static const double MaxSingularValueRatio;

  void   _initialize();
  void   _accuOne(unsigned dsX, int frameX, unsigned accuX, float gamma);
  void	 _makeSymmetric(gsl_matrix* mat);
  double _calcCofactor(const gsl_matrix* w, unsigned i0, gsl_vector* p);

  VectorFloatFeatureStreamPtr			_src;
  DistribSetBasicPtr				_dss;

  unsigned					_topN;
  float						_shift;
  
  gsl_matrix**					_w;

  float*					_beta;
  double*					_count;
  gsl_matrix**					_z;
  gsl_matrix***					_Gi;

  bool*						_dsXA;
  unsigned					_dssN;
  unsigned					_dimN;
  unsigned					_featX;
  
  float*					_addCount;
  SortElement*					_gammaX;

  unsigned					_maxAccu;	// No of accus/trfs allocated
  unsigned					_maxTran;

  gsl_matrix**					_U;
  gsl_matrix**					_V;
  gsl_vector**					_singularVals;

  gsl_vector*					_d;
  gsl_vector*					_p;
  gsl_vector*					_workspace;
  gsl_vector*					_workspaceCof;

  gsl_permutation*				_permutation;
  gsl_matrix*					_cofactorU;
  gsl_matrix*					_cofactorV;
  gsl_vector*					_cofactorS;
  gsl_matrix*					_inverseMatrix;
};

typedef Inherit<FeatureSpaceAdaptationFeature, VectorFloatFeatureStreamPtr> FeatureSpaceAdaptationFeaturePtr;


// ----- definition for class `FeatureSpaceAdaptationFeature::SortElement' -----
// 
class FeatureSpaceAdaptationFeature::SortElement {
 public:
  SortElement() { }
  SortElement(unsigned refX, float gamma)
    : _refX(refX), _gamma(gamma) { }

  unsigned		_refX;
  float			_gamma;
};

#endif

