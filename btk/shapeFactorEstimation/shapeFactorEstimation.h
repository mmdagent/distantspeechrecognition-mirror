//                              -*- C++ -*-
//
//                          Beamforming Toolkit
//                                 (btk)
//
//  Module:  btk.shapeFactorEstimation
//  Purpose: Estimation of shape factors for the generalized Gaussian pdf.
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

#ifndef _shapeFactorEstimation_h_
#define _shapeFactorEstimation_h_

#include <list>
#include <vector>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>

#include "common/mlist.h"
#include "common/refcount.h"


// ----- definition for class `SubbandFeature' -----
//
class SubbandFeature {
public:
  SubbandFeature(const gsl_vector_complex* samps, const gsl_vector* vars);
  SubbandFeature(const SubbandFeature& );
  SubbandFeature(FILE* fp);
  ~SubbandFeature();

  const gsl_vector_complex* samples() const { return _samples; }
  const gsl_vector* variances() const { return _variances; }
  void write(FILE* fp) const;
  void read(FILE* fp);
  double logLhood(unsigned subbandX, double f) const;

  unsigned subbandsN() const { return _size / 2 + 1; }

private:
  unsigned					_size;
  gsl_vector_complex*				_samples;
  gsl_vector*					_variances;
};


// ----- definition for class `ShapeFactorFeatures' -----
//
class ShapeFactorFeatures : public Countable {
  typedef std::list<SubbandFeature>		_FeatureList;
  typedef _FeatureList::iterator		_FeatureListIterator;
  typedef _FeatureList::const_iterator		_FeatureListConstIterator;

  typedef std::vector<_FeatureList>		_FeatureListList;
  typedef _FeatureListList::iterator		_FeatureListListIterator;
  typedef _FeatureListList::const_iterator	_FeatureListListConstIterator;

 public:
  ShapeFactorFeatures(unsigned classesN, unsigned maxFrames = 500);
  ~ShapeFactorFeatures();

  void insert(unsigned classX, const gsl_vector_complex* samps, const gsl_vector* vars);
  void write(const String& fileName) const;
  void read(const String& fileName);
  const SubbandFeature& getFeature(unsigned stateX, unsigned timeX) const;
  unsigned featuresN(unsigned classX) const;
  void add(const ShapeFactorFeatures& fromFeatures);
  unsigned classN() const { return _classesN; }
  void clear();

  double logLhood(unsigned classX, unsigned subbandX, double f) const;

 private:
  unsigned					_classesN;
  unsigned					_maxFrames;
  _FeatureListList				_allFeatures;
};

typedef refcountable_ptr<ShapeFactorFeatures> ShapeFactorFeaturesPtr;


// ----- definition for class `ShapeFactors' -----
//
class ShapeFactors : public Countable {
  typedef std::vector<double>			_ShapeVector;
  typedef _ShapeVector::iterator		_ShapeVectorIterator;
  typedef _ShapeVector::const_iterator		_ShapeVectorConstIterator;

  typedef std::map<unsigned, _ShapeVector>	_ShapeList;
  typedef _ShapeList::iterator			_ShapeListIterator;
  typedef _ShapeList::const_iterator		_ShapeListConstIterator;

  static const double   minimum;
  static const double   lowerLimit;
  static const double   upperLimit;
  static const unsigned MaximumIterations;
  static const unsigned MinimumFeatures;
  static const double   DefaultFactor;

 public:
  ShapeFactors(const ShapeFactorFeaturesPtr& features = NULL);

  void estimate(unsigned classX);
  void read(const String& fileName);
  void write(const String& fileName) const;
  void writeParam(const String& fileName, unsigned classX, unsigned subX) const;
  void clear();

  const _ShapeVector& operator[](unsigned classX) const;

private:
  ShapeFactorFeaturesPtr			_features;
  _ShapeList					_allFactors;
};

typedef refcountable_ptr<ShapeFactors> ShapeFactorsPtr;

#endif
