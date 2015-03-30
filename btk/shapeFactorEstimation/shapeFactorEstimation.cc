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

#include "common/mach_ind_io.h"
#include "shapeFactorEstimation/shapeFactorEstimation.h"

#include <math.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_min.h>

static double Bc(double f)
{
  // printf("x = %g\n", f);  fflush(stdout);
  double retvalue = sqrt(gsl_sf_gamma(2.0 / f) / gsl_sf_gamma(4.0 / f));
  // printf("B_c = %g for x = %g\n", retvalue, f);
  return retvalue;
}

static double logLikelihoodGG(double f, double sigma2, gsl_complex sample)
{
  double sigma = sqrt(sigma2);
  double bc    = Bc(f);
  double lhood = log((2.0 * M_PI / f) * gsl_sf_gamma(2.0 / f) * bc * bc * sigma2);
  lhood       += pow(gsl_complex_abs(sample) / (bc * sigma), f);

  return lhood;
}

typedef struct { ShapeFactorFeaturesPtr features; unsigned classX; unsigned subbandX; } ShapeFactorParameters;

static double shapeFactor_f(double f, void* sfparams)
{
  ShapeFactorParameters* shapeFactorParameters = (ShapeFactorParameters*) sfparams;
  ShapeFactorFeaturesPtr features(shapeFactorParameters->features);
  unsigned classX   = shapeFactorParameters->classX;
  unsigned subbandX = shapeFactorParameters->subbandX;

  double lhood = features->logLhood(classX, subbandX, f);

  // printf("classX = %d : subbandX = %d : func = %g\n", classX, subbandX, f);

  return lhood;
}

// ----- methods for class `SubbandFeature' -----
//
SubbandFeature::SubbandFeature(const gsl_vector_complex* samps, const gsl_vector* vars)
  : _size(samps->size), _samples(gsl_vector_complex_alloc(_size)), _variances(gsl_vector_alloc(_size / 2 + 1))
{
  gsl_vector_complex_memcpy(_samples, samps);
  gsl_vector_memcpy(_variances, vars);
}

SubbandFeature::SubbandFeature(const SubbandFeature& feat)
  : _size(feat._size), _samples(gsl_vector_complex_alloc(_size)), _variances(gsl_vector_alloc(_size / 2 + 1))
{
  gsl_vector_complex_memcpy(_samples, feat._samples);
  gsl_vector_memcpy(_variances, feat._variances);
}

SubbandFeature::SubbandFeature(FILE* fp)
  : _size(0), _samples(NULL), _variances(NULL)
{
  read(fp);
}

SubbandFeature::~SubbandFeature()
{
  gsl_vector_complex_free(_samples);
  gsl_vector_free(_variances);
}

void SubbandFeature::write(FILE* fp) const
{
  write_int(fp, _size);
  gsl_vector_complex_fwrite(fp, _samples);
  gsl_vector_fwrite(fp, _variances);
}

void SubbandFeature::read(FILE* fp)
{
  if (_samples == NULL) {
    _size      = read_int(fp);
    _samples   = gsl_vector_complex_alloc(_size);
    _variances = gsl_vector_alloc(_size / 2 + 1);
  } else {
    unsigned sz = read_int(fp);
    if (sz != _size)
      throw jconsistency_error("Mismatch in feature sizes (%d vs. %d).", sz, _size);
  }

  gsl_vector_complex_fread(fp, _samples);
  gsl_vector_fread(fp, _variances);
}

double SubbandFeature::logLhood(unsigned subbandX, double f) const
{
  return logLikelihoodGG(f, gsl_vector_get(_variances, subbandX), gsl_vector_complex_get(_samples, subbandX));
}

// ----- methods for class `ShapeFactorFeatures' -----
//
ShapeFactorFeatures::ShapeFactorFeatures(unsigned classesN, unsigned maxFrames)
  : _classesN(classesN), _maxFrames(maxFrames), _allFeatures(_classesN) { }

ShapeFactorFeatures::~ShapeFactorFeatures()
{
}

void ShapeFactorFeatures::insert(unsigned classX, const gsl_vector_complex* samps, const gsl_vector* vars)
{
  /*
  printf("subband samples has length %d\n", samps->size);
  printf("vars has length %d\n", vars->size);
  */

  _FeatureList& flist(_allFeatures[classX]);

  if (_maxFrames > 0 && flist.size() > _maxFrames) return;

  flist.push_back(SubbandFeature(samps, vars));
}

void ShapeFactorFeatures::write(const String& fileName) const
{
  FILE* fp = fileOpen(fileName, "w");
  write_int(fp, _classesN);
  for (unsigned classX = 0; classX < _classesN; classX++) {
    const _FeatureList& flist(_allFeatures[classX]);
    unsigned featuresN = flist.size();
    write_int(fp, classX);
    write_int(fp, featuresN);
    for (_FeatureListConstIterator itr = flist.begin(); itr != flist.end(); itr++) {
      const SubbandFeature& feature(*itr);
      feature.write(fp);
    }
  }
  fileClose(fileName, fp);
}

void ShapeFactorFeatures::read(const String& fileName)
{
  cout << "Reading subband features from file " << fileName << endl;

  FILE* fp = fileOpen(fileName, "r");
  unsigned classesN = read_int(fp);
  if (classesN != _classesN)
    throw jconsistency_error("Mismatch in number of classes (%d vs. %d).",
			     classesN, _classesN);

  for (unsigned classX = 0; classX < _classesN; classX++) {
    _FeatureList& flist(_allFeatures[classX]);

    unsigned clX = read_int(fp);
    if (clX != classX)
      throw jconsistency_error("Mismatch in class index (%d vs. %d).",
			       clX, classX);

    unsigned featuresN = read_int(fp);
    cout << "Class " << classX << " : Features " << featuresN << endl;
    for (unsigned featX = 0; featX < featuresN; featX++) {
      flist.push_back(SubbandFeature(fp));
    }
  }
  fileClose(fileName, fp);
}

void ShapeFactorFeatures::clear()
{
  for (_FeatureListListIterator citr = _allFeatures.begin(); citr != _allFeatures.end(); citr++) {
    _FeatureList& flist(*citr);
    flist.erase(flist.begin(), flist.end());
  }
}

const SubbandFeature& ShapeFactorFeatures::getFeature(unsigned stateX, unsigned timeX) const
{
  const _FeatureList& flist(_allFeatures[stateX]);
  _FeatureListConstIterator itr = flist.begin();
  for (unsigned t = 0; t <= timeX; t++) itr++;

  return (*itr);
}

unsigned ShapeFactorFeatures::featuresN(unsigned stateX) const
{
  return _allFeatures[stateX].size();
}

void ShapeFactorFeatures::add(const ShapeFactorFeatures& fromFeatures)
{
  if (fromFeatures._classesN != _classesN)
    throw jconsistency_error("Mismatch in number of classes (%d vs. %d).",
			     fromFeatures._classesN, _classesN);

  for (unsigned classX = 0; classX < _classesN; classX++) {
    const _FeatureList& fromList(fromFeatures._allFeatures[classX]);
    _FeatureList& toList(_allFeatures[classX]);

    for (_FeatureListConstIterator itr = fromList.begin(); itr != fromList.end(); itr++) {
      toList.push_back(SubbandFeature(*itr));
    }
  }
}

double ShapeFactorFeatures::logLhood(unsigned classX, unsigned subbandX, double f) const
{
  double lhood = 0.0;
  const _FeatureList& flist(_allFeatures[classX]);
  for (_FeatureListConstIterator itr = flist.begin(); itr != flist.end(); itr++)
    lhood += (*itr).logLhood(subbandX, f);

  return lhood;
}

// ----- methods for class `ShapeFactors' -----
//
ShapeFactors::ShapeFactors(const ShapeFactorFeaturesPtr& features)
  : _features(features) { }

const ShapeFactors::_ShapeVector& ShapeFactors::operator[](unsigned classX) const
{
  _ShapeListConstIterator itr = _allFactors.find(classX);
  if (itr == _allFactors.end())
    throw jconsistency_error("Found no shape factors for class %d.", classX);

  return (*itr).second;
}

void ShapeFactors::clear()
{
  _allFactors.erase(_allFactors.begin(), _allFactors.end());
}

void ShapeFactors::read(const String& fileName)
{
  FILE* fp = fileOpen(fileName, "r");
  unsigned classesN = read_int(fp);
  for (unsigned clX = 0; clX < classesN; clX++) {
    unsigned classX   = read_int(fp);
    unsigned factorsN = read_int(fp);
    _ShapeVector factors(factorsN);
    for (unsigned facX = 0; facX < factorsN; facX++) {
      factors[facX] = read_float(fp);
    }
    _allFactors[classX] = factors;
  }
  fileClose(fileName, fp);
}

void ShapeFactors::write(const String& fileName) const
{
  FILE* fp = fileOpen(fileName, "w");
  unsigned classesN = _allFactors.size();
  write_int(fp, classesN);
  for (_ShapeListConstIterator itr = _allFactors.begin(); itr != _allFactors.end(); itr++) {
    unsigned classX = (*itr).first;
    const _ShapeVector& factors((*itr).second);
    unsigned factorsN = factors.size();
    write_int(fp, classX);
    write_int(fp, factorsN);
    for (unsigned i = 0; i < factorsN; i++) {
      write_float(fp, factors[i]);
    }
  }
  fileClose(fileName, fp);
}

void ShapeFactors::writeParam(const String& fileName, unsigned classX, unsigned subX) const
{
  FILE* fp = fileOpen(fileName, "w");
  _ShapeListConstIterator itr = _allFactors.find(classX);
  const _ShapeVector& flist((*itr).second);
  double sigma = 1.0;
  double f     = flist[subX];
  double mean  = 0.0;
  double A     = sigma * Bc(f);
  double g3    = gsl_sf_gamma(1.0 + 1.0 / f);
  double val   = log(2 * g3);
  double lNF   = -(val + log(A));

  fprintf(fp, "%e %e %e\n", sigma, f, mean);
  fprintf(fp, "%e %e\n", A, lNF);
  fileClose(fileName, fp);
}

const double   ShapeFactors::minimum           = 0.3;
const double   ShapeFactors::lowerLimit        = 0.1;
const double   ShapeFactors::upperLimit        = 2.0;
const unsigned ShapeFactors::MaximumIterations =  20;
const unsigned ShapeFactors::MinimumFeatures   = 200;
const double   ShapeFactors::DefaultFactor     = 1.0;

void ShapeFactors::estimate(unsigned classX)
{
  unsigned featN = _features->featuresN(classX);
  if (featN < MinimumFeatures) {
    printf("Class %d has only %d features ... \n", classX, featN);  fflush(stdout);  return;
  }

  const gsl_min_fminimizer_type* T = gsl_min_fminimizer_brent;
  gsl_min_fminimizer* s = gsl_min_fminimizer_alloc(T);

  const SubbandFeature& subbandFeature(_features->getFeature(classX, /* timeX= */ 0));
  unsigned subbandsN = subbandFeature.subbandsN();
  _ShapeVector factors(subbandsN);
  for (unsigned subX = 0; subX < subbandsN; subX++) {
    ShapeFactorParameters sfparams = {_features, classX, subX};
    gsl_function F;
    F.function = &shapeFactor_f;
    F.params   = &sfparams;
    double ll = lowerLimit, ul = upperLimit, mn = minimum;
    double fl = shapeFactor_f(ll, &sfparams), fu = shapeFactor_f(ul, &sfparams), fm;
    int status = gsl_min_find_bracket(&F, &mn, &fm, &ll, &fl, &ul, &fu, /* eval_max= */ 10);
    if (status == GSL_FAILURE) {
	factors[subX] = DefaultFactor;
	printf("classX = %d : subbandX = %d : failed to bracket minimum : setting f = %g\n",
	       classX, subX, DefaultFactor);
	continue;
    }
    status = gsl_min_fminimizer_set(s, &F, mn, ll, ul);
      
    unsigned itnX = 0;
    do {
      itnX++;
      status = gsl_min_fminimizer_iterate(s);

      double m = gsl_min_fminimizer_x_minimum(s);
      double a = gsl_min_fminimizer_x_lower(s);
      double b = gsl_min_fminimizer_x_upper(s);

      status = gsl_min_test_interval(a, b, 0.001, 0.0);

      if (status == GSL_SUCCESS) {
	double f = (a + b) / 2.0;
	printf("For classX = %d, subbandX = %d converged to f = %10.4f in %d iterations.\n",
	       classX, subX, f, itnX);
	factors[subX] = f;
      }
    } while (status == GSL_CONTINUE && itnX < MaximumIterations);
  }

  _allFactors[classX] = factors;
  gsl_min_fminimizer_free(s);
}
