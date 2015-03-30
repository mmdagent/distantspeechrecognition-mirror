//
//                       Beamforming Toolkit
//                              (btk)
//
//  Module:  btk.beamforming
//  Purpose: Beamforming and speaker tracking with a spherical microphone array.
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

#include "beamformer/tracker.h"

#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_trig.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_legendre.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <algorithm>


// ----- methods for class `BaseDecomposition::SubbandEntry' -----
//
BaseDecomposition::SubbandEntry::SubbandEntry(unsigned subbandX, const gsl_complex& Bkl)
  : _subbandX(subbandX), _bkl(Bkl)
{
  // printf("Subband %4d : Magnitude %8.2f\n", _subbandX, gsl_complex_abs(_bkl));  fflush(stdout);
}

MemoryManager<BaseDecomposition::SubbandEntry>& BaseDecomposition::SubbandEntry::memoryManager()
{
  static MemoryManager<SubbandEntry> _MemoryManager("SubbandEntry::memoryManager");
  return _MemoryManager;
}


// ----- methods for class `BaseDecomposition::SubbandEntry' -----
//
BaseDecomposition::SubbandList::SubbandList(const gsl_vector_complex* bkl, unsigned useSubbandsN)
  : _subbandsN(bkl->size), _useSubbandsN(useSubbandsN == 0 ? _subbandsN : useSubbandsN), _subbands(new SubbandEntry*[_subbandsN])
{
  // create and sort a new list of subband entries
  for (unsigned subX = 0; subX < _subbandsN; subX++)
    _subbands[subX] = new SubbandEntry(subX, gsl_vector_complex_get(bkl, subX));

  sort(_subbands, _subbands + _subbandsN, GreaterThan());

  /*
  for (unsigned subX = 0; subX < _useSubbandsN; subX++) {
    const SubbandEntry& subbandEntry(*(_subbands[subX]));
    gsl_complex Bkl		= subbandEntry.bkl();
    unsigned subbandX		= subbandEntry.subbandX();
    printf("Subband %4d : Magnitude %8.2f\n", subbandX, gsl_complex_abs(Bkl));
  }
  fflush(stdout);
  */
}

BaseDecomposition::SubbandList::~SubbandList()
{
  // cout << "Deallocating 'SubbandList'." << endl;

  for (unsigned subX = 0; subX < _subbandsN; subX++)
    delete _subbands[subX];
  delete[] _subbands;
}


// ----- methods for class `BaseDecomposition' -----
//
const unsigned    BaseDecomposition::_StateN		= 2;
const unsigned    BaseDecomposition::_ChannelsN		= 32;
const double      BaseDecomposition::_SpeedOfSound	= 343740.0;
const gsl_complex BaseDecomposition::_ComplexZero	= gsl_complex_rect(0.0, 0.0);
const gsl_complex BaseDecomposition::_ComplexOne	= gsl_complex_rect(1.0, 0.0);

BaseDecomposition::BaseDecomposition(unsigned orderN, unsigned subbandsN, double a, double sampleRate, unsigned useSubbandsN, bool spatial)
  : _orderN(orderN), _modesN((_orderN + 1) * (_orderN + 1)), _subbandsN(subbandsN), _subbandsN2(subbandsN / 2),
    _useSubbandsN((useSubbandsN == 0) ? (_subbandsN2 + 1) : useSubbandsN), _subbandLengthN(spatial ? _ChannelsN : _modesN),
    _a(a), _sampleRate(sampleRate),
    _bn(new gsl_vector_complex*[_subbandsN2 + 1]), _theta_s(gsl_vector_calloc(_ChannelsN)), _phi_s(gsl_vector_calloc(_ChannelsN)),
    _sphericalComponent(new gsl_vector_complex*[_modesN]),
    _bkl(gsl_vector_complex_calloc(_subbandsN2 + 1)),
    _dbkl_dtheta(gsl_vector_complex_calloc(_subbandsN2 + 1)), _dbkl_dphi(gsl_vector_complex_calloc(_subbandsN2 + 1)),
    _gkl(new gsl_vector_complex*[_subbandsN2 + 1]),
    _dgkl_dtheta(new gsl_vector_complex*[_subbandsN2 + 1]), _dgkl_dphi(new gsl_vector_complex*[_subbandsN2 + 1]),
    _vkl(gsl_vector_complex_calloc(_subbandLengthN)),
    _Hbar_k(gsl_matrix_complex_calloc(_useSubbandsN * _subbandLengthN, _StateN)), _yhat_k(gsl_vector_complex_calloc(_useSubbandsN * _subbandLengthN)),
    _subbandList(NULL)
{
  _setEigenMikeGeometry();

  // _bn contains the factors 4 \pi i^n b_n(ka)
  for (unsigned subbandX = 0; subbandX <= _subbandsN2; subbandX++) {
    double ka = 2.0 * M_PI * subbandX * _a * _sampleRate / (_subbandsN * _SpeedOfSound);
    _bn[subbandX] = gsl_vector_complex_calloc(_orderN + 1);
    for (int n = 0; n <= _orderN; n++) {
      gsl_complex in = _calc_in(n);
      const gsl_complex factor = gsl_complex_mul(gsl_complex_mul(gsl_complex_rect(4.0 * M_PI, 0.0), in), modalCoefficient(n, ka));
      gsl_vector_complex_set(_bn[subbandX], n, factor);
    }
  }

  // _sphericalComponent[idx] contains the spherical components Y^m_n(\theta_s, \phi_s) for each channel s
  unsigned idx = 0;
  for (int n = 0; n <= _orderN; n++) {
    for (int m = -n; m <= n; m++) {
      _sphericalComponent[idx] = gsl_vector_complex_calloc(_ChannelsN);
      for (unsigned chanX = 0; chanX < _ChannelsN; chanX++) {
	double theta_s = gsl_vector_get(_theta_s, chanX);
	double phi_s   = gsl_vector_get(_phi_s, chanX);
	gsl_complex Ynm = gsl_complex_conjugate(harmonic(n, m, theta_s, phi_s));
	gsl_vector_complex_set(_sphericalComponent[idx], chanX, Ynm);
      }
      idx++;
    }
  }

  // work space for storing $\mathbf{g}_{k,l}(\theta, \phi)$ and its derivatives w.r.t. $\theta$ and $\phi$
  for (unsigned subbandX = 0; subbandX <= _subbandsN2; subbandX++) {
    _gkl[subbandX] = gsl_vector_complex_calloc(_subbandLengthN);
    _dgkl_dtheta[subbandX] = gsl_vector_complex_calloc(_subbandLengthN);
    _dgkl_dphi[subbandX] = gsl_vector_complex_calloc(_subbandLengthN);
  }
}

BaseDecomposition::~BaseDecomposition()
{
  for (unsigned subbandX = 0; subbandX <= _subbandsN2; subbandX++)
    gsl_vector_complex_free(_bn[subbandX]);
  delete[] _bn;

  gsl_vector_free(_theta_s);
  gsl_vector_free(_phi_s);

  unsigned idx = 0;
  for (int n = 0; n <= _orderN; n++) {
    for (int m = -n; m <= n; m++) {
      gsl_vector_complex_free(_sphericalComponent[idx]);
      idx++;
    }
  }

  gsl_vector_complex_free(_bkl);
  gsl_vector_complex_free(_dbkl_dtheta);
  gsl_vector_complex_free(_dbkl_dphi);
  gsl_vector_complex_free(_vkl);
  gsl_matrix_complex_free(_Hbar_k);
  gsl_vector_complex_free(_yhat_k);

  for (unsigned subbandX = 0; subbandX <= _subbandsN2; subbandX++) {
    gsl_vector_complex_free(_gkl[subbandX]);
    gsl_vector_complex_free(_dgkl_dtheta[subbandX]);
    gsl_vector_complex_free(_dgkl_dphi[subbandX]);
  }
  delete[] _gkl;  delete[] _dgkl_dtheta;  delete[] _dgkl_dphi;
}

gsl_complex BaseDecomposition::modalCoefficient(unsigned order, unsigned subbandX) const
{
  return gsl_vector_complex_get(_bn[subbandX], order);
}

gsl_complex BaseDecomposition::_calculate_Gnm(unsigned subbandX, int n, int m, double theta, double phi)
{
  gsl_complex Gnm = gsl_complex_mul(gsl_vector_complex_get(_bn[subbandX], n), harmonic(n, m, theta, phi));

  return Gnm;
}

gsl_complex BaseDecomposition::_calculate_dGnm_dtheta(unsigned subbandX, int n, int m, double theta, double phi)
{
  gsl_complex Gnm;

  return Gnm;
}


/**
   @brief set the geometry of the EigenMikeR
*/
void BaseDecomposition::_setEigenMikeGeometry()
{ 
   gsl_vector_set(_theta_s, 0, 69.0 * M_PI / 180.0);
   gsl_vector_set(_phi_s,   0, 0.0);

   gsl_vector_set(_theta_s, 1, 90.0 * M_PI / 180.0);
   gsl_vector_set(_phi_s,   1, 32.0 * M_PI / 180.0);

   gsl_vector_set(_theta_s, 2, 111.0 * M_PI / 180.0);
   gsl_vector_set(_phi_s,   2, 0.0);

   gsl_vector_set(_theta_s, 3,  90.0 * M_PI / 180.0);
   gsl_vector_set(_phi_s,   3, 328.0 * M_PI / 180.0);

   gsl_vector_set(_theta_s, 4, 32.0 * M_PI / 180.0);
   gsl_vector_set(_phi_s,   4, 0.0);

   gsl_vector_set(_theta_s, 5, 55.0 * M_PI / 180.0);
   gsl_vector_set(_phi_s,   5, 45.0 * M_PI / 180.0);

   gsl_vector_set(_theta_s, 6, 90.0 * M_PI / 180.0);
   gsl_vector_set(_phi_s,   6, 69.0 * M_PI / 180.0);

   gsl_vector_set(_theta_s, 7, 125.0 * M_PI / 180);
   gsl_vector_set(_phi_s,   7, 45.0  * M_PI / 180);

   gsl_vector_set(_theta_s, 8, 148.0 * M_PI / 180);
   gsl_vector_set(_phi_s,   8, 0.0);

   gsl_vector_set(_theta_s, 9, 125.0 * M_PI / 180.0);
   gsl_vector_set(_phi_s,   9, 315.0 * M_PI / 180.0);

   gsl_vector_set(_theta_s, 10,  90.0 * M_PI / 180.0);
   gsl_vector_set(_phi_s,   10, 291.0 * M_PI / 180.0);

   gsl_vector_set(_theta_s, 11,  55.0 * M_PI / 180.0);
   gsl_vector_set(_phi_s,   11, 315.0 * M_PI / 180.0);

   gsl_vector_set(_theta_s, 12, 21.0 * M_PI / 180.0);
   gsl_vector_set(_phi_s,   12, 91.0 * M_PI / 180.0);

   gsl_vector_set(_theta_s, 13, 58.0 * M_PI / 180.0);
   gsl_vector_set(_phi_s,   13, 90.0 * M_PI / 180.0);

   gsl_vector_set(_theta_s, 14, 121.0 * M_PI / 180.0);
   gsl_vector_set(_phi_s,   14,  90.0 * M_PI / 180.0);

   gsl_vector_set(_theta_s, 15, 159.0 * M_PI / 180.0);
   gsl_vector_set(_phi_s,   15,  89.0 * M_PI / 180.0);

   gsl_vector_set(_theta_s, 16,  69.0 * M_PI / 180.0);
   gsl_vector_set(_phi_s,   16, 180.0 * M_PI / 180.0);

   gsl_vector_set(_theta_s, 17,  90.0 * M_PI / 180.0);
   gsl_vector_set(_phi_s,   17, 212.0 * M_PI / 180.0);

   gsl_vector_set(_theta_s, 18, 111.0 * M_PI / 180.0);
   gsl_vector_set(_phi_s,   18, 180.0 * M_PI / 180.0);

   gsl_vector_set(_theta_s, 19,  90.0 * M_PI / 180.0);
   gsl_vector_set(_phi_s,   19, 148.0 * M_PI / 180.0);

   gsl_vector_set(_theta_s, 20,  32.0 * M_PI / 180.0);
   gsl_vector_set(_phi_s,   20, 180.0 * M_PI / 180.0);

   gsl_vector_set(_theta_s, 21,  55.0 * M_PI / 180.0);
   gsl_vector_set(_phi_s,   21, 225.0 * M_PI / 180.0);

   gsl_vector_set(_theta_s, 22,  90.0 * M_PI / 180.0);
   gsl_vector_set(_phi_s,   22, 249.0 * M_PI / 180.0);

   gsl_vector_set(_theta_s, 23, 125.0 * M_PI / 180.0);
   gsl_vector_set(_phi_s,   23, 225.0 * M_PI / 180.0);
   
   gsl_vector_set(_theta_s, 24, 148.0 * M_PI / 180.0);
   gsl_vector_set(_phi_s,   24, 180.0 * M_PI / 180.0);

   gsl_vector_set(_theta_s, 25, 125.0 * M_PI / 180.0);
   gsl_vector_set(_phi_s,   25, 135.0 * M_PI / 180.0);

   gsl_vector_set(_theta_s, 26,  90.0 * M_PI / 180.0);
   gsl_vector_set(_phi_s,   26, 111.0 * M_PI / 180.0);

   gsl_vector_set(_theta_s, 27,  55.0 * M_PI / 180.0);
   gsl_vector_set(_phi_s,   27, 135.0 * M_PI / 180.0);

   gsl_vector_set(_theta_s, 28,  21.0 * M_PI / 180.0);
   gsl_vector_set(_phi_s,   28, 269.0 * M_PI / 180.0);

   gsl_vector_set(_theta_s, 29,  58.0 * M_PI / 180.0);
   gsl_vector_set(_phi_s,   29, 270.0 * M_PI / 180.0);

   gsl_vector_set(_theta_s, 30, 122.0 * M_PI / 180.0);
   gsl_vector_set(_phi_s,   30, 270.0 * M_PI / 180.0);

   gsl_vector_set(_theta_s, 31, 159.0 * M_PI / 180.0);
   gsl_vector_set(_phi_s,   31, 271.0 * M_PI / 180.0);

   /*
   for (unsigned s = 0; s < 32; s++)
     fprintf(stderr,"%02d : %6.2f %6.2f\n", s, gsl_vector_get(_theta_s, s),gsl_vector_get(_phi_s,s));
   */
}

gsl_complex BaseDecomposition::_calc_in(int n)
{
  int modulo = n % 4;
  switch (modulo) {
  case 0:
    return gsl_complex_rect(1.0, 0.0);
  case 1:
    return gsl_complex_rect(0.0, 1.0);
  case 2:
    return gsl_complex_rect(-1.0, 0.0);
  case 3:
    return gsl_complex_rect(0.0, -1.0);
  }
}

void BaseDecomposition::reset() { }

#if 1

// calculate \bar{Y}^m_n(\theta, \phi)
gsl_complex BaseDecomposition::harmonic(int order, int degree, double theta, double phi)
{
  if (order < degree || order < -degree)
    fprintf(stderr, "The order must be less than the degree but %d > %dn", order, degree);

  int status;
  gsl_sf_result sphPnm;
  if (degree >= 0) {
    // \sqrt{(2n+1)/(4\pi)} \sqrt{(n-m)!/(n+m)!} P_n^m(x), and derivatives, m >= 0, n >= m, |x| <= 1
    status = gsl_sf_legendre_sphPlm_e(order /* =n */,  degree /* =m */, cos(theta), &sphPnm);
  } else {
    status = gsl_sf_legendre_sphPlm_e(order /* =n */, -degree /* =m */, cos(theta), &sphPnm);
    if(((-degree) % 2) == 1)
      sphPnm.val = -sphPnm.val;
  }

  gsl_complex Ynm = gsl_complex_mul_real(gsl_complex_polar(1.0, -degree*phi), sphPnm.val);
  //fprintf(stderr,"%e %e \n", sphPnm.val, sphPnm.err); 

  return Ynm;
}

#else

// calculate \bar{Y}^m_n(\theta, \phi)
gsl_complex BaseDecomposition::harmonic(int order, int degree, double theta, double phi)
{
  if (order < degree || order < -degree)
    fprintf(stderr, "The order must be less than the degree but %d > %dn", order, degree);

  gsl_sf_result sphPnm;
  int deg	  = abs(degree);
  // \sqrt{(2n+1)/(4\pi)} \sqrt{(n-m)!/(n+m)!} P_n^m(x), and derivatives, m >= 0, n >= m, |x| <= 1
  int status	  = gsl_sf_legendre_sphPlm_e(order /* =n */,  deg /* =m */, cos(theta), &sphPnm);
  gsl_complex Ynm = gsl_complex_mul_real(gsl_complex_polar(1.0, -deg * phi), sphPnm.val);
  if (degree < 0) {
    // gsl_complex Ynm = gsl_complex_conjugate(Ynm);
    if (((-degree) % 2) == 1)
      Ynm = gsl_complex_mul_real(Ynm, -1.0);
  }

  return Ynm;
}

#endif

// retrieve previously stored spherical harmonic for 'channelX'
gsl_complex BaseDecomposition::harmonic(int order, int degree, unsigned channelX) const
{
  unsigned idx = 0;
  for (int n = 0; n <= _orderN; n++) {
    for (int m = -n; m <= n; m++) {
      if (n == order && m == degree)
	return gsl_vector_complex_get(_sphericalComponent[idx], channelX);
      idx++;
    }
  }
}

#if 1

// calculate the normalization factor \sqrt{(4\pi)/(2n+1)} \sqrt{(n-m)!/(n+m)!}
double BaseDecomposition::_calculateNormalization(int order, int degree)
{
  double norm	= sqrt((2 * order + 1) / (4.0 * M_PI));
  double factor = 1.0;

  if (degree >= 0) {
    int m = degree;
    while (m > -degree) {
      factor *= (order + m);
      m--;
    }
    norm /= sqrt(factor);
  } else {
    int m = -degree;
    while (m > degree) {
      factor *= (order + m);
      m--;
    }
    norm *= sqrt(factor);
  }

  return norm;
}

#else

// calculate the normalization factor \sqrt{(4\pi)/(2n+1)} \sqrt{(n-m)!/(n+m)!}
double BaseDecomposition::_calculateNormalization(int order, int degree)
{
  double norm	= sqrt((2 * order + 1) / (4.0 * M_PI));
  double factor = sqrt(tgamma(order - abs(degree) + 1.0) / tgamma(order + abs(degree) + 1.0));

  if ((degree < 0) && ((-degree) % 2 == 1))
    norm *= -1;

  return norm * factor;
}

#endif

double BaseDecomposition::_calculatePnm(int order, int degree, double theta)
{
  double result = gsl_sf_legendre_Plm(order, abs(degree), cos(theta));
  if (degree < 0) {
    int m = -degree;
    double factor = 1.0;
    while (m > degree) {
      factor *= (order + m);
      m--;
    }
    result /= factor;
    if (((-degree) % 2) == 1)
      result *= -1;
  }

  return result;
}

// calculate $d P^m_n(\cos\theta) / d \theta$
double BaseDecomposition::_calculate_dPnm_dtheta(int order, int degree, double theta) 
{
  // still need to test for the case |theta| = 1
  double cosTheta  = cos(theta);
  double cosTheta2 = cosTheta * cosTheta;
  double dPnm_dx   = ((degree - order - 1) * _calculatePnm(order + 1, degree, theta)
		      + (order + 1) * cosTheta * _calculatePnm(order, degree, theta)) / (1.0 - cosTheta2);

  return dPnm_dx;
}

// calculate \partial \bar{Y}^m_n(\theta, \phi) / \partial \theta
gsl_complex BaseDecomposition::harmonicDerivPolarAngle(int order, int degree, double theta, double phi)
{
  double      dPnm_dx	= _calculate_dPnm_dtheta(order, degree, theta);
  double      norm	= _calculateNormalization(order, degree);
  double      factor	= -norm * dPnm_dx * sin(theta);
  gsl_complex result	= gsl_complex_mul_real(gsl_complex_polar(1.0, -degree * phi), factor);

  return result;
}

// calculate \partial \bar{Y}^m_n(\theta, \phi) / \partial \phi
gsl_complex BaseDecomposition::harmonicDerivAzimuth(int order, int degree, double theta, double phi)
{
  if (order < degree || order < -degree)
    fprintf(stderr, "The order must be less than the degree but %d > %dn", order, degree);
  
  gsl_complex Ynm	= harmonic(order, degree, theta, phi);
  gsl_complex dYmn_dphi	= gsl_complex_mul_real(gsl_complex_mul(Ynm, gsl_complex_rect(0.0, -1.0)), degree);

  return dYmn_dphi;
}

gsl_complex BaseDecomposition::modalCoefficient(unsigned order, double ka)
{
  if(ka == 0.0)
    return gsl_complex_rect(1.0, 0.0);
  
  gsl_complex bn;
  switch (order) {
  case 0:
    {
      double      j0 = gsl_sf_sinc(ka / M_PI);
      gsl_complex h0 = gsl_complex_rect(j0, - cos(ka) / ka);
      double      val1 = ka * cos(ka) - sin(ka);
      gsl_complex val2 = gsl_complex_mul(gsl_complex_rect(ka, 1), gsl_complex_polar(1, ka));
      gsl_complex grad = gsl_complex_div(gsl_complex_rect(val1, 0), val2);
      bn = gsl_complex_sub(gsl_complex_rect(j0, 0), gsl_complex_mul(grad, h0));
      
      // bn = ( i * cos(ka) + sin(x) ) ./ ( i + ka );
      // bn = gsl_complex_div( gsl_complex_rect( sin(ka), cos(ka) ), gsl_complex_rect( ka, 1.0 ) );
    }
    break;
  case 1:
    {
      // bn = x .* ( - cos(x) + i * sin(x) ) ./ (-2 + 2 * i * x + x.^2);
      double ka2 = ka * ka;
      bn = gsl_complex_mul_real(gsl_complex_div(gsl_complex_rect(-cos(ka), sin(ka)),
						gsl_complex_rect(ka2 - 2, 2 * ka)), ka);
      //printf("E %e %e\n",ka, gsl_complex_abs(bn));
    }
    break;
  case 2:
    {
      // bn = i * x.^2 .* (cos(x) - i * sin(x)) ./ (-9*i - 9*x + 4*i*x.^2 + x.^3);
      double ka2 = ka  * ka;
      double ka3 = ka2 * ka; 
      bn = gsl_complex_mul_imag(gsl_complex_div(gsl_complex_rect(cos(ka), -sin(ka)),
						gsl_complex_rect(ka3 - 9*ka, 4*ka2 - 9)), ka2);
    }
    break;
  case 3:
    {
      // bn = x.^3 .* (cos(x) - i * sin(x)) ./ (60 - 60 * i * x - 27 * x.^2 + 7 * i * x.^3 + x.^4);
      double ka2 = ka  * ka;
      double ka3 = ka2 * ka; 
      double ka4 = ka2 * ka2;
      bn = gsl_complex_mul_real(gsl_complex_div(gsl_complex_rect(cos(ka), -sin(ka) ),
						gsl_complex_rect(ka4 - 27 * ka2 + 60, 7 * ka3 - 60 * ka)), ka3);
    }
    break;
  case 4:
    {
      //  bn = x.^4 .* (i * cos(x) + sin(x)) ./ (525*i + 525*x - 240 * i * x.^2 - 65 * x.^3 + 11 * i * x.^4 + x.^5);
      double ka2 = ka  * ka;
      double ka3 = ka2 * ka; 
      double ka4 = ka2 * ka2;
      double ka5 = ka4 * ka;
      bn = gsl_complex_mul_real(gsl_complex_div(gsl_complex_rect(sin(ka), cos(ka)),
						gsl_complex_rect(ka5 - 65 * ka3 + 525 * ka, 11 * ka4 - 240 * ka2 + 525)), ka4);
    }
    break;
  case 5:
    {
      double ka2 = ka  * ka;
      double ka3 = ka2 * ka; 
      double ka4 = ka2 * ka2;
      double ka5 = ka4 * ka;
      double ka6 = ka5 * ka;
      // bn = x.^5 .* (cos(x) - i * sin(x)) ./ (-5670 + 5670 * i * x + 2625 * x.^2 - 735 * i * x.^3 - 135 * x.^4 + 16 * i * x.^5 + x.^6);
      bn = gsl_complex_mul_real(gsl_complex_div(gsl_complex_rect(cos(ka), -sin(ka) ),
						gsl_complex_rect(ka6 - 135 * ka4 + 2625 * ka2 - 5670,
								 16 * ka5 - 735 * ka3 + 5670 * ka)), ka5);
    }
    break;
  case 6:
    {
      double ka2 = ka  * ka;
      double ka3 = ka2 * ka; 
      double ka4 = ka2 * ka2;
      double ka5 = ka4 * ka;
      double ka6 = ka5 * ka;
      double ka7 = ka6 * ka;
      //  bn = i*x.^6 .* (cos(x) - i * sin(x)) ./ (-72765 * i - 72765 * x + 34020 * i * x.^2 + 9765 * x.^3 - 1890 * i * x.^4 - 252 * x.^5 + 22 * i * x.^6 + x.^7);
      bn = gsl_complex_mul_imag( gsl_complex_div( gsl_complex_rect(cos(ka), -sin(ka)),
						  gsl_complex_rect(ka7 - 252 * ka5 + 9765 * ka3 - 72765 * ka ,
								   22 * ka6 - 1890 * ka4 + 34020 * ka2 -72765)), ka6);
    }
    break;
  case 7:
    {
      double ka2 = ka  * ka;
      double ka3 = ka2 * ka; 
      double ka4 = ka2 * ka2;
      double ka5 = ka4 * ka;
      double ka6 = ka5 * ka;
      double ka7 = ka6 * ka;
      double ka8 = ka7 * ka;
      //bn = x.^7 .* (cos(x) - i * sin(x)) ./ (1081080 - 1081080 * i * x - 509355 * x.^2 + 148995 * i * x.^3 + 29925 * x.^4 + -4284 * i * x.^5 - 434 * x.^6 + 29 * i * x.^7 + x.^8);
      bn = gsl_complex_mul_real( gsl_complex_div( gsl_complex_rect( cos(ka), -sin(ka) ),
						  gsl_complex_rect( 1081080- 509355 * ka2 + 29925 * ka4 - 434 * ka6 + ka8, - 1081080 * ka + 148995 * ka3 - 4284 * ka5 + 29 * ka7)), ka7);
    }
    break;
  case 8:
    {
      double ka2 = ka  * ka;
      double ka3 = ka2 * ka; 
      double ka4 = ka2 * ka2;
      double ka5 = ka4 * ka;
      double ka6 = ka5 * ka;
      double ka7 = ka6 * ka;
      double ka8 = ka7 * ka;
      double ka9 = ka8 * ka;
      //bn = x.^8 .* (i * cos(x) + sin(x)) ./ (18243225 * i + 18243225 * x - 8648640 * i * x.^2 - 2567565 * x.^3 + 530145 * i * x.^4 + 79695 * x.^5 - 8820 * i * x.^6 - 702 * x.^7 + 37 * i * x.^8 + x.^9);
      bn = gsl_complex_mul_real(gsl_complex_div(gsl_complex_rect(sin(ka), cos(ka)),
						gsl_complex_rect(18243225 * ka - 2567565 * ka3 + 79695 * ka5 - 702 * ka7 + ka9,
								 18243225 - 8648640 * ka2 + 530145 * ka4 - 8820 * ka6 + 37 * ka8)), ka8);
    }
    break;
  default:
    {
      int status;
      gsl_sf_result jn, jn_p, jn_n;
      gsl_sf_result yn, yn_p, yn_n;
      gsl_complex   hn, hn_p, hn_n;
      double djn, dyn;
      gsl_complex dhn;
      gsl_complex val, grad;
      
      status = gsl_sf_bessel_jl_e( order, ka, &jn);// the (regular) spherical Bessel function of the first kind
      status = gsl_sf_bessel_yl_e( order, ka, &yn);// the (irregular) spherical Bessel function of the second kind
      hn = gsl_complex_rect(jn.val, yn.val); // Spherical Hankel function of the first kind
      
      status = gsl_sf_bessel_jl_e( order-1, ka, &jn_p );
      status = gsl_sf_bessel_jl_e( order+1, ka, &jn_n );
      djn = ( jn_p.val - jn_n.val ) / 2;

      status = gsl_sf_bessel_yl_e( order-1, ka, &yn_p );
      status = gsl_sf_bessel_yl_e( order+1, ka, &yn_n );
      dyn = ( yn_p.val - yn_n.val ) / 2;

      hn_p = gsl_complex_rect( jn_p.val, yn_p.val );
      hn_n = gsl_complex_rect( jn_n.val, yn_n.val );
      
      val = gsl_complex_div_real( gsl_complex_add( hn, gsl_complex_mul_real( hn_n, ka ) ), ka );
      dhn = gsl_complex_mul_real( gsl_complex_sub( hn_p, val ), 0.5 );
      
      //printf ("status  = %s\n", gsl_strerror(status));
      //printf ("J0(5.0) = %.18f +/- % .18f\n", result.val, result.err);  
      
      grad = gsl_complex_div( gsl_complex_rect( djn, 0 ), dhn );
      bn   = gsl_complex_add_real( gsl_complex_negative( gsl_complex_mul( grad, hn ) ), jn.val );
    }
    break;
  }

  return bn;
}


// ----- methods for class `ModalDecomposition' -----
//
ModalDecomposition::ModalDecomposition(unsigned orderN, unsigned subbandsN, double a, double sampleRate, unsigned useSubbandsN)
  : BaseDecomposition(orderN, subbandsN, a, sampleRate, useSubbandsN, /* spatial= */ false) { }

void ModalDecomposition::estimateBkl(double theta, double phi, const gsl_vector_complex* snapshot, unsigned subbandX)
{
  calculate_gkl(theta, phi, subbandX);
  transform(snapshot, _vkl);
  gsl_complex eta, del;
  gsl_blas_zdotc(_gkl[subbandX], _vkl, &eta);
  gsl_blas_zdotc(_gkl[subbandX], _gkl[subbandX], &del);

  // calculate $B_{k,l}$
  double 	delta		= GSL_REAL(del);
  gsl_complex	Bkl		= gsl_complex_div_real(eta, delta);
  gsl_vector_complex_set(_bkl, subbandX, Bkl);

  // calculate $\partial B_{k,l} / \partial \theta$
  gsl_complex deta_dtheta;
  gsl_blas_zdotc(_dgkl_dtheta[subbandX], _vkl, &deta_dtheta);

  gsl_complex deta_dphi;
  gsl_blas_zdotc(_dgkl_dphi[subbandX], _vkl, &deta_dphi);

  double ddelta_dtheta = 0.0;
  for (int n = 0; n <= _orderN; n++) {
    gsl_complex bn		= gsl_vector_complex_get(_bn[subbandX], n);
    for (int m = -n; m <= n; m++) {
      double norm2		= M_PI * _calculateNormalization(n, m) * gsl_complex_abs(bn);
      double Pnm		= _calculatePnm(n, m, theta);
      double dPnm_dtheta	= _calculate_dPnm_dtheta(n, m, theta);

      double dG2nm_dtheta	= -32.0 * norm2 * norm2 * Pnm * dPnm_dtheta * sin(theta);
      ddelta_dtheta		+= dG2nm_dtheta;
    }
  }
  gsl_complex dBkl_dtheta	= gsl_complex_sub(gsl_complex_mul_real(deta_dtheta, delta), gsl_complex_mul_real(eta, ddelta_dtheta));
  dBkl_dtheta			= gsl_complex_div_real(dBkl_dtheta, delta * delta);
  gsl_vector_complex_set(_dbkl_dtheta, subbandX, dBkl_dtheta);

  gsl_complex dBkl_dphi		= gsl_complex_div_real(deta_dphi, delta);
  gsl_vector_complex_set(_dbkl_dphi, subbandX, dBkl_dphi);

  // printf("Estimated subband %d out of %d\n", subbandX, _subbandsN2);
  if (subbandX == _subbandsN2)
    _subbandList = new SubbandList(_bkl, _useSubbandsN);
}

void ModalDecomposition::transform(const gsl_vector_complex* initial, gsl_vector_complex* transformed)
{
  gsl_complex Fmn;
  unsigned idx = 0;
  for (int n = 0; n <= _orderN; n++) {				// order
    for (int m = -n ; m <= n; m++) {				// degree
      gsl_blas_zdotc(_sphericalComponent[idx], initial, &Fmn);	// Ynm^H X; see GSL documentation
      gsl_vector_complex_set(transformed, idx, Fmn);
      idx++;
    }
  }
}

const gsl_matrix_complex* ModalDecomposition::linearize(gsl_vector* xk, int frameX)
{
  double theta	= gsl_vector_get(xk, 0);
  double phi	= gsl_vector_get(xk, 1);
  unsigned rowX = 0;
  
  for (Iterator itr(_subbandList); itr.more(); itr++) {
    const SubbandEntry& subbandEntry(*itr);
    gsl_complex Bkl		= subbandEntry.bkl();
    unsigned subbandX		= subbandEntry.subbandX();
    gsl_complex dBkl_dtheta	= gsl_vector_complex_get(_dbkl_dtheta, subbandX);
    gsl_complex dBkl_dphi	= gsl_vector_complex_get(_dbkl_dphi, subbandX);

    unsigned idx = 0;
    for (int n = 0; n <= _orderN; n++) {
      gsl_complex bn		= gsl_vector_complex_get(_bn[subbandX], n);
      for (int m = -n; m <= n; m++) {
	gsl_complex dtheta	= gsl_complex_add(gsl_complex_mul(Bkl, gsl_vector_complex_get(_dgkl_dtheta[subbandX], idx)),
						  gsl_complex_mul(gsl_vector_complex_get(_gkl[subbandX], idx), dBkl_dtheta));
	gsl_matrix_complex_set(_Hbar_k, rowX, /* colX= */ 0, dtheta);

	gsl_complex dphi	= gsl_complex_add(gsl_complex_mul(Bkl, gsl_vector_complex_get(_dgkl_dphi[subbandX], idx)),
						  gsl_complex_mul(gsl_vector_complex_get(_gkl[subbandX], idx), dBkl_dphi));
	gsl_matrix_complex_set(_Hbar_k, rowX, /* colX= */ 1, dphi);
	idx++;  rowX++;
      }
    }
  }

  return _Hbar_k;
}

const gsl_vector_complex* ModalDecomposition::predictedObservation(gsl_vector* xk, int frameX)
{
  double theta	= gsl_vector_get(xk, 0);
  double phi	= gsl_vector_get(xk, 1);

  unsigned rowX = 0;
  for (Iterator itr(_subbandList); itr.more(); itr++) {
    const SubbandEntry& subbandEntry(*itr);
    gsl_complex Bkl		= subbandEntry.bkl();
    unsigned subbandX		= subbandEntry.subbandX();

    for (int n = 0; n <= _orderN; n++) {
      for (int m = -n; m <= n; m++) {
	gsl_complex Gnm = _calculate_Gnm(subbandX, n, m, theta, phi);
	gsl_complex Vkl = gsl_complex_mul(Gnm, Bkl);
	gsl_vector_complex_set(_yhat_k, rowX, Vkl);
	rowX++;
      }
    }
  }

  return _yhat_k;
}

// estimate vectors $g_{k,l}$, $\partial g_{k,l} / \partial \theta$, and, $\partial g_{k,l} / \partial \phi$
void ModalDecomposition::calculate_gkl(double theta, double phi, unsigned subbandX)
{
  unsigned idx = 0;
  for (int n = 0; n <= _orderN; n++) {
    for (int m = -n; m <= n; m++) {
      gsl_complex bn = gsl_vector_complex_get(_bn[subbandX], n);
      gsl_complex coefficient = gsl_complex_mul(bn, harmonic(n, m, theta, phi));
      gsl_vector_complex_set(_gkl[subbandX], idx, coefficient);

      gsl_complex d_dtheta = gsl_complex_mul(bn, harmonicDerivPolarAngle(n, m, theta, phi));
      gsl_vector_complex_set(_dgkl_dtheta[subbandX], idx, d_dtheta);

      gsl_complex d_dphi = gsl_complex_mul(bn, harmonicDerivAzimuth(n, m, theta, phi));
      gsl_vector_complex_set(_dgkl_dphi[subbandX], idx, d_dphi);
      idx++;
    }
  }
}


// ----- methods for class `SpatialDecomposition' -----
//
SpatialDecomposition::SpatialDecomposition(unsigned orderN, unsigned subbandsN, double a, double sampleRate, unsigned useSubbandsN)
  : BaseDecomposition(orderN, subbandsN, a, sampleRate, useSubbandsN, /* spatial= */ true) { }

void SpatialDecomposition::estimateBkl(double theta, double phi, const gsl_vector_complex* snapshot, unsigned subbandX)
{
  calculate_gkl(theta, phi, subbandX);
  gsl_vector_complex_memcpy(_vkl, snapshot);
  gsl_complex eta, del;
  gsl_blas_zdotc(_gkl[subbandX], _vkl, &eta);
  gsl_blas_zdotc(_gkl[subbandX], _gkl[subbandX], &del);

  // calculate $B_{k,l}$
  double 	delta		= GSL_REAL(del);
  gsl_complex	Bkl		= gsl_complex_div_real(eta, delta);
  gsl_vector_complex_set(_bkl, subbandX, Bkl);

  // printf("Estimated subband %d out of %d\n", subbandX, _subbandsN2);
  if (subbandX == _subbandsN2)
    _subbandList = new SubbandList(_bkl, _useSubbandsN);
}

const gsl_matrix_complex* SpatialDecomposition::linearize(gsl_vector* xk, int frameX)
{
  double theta	= gsl_vector_get(xk, 0);
  double phi	= gsl_vector_get(xk, 1);

  unsigned rowX = 0;
  for (Iterator itr(_subbandList); itr.more(); itr++) {
    const SubbandEntry& subbandEntry(*itr);
    gsl_complex Bkl		= subbandEntry.bkl();
    unsigned subbandX		= subbandEntry.subbandX();
    
    for (unsigned s = 0; s < _ChannelsN; s++) {
      gsl_complex dtheta	= gsl_complex_mul(Bkl, gsl_vector_complex_get(_dgkl_dtheta[subbandX], s));
      gsl_matrix_complex_set(_Hbar_k, rowX, /* colX= */ 0, dtheta);

      gsl_complex dphi		= gsl_complex_mul(Bkl, gsl_vector_complex_get(_dgkl_dphi[subbandX], s));
      gsl_matrix_complex_set(_Hbar_k, rowX, /* colX= */ 1, dphi);
      rowX++;
    }
  }

  return _Hbar_k;
}

const gsl_vector_complex* SpatialDecomposition::predictedObservation(gsl_vector* xk, int frameX)
{
  double theta	= gsl_vector_get(xk, 0);
  double phi	= gsl_vector_get(xk, 1);

  unsigned rowX = 0;
  for (Iterator itr(_subbandList); itr.more(); itr++) {
    const SubbandEntry& subbandEntry(*itr);
    gsl_complex Bkl		= subbandEntry.bkl();
    unsigned    subbandX	= subbandEntry.subbandX();

    for (unsigned s = 0; s < _ChannelsN; s++) {
      gsl_complex gkl		= gsl_complex_mul(gsl_vector_complex_get(_gkl[subbandX], s), Bkl);
      gsl_vector_complex_set(_yhat_k, rowX, gkl);
      rowX++;
    }
  }

  return _yhat_k;
}

void SpatialDecomposition::calculate_gkl(double theta, double phi, unsigned subbandX)
{
  for (unsigned s = 0; s < _ChannelsN; s++) {
    double theta_s		= gsl_vector_get(_theta_s, s);
    double phi_s		= gsl_vector_get(_phi_s, s);
    gsl_complex sum_n		= _ComplexZero;
    gsl_complex dsum_n_dtheta	= _ComplexZero;
    gsl_complex dsum_n_dphi	= _ComplexZero;
    for (int n = 0; n <= _orderN; n++) {
      gsl_complex b_n 		= gsl_vector_complex_get(_bn[subbandX], n);
      gsl_complex sum_m		= _ComplexZero;
      gsl_complex dsum_m_dtheta	= _ComplexZero;
      gsl_complex dsum_m_dphi	= _ComplexZero;
      for (int m = -n; m <= n; m++) {
	gsl_complex Ynm_s		= gsl_complex_conjugate(harmonic(n, m, theta_s, phi_s));
	gsl_complex coef_m		= gsl_complex_mul(Ynm_s, harmonic(n, m, theta, phi));
	sum_m				= gsl_complex_add(sum_m, coef_m);

	gsl_complex dcoef_m_dtheta	= gsl_complex_mul(Ynm_s, harmonicDerivPolarAngle(n, m, theta, phi));
	dsum_m_dtheta			= gsl_complex_add(dsum_m_dtheta, dcoef_m_dtheta);

	gsl_complex dcoef_m_dphi		= gsl_complex_mul(Ynm_s, harmonicDerivAzimuth(n, m, theta, phi));
	dsum_m_dphi			= gsl_complex_add(dsum_m_dphi, dcoef_m_dphi);
      }
      sum_n				= gsl_complex_add(sum_n,        gsl_complex_mul(b_n, sum_m));
      dsum_n_dtheta			= gsl_complex_add(dsum_n_dtheta, gsl_complex_mul(b_n, dsum_m_dtheta));
      dsum_n_dphi			= gsl_complex_add(dsum_n_dphi,   gsl_complex_mul(b_n, dsum_m_dphi));
    }
    gsl_vector_complex_set(_gkl[subbandX],         s, sum_n);
    gsl_vector_complex_set(_dgkl_dtheta[subbandX], s, dsum_n_dtheta);
    gsl_vector_complex_set(_dgkl_dphi[subbandX],   s, dsum_n_dphi);
  }
}


// ----- methods for class `BaseSphericalArrayTracker' -----
//
const unsigned    BaseSphericalArrayTracker::_StateN		= 2;
const gsl_complex BaseSphericalArrayTracker::_ComplexZero	= gsl_complex_rect(0.0, 0.0);
const gsl_complex BaseSphericalArrayTracker::_ComplexOne	= gsl_complex_rect(1.0, 0.0);
const double      BaseSphericalArrayTracker::_Epsilon		= 0.01;
const double	  BaseSphericalArrayTracker::_Tolerance		= 1.0e-04;

BaseSphericalArrayTracker::
BaseSphericalArrayTracker(BaseDecompositionPtr& baseDecomposition, double sigma2_u, double sigma2_v, double sigma2_init,
			  unsigned maxLocalN, const String& nm)
  : VectorFloatFeatureStream(_StateN, nm), _firstFrame(true),
    _subbandsN(baseDecomposition->subbandsN()), _subbandsN2(baseDecomposition->subbandsN2()),
    _useSubbandsN(baseDecomposition->useSubbandsN()),
    _modesN(baseDecomposition->modesN()), _subbandLengthN(baseDecomposition->subbandLengthN()),
    _observationN(baseDecomposition->useSubbandsN() * baseDecomposition->subbandLengthN()),
    _maxLocalN(maxLocalN), _endOfSamples(false),
    _sigma_init(sqrt(sigma2_init)),
    _baseDecomposition(baseDecomposition),
    _U(gsl_matrix_calloc(_StateN, _StateN)),
    _V(gsl_matrix_calloc(2 * (_subbandsN2 + 1) * _subbandLengthN, 2 * (_subbandsN2 + 1) * _subbandLengthN)),
    _K_k_k1(gsl_matrix_calloc(_StateN, _StateN)),
    _prearray(gsl_matrix_calloc((2 * _observationN + _StateN), (2 * _observationN + 2 * _StateN))),
    _vk(gsl_vector_complex_calloc(_observationN)),
    _Hbar_k(gsl_matrix_calloc(2 * _observationN, _StateN)), _yhat_k(gsl_vector_calloc(2 * _observationN)),
    _correction(gsl_vector_calloc(_StateN)),
    _position(gsl_vector_calloc(_StateN)), _eta_i(gsl_vector_calloc(_StateN)), _delta(gsl_vector_calloc(_StateN)),
    _residual(gsl_vector_complex_calloc(_observationN)), _residual_real(gsl_vector_calloc(2 * _observationN)),
    _scratch(gsl_vector_calloc(2 * _observationN))
{
  nextSpeaker();

  // initialize covariance matrices
  for (unsigned n = 0; n < _StateN; n++) {
    gsl_matrix_set(_U, n, n, sqrt(sigma2_u));
    gsl_matrix_set(_K_k_k1, n, n, sqrt(_sigma_init));
  }

  for (unsigned n = 0; n < (2 * (_subbandsN2 + 1) * _subbandLengthN); n++)
    gsl_matrix_set(_V, n, n, sqrt(sigma2_v));
}

void BaseSphericalArrayTracker::nextSpeaker()
{
  reset();

  _firstFrame = true;

  // set initial position: theta = 0.5, phi = 0
  gsl_vector_set(_position, 0, 0.5);
  gsl_vector_set(_position, 1, 0.0);

  // re-initialize state estimation error covariance matrix
  gsl_matrix_set_zero(_K_k_k1);
  for (unsigned n = 0; n < _StateN; n++)
    gsl_matrix_set(_K_k_k1, n, n, sqrt(_sigma_init));
}

void BaseSphericalArrayTracker::setInitialPosition(double theta, double phi)
{
  gsl_vector_set(_position, 0, theta);
  gsl_vector_set(_position, 1, phi);
}

BaseSphericalArrayTracker::~BaseSphericalArrayTracker()
{
  gsl_matrix_free(_U);
  gsl_matrix_free(_V);
  gsl_matrix_free(_K_k_k1);
  gsl_matrix_free(_prearray);
  gsl_vector_complex_free(_vk);
  gsl_matrix_free(_Hbar_k);
  gsl_vector_free(_yhat_k);
  gsl_vector_free(_correction);
  gsl_vector_free(_position);
  gsl_vector_free(_eta_i);
  gsl_vector_free(_delta);
  gsl_vector_complex_free(_residual);
  gsl_vector_free(_residual_real);
  gsl_vector_free(_scratch);
}

void BaseSphericalArrayTracker::setChannel(VectorComplexFeatureStreamPtr& chan)
{
  _channelList.push_back(chan);
}

// "realify" 'Vk' and place into 'V'
void BaseSphericalArrayTracker::setV(const gsl_matrix_complex* Vk, unsigned subbandX)
{
  gsl_matrix_view  _Vk = gsl_matrix_submatrix(_V, /* m1= */ 2 * subbandX * _subbandLengthN, /* n1= */ 2 * subbandX * _subbandLengthN,
					      2 * _subbandLengthN, 2 * _subbandLengthN);
  for (unsigned m = 0; m < _subbandLengthN; m++) {
    for (unsigned n = 0; n <= m; n++) {
      double c = GSL_REAL(gsl_matrix_complex_get(Vk, m, n));
      gsl_matrix_set(&_Vk.matrix, m, n, c);
      gsl_matrix_set(&_Vk.matrix, m + _subbandLengthN, n + _subbandLengthN, c);

      double s = GSL_IMAG(gsl_matrix_complex_get(Vk, m, n));
      gsl_matrix_set(&_Vk.matrix, m + _subbandLengthN, n,  s);
    }
  }
  gsl_linalg_cholesky_decomp(&_Vk.matrix);

  // zero out upper triangular portion of Cholesky decomposition
  for (unsigned m = 0; m < 2 * _subbandLengthN; m++)
    for (unsigned n = m + 1; n < 2 * _subbandLengthN; n++)
      gsl_matrix_set(&_Vk.matrix, m, n, 0.0);
}

// calculate squared-error with current state estimate
double BaseSphericalArrayTracker::_calculateResidual()
{
  const gsl_vector_complex* yhat_k = _baseDecomposition->predictedObservation(_eta_i, _frameX + 1);
  gsl_vector_complex_memcpy(_residual, _vk);
  
  gsl_vector_complex_sub(_residual, yhat_k);
  double residual = 0.0;
  for (unsigned observationX = 0; observationX < _observationN; observationX++)
    residual += gsl_complex_abs2(gsl_vector_complex_get(_residual, observationX));
  residual /= _observationN;

  return residual;
}
    
void BaseSphericalArrayTracker::_copyPosition()
{
  for (unsigned positionX = 0; positionX < _StateN; positionX++)
    gsl_vector_float_set(_vector, positionX, gsl_vector_get(_position, positionX));
}

void BaseSphericalArrayTracker::_printMatrix(const gsl_matrix_complex* mat)
{
  for (unsigned m = 0; m < mat->size1; m++) {
    for (unsigned n = 0; n < mat->size2; n++) {
      gsl_complex value = gsl_matrix_complex_get(mat, m, n);
      printf("%8.4f %8.4f  ", GSL_REAL(value), GSL_IMAG(value));
    }
    printf("\n");
  }
}

void BaseSphericalArrayTracker::_printMatrix(const gsl_matrix* mat)
{
  for (unsigned m = 0; m < mat->size1; m++) {
    for (unsigned n = 0; n < mat->size2; n++) {
      double value = gsl_matrix_get(mat, m, n);
      printf("%8.4f ", value);
    }
    printf("\n");
  }
}

void BaseSphericalArrayTracker::_printVector(const gsl_vector_complex* vec)
{
  for (unsigned n = 0; n < vec->size; n++) {
    gsl_complex value = gsl_vector_complex_get(vec, n);
    printf("%8.4f %8.4f\n", GSL_REAL(value), GSL_IMAG(value));
  }
}

void BaseSphericalArrayTracker::_printVector(const gsl_vector* vec)
{
  for (unsigned n = 0; n < vec->size; n++) {
    double value = gsl_vector_get(vec, n);
    printf("%8.4f\n", value);
  }
}

// calculate the cosine 'c' and sine 's' parameters of Givens
// rotation that rotates 'v2' into 'v1'
double BaseSphericalArrayTracker::_calcGivensRotation(double v1, double v2, double& c, double& s)
{
  double norm = sqrt(v1 * v1 + v2 * v2);

  if (norm == 0.0)
    throw jarithmetic_error("calcGivensRotation: Norm is zero.");

  c = v1 / norm;
  // gsl_complex_div_real(v1, norm);

  s = v2 / norm;
  // gsl_complex_div_real(gsl_complex_conjugate(v2), norm);

  // return gsl_complex_rect(norm, 0.0);
  return norm;
}

// apply a previously calculated Givens rotation
void BaseSphericalArrayTracker::_applyGivensRotation(double v1, double v2, double c, double s, double& v1p, double& v2p)
{
  v1p = c * v1 + s * v2;
  /*
    gsl_complex_add(gsl_complex_mul(gsl_complex_conjugate(c), v1),
		    gsl_complex_mul(s, v2));
  */

  v2p = c * v2 - s * v1;
  /*
    gsl_complex_sub(gsl_complex_mul(c, v2),
		    gsl_complex_mul(gsl_complex_conjugate(s), v1));
  */
}

void BaseSphericalArrayTracker::_realify(const gsl_matrix_complex* Hbar_k, const gsl_vector_complex* yhat_k)
{
  for (unsigned subX = 0; subX < _useSubbandsN; subX++) {
    for (unsigned obsX = 0; obsX < _subbandLengthN; obsX++) {
      for (unsigned stateX = 0; stateX < BaseDecomposition::_StateN; stateX++) {
	gsl_matrix_set(_Hbar_k,
		        2 * subX      * _subbandLengthN + obsX, stateX, GSL_REAL(gsl_matrix_complex_get(Hbar_k, subX * _subbandLengthN + obsX, stateX)));
	gsl_matrix_set(_Hbar_k,
		       (2 * subX + 1) * _subbandLengthN + obsX, stateX, GSL_IMAG(gsl_matrix_complex_get(Hbar_k, subX * _subbandLengthN + obsX, stateX)));
      }
      gsl_vector_set(_yhat_k,  2 * subX      * _subbandLengthN + obsX, GSL_REAL(gsl_vector_complex_get(yhat_k, subX * _subbandLengthN + obsX)));
      gsl_vector_set(_yhat_k, (2 * subX + 1) * _subbandLengthN + obsX, GSL_IMAG(gsl_vector_complex_get(yhat_k, subX * _subbandLengthN + obsX)));
    }
  }
}

void BaseSphericalArrayTracker::_realifyResidual()
{
  for (unsigned subX = 0; subX < _useSubbandsN; subX++) {
    for (unsigned obsX = 0; obsX < _subbandLengthN; obsX++) {
      gsl_vector_set(_residual_real,  2 * subX      * _subbandLengthN + obsX, GSL_REAL(gsl_vector_complex_get(_residual, subX * _subbandLengthN + obsX)));
      gsl_vector_set(_residual_real, (2 * subX + 1) * _subbandLengthN + obsX, GSL_IMAG(gsl_vector_complex_get(_residual, subX * _subbandLengthN + obsX)));
    }
  }
}

void BaseSphericalArrayTracker::_update(const gsl_matrix_complex* Hbar_k, const gsl_vector_complex* yhat_k, const SubbandListPtr& subbandList)
{
  // copy matrix components into prearray
  gsl_matrix_set_zero(_prearray);
  
  // copy subband-dependent observation noise covariance matrices into prearray
  unsigned subX = 0;
  for (Iterator itr(subbandList); itr.more(); itr++) {
    const    SubbandEntry& subbandEntry(*itr);
    unsigned subbandX	= subbandEntry.subbandX();

    // cout << "subX = " << subX << " subbandX = " << subbandX << endl;

    gsl_matrix_view  Vk = gsl_matrix_submatrix(_prearray, /* m1= */ 2 * subX * _subbandLengthN, /* n1= */ 2 * subX * _subbandLengthN,
					       2 * _subbandLengthN, 2 * _subbandLengthN);
    gsl_matrix_view _Vk = gsl_matrix_submatrix(_V, /* m1= */ 2 * subbandX * _subbandLengthN, /* n1= */ 2 * subbandX * _subbandLengthN,
					       2 * _subbandLengthN, 2 * _subbandLengthN);
    gsl_matrix_memcpy(&Vk.matrix, &_Vk.matrix);
    subX++;
  }
  _realify(Hbar_k, yhat_k);
  gsl_matrix_view HkKkk1 = gsl_matrix_submatrix(_prearray, /* m1= */ 0, 2 * _observationN, 2 * _observationN, _StateN);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, _Hbar_k, _K_k_k1, 0.0, &HkKkk1.matrix);

  gsl_matrix_view K_k_k1 = gsl_matrix_submatrix(_prearray, 2 * _observationN, 2 * _observationN, _StateN, _StateN);
  gsl_matrix_memcpy(&K_k_k1.matrix, _K_k_k1);

  gsl_matrix_view Uk = gsl_matrix_submatrix(_prearray, 2 * _observationN, 2 * _observationN + _StateN, _StateN, _StateN);
  gsl_matrix_memcpy(&Uk.matrix, _U);

  // calculate postarray by imposing lower triangular form on prearray
  _lowerTriangularize();

  // conventional innovation vector
  gsl_vector_complex_memcpy(_residual, _vk);  _realifyResidual();
  gsl_vector_sub(_residual_real, _yhat_k);

  // extra term required by IEKF
  gsl_vector_memcpy(_delta, _position);
  gsl_vector_sub(_delta, _eta_i);
  gsl_blas_dgemv(CblasNoTrans, 1.0, _Hbar_k, _delta, 0.0, _scratch);
  gsl_vector_sub(_residual_real, _scratch);

  // perform (local) state update
  gsl_matrix_view  Vk = gsl_matrix_submatrix(_prearray, /* m1= */ 0, /* n1= */ 0, 2 * _observationN, 2 * _observationN);
  gsl_blas_dtrsv(CblasLower, CblasNoTrans, CblasNonUnit, &Vk.matrix, _residual_real);
  gsl_matrix_view B21 = gsl_matrix_submatrix(_prearray, 2 * _observationN, 0, _StateN, 2 * _observationN);
  gsl_blas_dgemv(CblasNoTrans, 1.0, &B21.matrix, _residual_real, 0.0, _correction);

  cout << endl << "Correction:" << endl;
  _printVector(_correction);
  gsl_vector_add(_eta_i, _correction);
  _checkPhysicalConstraints();
}

// maintain the spherical coordinates within physical bounds
void BaseSphericalArrayTracker::_checkPhysicalConstraints()
{
  double theta = gsl_vector_get(_eta_i, 0);
  double phi   = gsl_vector_get(_eta_i, 1);

  // constrain polar angle to 0 < theta < pi
  if (theta < _Epsilon) {
    gsl_vector_set(_eta_i, 0, _Epsilon);
    printf("Limiting polar angle %6.2f to %6.2f\n", theta, _Epsilon);
  } else if (theta > M_PI - _Epsilon) {
    gsl_vector_set(_eta_i, 0, M_PI - _Epsilon);
    printf("Limiting polar angle %6.2f to %6.2f\n", theta, M_PI - _Epsilon);
  } else
    gsl_vector_set(_eta_i, 0, theta);

  // not necessary to constrain the azimuth
  gsl_vector_set(_eta_i, 1, phi);
}

// calculate postarray by imposing lower triangular form on prearray
void BaseSphericalArrayTracker::_lowerTriangularize()
{
  // zero out upper portion of A12 row by row
  gsl_matrix_view A11 = gsl_matrix_submatrix(_prearray, 0, 0, 2 * _observationN + _StateN, 2 * _observationN);
  gsl_matrix_view A12 = gsl_matrix_submatrix(_prearray, 0, 2 * _observationN, 2 * _observationN + _StateN, _StateN);

  for (unsigned rowX = 0; rowX < 2 * _observationN; rowX++) {
    for (unsigned colX = 0; colX < _StateN; colX++) {
      // element to be annihilated
      double c, s;
      double v1 = gsl_matrix_get(&A11.matrix, rowX, rowX);
      double v2 = gsl_matrix_get(&A12.matrix, rowX, colX);
      gsl_matrix_set(&A11.matrix, rowX, rowX, _calcGivensRotation(v1, v2, c, s));
      gsl_matrix_set(&A12.matrix, rowX, colX, 0.0);

      // complete rotation on remainder of column
      for (unsigned n = rowX + 1; n < 2 * _observationN + _StateN; n++) {
	double v1p, v2p;
	v1 = gsl_matrix_get(&A11.matrix, n, rowX);
	v2 = gsl_matrix_get(&A12.matrix, n, colX);
	_applyGivensRotation(v1, v2, c, s, v1p, v2p);
	gsl_matrix_set(&A11.matrix, n, rowX, v1p);
	gsl_matrix_set(&A12.matrix, n, colX, v2p);
      }
    }
  }

  // lower triangularize A22 row by row
  gsl_matrix_view A22 = gsl_matrix_submatrix(_prearray, 2 * _observationN, 2 * _observationN, _StateN, _StateN);
  for (unsigned rowX = 0; rowX < _StateN; rowX++) {
    for (unsigned colX = rowX + 1; colX < _StateN; colX++) {
      // element to be annihilated
      double c, s;
      double v1 = gsl_matrix_get(&A22.matrix, rowX, rowX);
      double v2 = gsl_matrix_get(&A22.matrix, rowX, colX);
      gsl_matrix_set(&A22.matrix, rowX, rowX, _calcGivensRotation(v1, v2, c, s));
      gsl_matrix_set(&A22.matrix, rowX, colX, 0.0);

      // complete rotation on remainder of column
      for (unsigned n = rowX + 1; n < _StateN; n++) {
	double v1p, v2p;
	v1 = gsl_matrix_get(&A22.matrix, n, rowX);
	v2 = gsl_matrix_get(&A22.matrix, n, colX);
	_applyGivensRotation(v1, v2, c, s, v1p, v2p);
	gsl_matrix_set(&A22.matrix, n, rowX, v1p);
	gsl_matrix_set(&A22.matrix, n, colX, v2p);
      }
    }
  }

  // zero out all of A23 by rotating it into A22 row by row
  gsl_matrix_view A23 = gsl_matrix_submatrix(_prearray, 2 * _observationN, 2 * _observationN + _StateN, _StateN, _StateN);
  for (unsigned rowX = 0; rowX < _StateN; rowX++) {
    for (unsigned colX = 0; colX < _StateN; colX++) {
      // element to be annihilated
      double c, s;
      double v1 = gsl_matrix_get(&A22.matrix, rowX, rowX);
      double v2 = gsl_matrix_get(&A23.matrix, rowX, colX);
      gsl_matrix_set(&A22.matrix, rowX, rowX, _calcGivensRotation(v1, v2, c, s));
      gsl_matrix_set(&A23.matrix, rowX, colX, 0.0);

      // complete rotation on remainder of column
      for (unsigned n = rowX + 1; n < _StateN; n++) {
	double v1p, v2p;
	v1 = gsl_matrix_get(&A22.matrix, n, rowX);
	v2 = gsl_matrix_get(&A23.matrix, n, colX);
	_applyGivensRotation(v1, v2, c, s, v1p, v2p);
	gsl_matrix_set(&A22.matrix, n, rowX, v1p);
	gsl_matrix_set(&A23.matrix, n, colX, v2p);
      }
    }
  }
}

void BaseSphericalArrayTracker::reset()
{
  _baseDecomposition->reset();

  for (_ChannelIterator itr = _channelList.begin(); itr != _channelList.end(); itr++)
    (*itr)->reset();

  if (_snapShotArray != NULL)
    _snapShotArray->zero();

  VectorFloatFeatureStream::reset();
  _endOfSamples = false;
}

void BaseSphericalArrayTracker::_allocImage()
{
  if(_snapShotArray == NULL)
    _snapShotArray = new SnapShotArray(_subbandsN, chanN());
}

// ----- methods for class `ModalSphericalArrayTracker' -----
//
ModalSphericalArrayTracker::
ModalSphericalArrayTracker(ModalDecompositionPtr& modalDecomposition, double sigma2_u, double sigma2_v, double sigma2_init,
			   unsigned maxLocalN, const String& nm)
  : BaseSphericalArrayTracker(modalDecomposition, sigma2_u, sigma2_v, sigma2_init, maxLocalN, nm) { }

const gsl_vector_float* ModalSphericalArrayTracker::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  // get new snapshots
  this->_allocImage();
  unsigned chanX = 0;
  for (_ChannelIterator itr = _channelList.begin(); itr != _channelList.end(); itr++) {
    const gsl_vector_complex* samp = (*itr)->next(_frameX + 1);
    if((*itr)->isEnd() == true) _endOfSamples = true;
    _snapShotArray->newSample(samp, chanX);  chanX++;
  }
  _snapShotArray->update();

  gsl_vector_memcpy(_eta_i, _position);
  for (unsigned localX = 0; localX < _maxLocalN; localX++) {
    double theta = gsl_vector_get(_eta_i, 0);
    double phi   = gsl_vector_get(_eta_i, 1);

    // estimate and sort Bkl factors
    for (unsigned subbandX = 0; subbandX <= _subbandsN2; subbandX++) {
      const gsl_vector_complex* snapshot = _snapShotArray->getSnapShot(subbandX);
      _baseDecomposition->estimateBkl(theta, phi, snapshot, subbandX);
    }

    unsigned subX = 0;
    for (BaseDecomposition::Iterator itr(_baseDecomposition->subbandList()); itr.more(); itr++) {
      const BaseDecomposition::SubbandEntry& subbandEntry(*itr);
      gsl_complex Bkl				= subbandEntry.bkl();
      unsigned    subbandX			= subbandEntry.subbandX();

      const gsl_vector_complex* snapshot	= _snapShotArray->getSnapShot(subbandX);
      gsl_vector_complex_view   vk		= gsl_vector_complex_subvector(_vk, subX * _modesN, _modesN);
      _baseDecomposition->transform(snapshot, &vk.vector);
      subX++;
    }

    // perform position estimate update
    const gsl_matrix_complex* Hbar_k = _baseDecomposition->linearize(_eta_i, _frameX + 1);
    const gsl_vector_complex* yhat_k = _baseDecomposition->predictedObservation(_eta_i, _frameX + 1);

    double residualBefore = _calculateResidual();

    _update(Hbar_k, yhat_k, _baseDecomposition->subbandList());

    double residualAfter = _calculateResidual();

    printf("Before local update %2d : Residual = %10.4e\n", localX, residualBefore);
    printf("After  local update %2d : Residual = %10.4e\n", localX, residualAfter);

    // test for convergence
    if ((residualBefore - residualAfter) / (residualBefore + residualAfter) < _Tolerance) break;
  }
  gsl_vector_memcpy(_position, _eta_i);
  _copyPosition();

  // copy out updated state estimation error covariance matrix
  gsl_matrix_view K_k_k1 = gsl_matrix_submatrix(_prearray, 2 * _observationN, 2 * _observationN, _StateN, _StateN);
  gsl_matrix_memcpy(_K_k_k1, &K_k_k1.matrix);

  cout << "K_k_k1" << endl;
  _printMatrix(_K_k_k1);

  _increment();
  return _vector;
}


// ----- methods for class `SpatialSphericalArrayTracker' -----
//
SpatialSphericalArrayTracker::
SpatialSphericalArrayTracker(SpatialDecompositionPtr& spatialDecomposition, double sigma2_u, double sigma2_v, double sigma2_init,
			     unsigned maxLocalN, const String& nm)
  : BaseSphericalArrayTracker(spatialDecomposition, sigma2_u, sigma2_v, sigma2_init, maxLocalN, nm) { }


const gsl_vector_float* SpatialSphericalArrayTracker::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  // get new snapshots
  this->_allocImage();
  unsigned chanX = 0;
  for (_ChannelIterator itr = _channelList.begin(); itr != _channelList.end(); itr++) {
    const gsl_vector_complex* samp = (*itr)->next(_frameX + 1);
    if((*itr)->isEnd() == true) _endOfSamples = true;
    _snapShotArray->newSample(samp, chanX);  chanX++;
  }
  _snapShotArray->update();

  gsl_vector_memcpy(_eta_i, _position);
  for (unsigned localX = 0; localX < _maxLocalN; localX++) {
    double theta = gsl_vector_get(_eta_i, 0);
    double phi   = gsl_vector_get(_eta_i, 1);

    // estimate and sort Bkl factors
    for (unsigned subbandX = 0; subbandX <= _subbandsN2; subbandX++) {
      const gsl_vector_complex* snapshot = _snapShotArray->getSnapShot(subbandX);
      _baseDecomposition->estimateBkl(theta, phi, snapshot, subbandX);
    }

    unsigned subX = 0;
    for (BaseDecomposition::Iterator itr(_baseDecomposition->subbandList()); itr.more(); itr++) {
      const BaseDecomposition::SubbandEntry& subbandEntry(*itr);
      gsl_complex Bkl				= subbandEntry.bkl();
      unsigned	  subbandX			= subbandEntry.subbandX();

      const gsl_vector_complex* snapshot 	= _snapShotArray->getSnapShot(subbandX);
      gsl_vector_complex_view   vk		= gsl_vector_complex_subvector(_vk, subX * BaseDecomposition::_ChannelsN, BaseDecomposition::_ChannelsN);
      gsl_vector_complex_memcpy(&vk.vector, snapshot);
      subX++;
    }

    double residualBefore = _calculateResidual();
    printf("Before local update %2d : Residual = %12.6e\n", localX, residualBefore);

    // perform position estimate update
    const gsl_matrix_complex* Hbar_k = _baseDecomposition->linearize(_eta_i, _frameX + 1);
    const gsl_vector_complex* yhat_k = _baseDecomposition->predictedObservation(_eta_i, _frameX + 1);

    _update(Hbar_k, yhat_k, _baseDecomposition->subbandList());

    theta = gsl_vector_get(_eta_i, 0);  phi = gsl_vector_get(_eta_i, 1);
    for (BaseDecomposition::Iterator itr(_baseDecomposition->subbandList()); itr.more(); itr++) {
      const BaseDecomposition::SubbandEntry& subbandEntry(*itr);
      unsigned   subbandX                      = subbandEntry.subbandX();
 
      _baseDecomposition->calculate_gkl(theta, phi, subbandX);
    }

    double residualAfter = _calculateResidual();
    printf("After  local update %2d : Residual = %12.6e\n", localX, residualAfter);

    // test for convergence
    if ((residualBefore - residualAfter) / (residualBefore + residualAfter) < _Tolerance) break;
  }
  gsl_vector_memcpy(_position, _eta_i);
  _copyPosition();

  // re-estimate and re-sort Bkl factors
  double theta = gsl_vector_get(_eta_i, 0);
  double phi   = gsl_vector_get(_eta_i, 1);
  for (unsigned subbandX = 0; subbandX <= _subbandsN2; subbandX++) {
    const gsl_vector_complex* snapshot = _snapShotArray->getSnapShot(subbandX);
    _baseDecomposition->estimateBkl(theta, phi, snapshot, subbandX);
  }

  // copy out updated state estimation error covariance matrix
  gsl_matrix_view K_k_k1 = gsl_matrix_submatrix(_prearray, 2 * _observationN, 2 * _observationN, _StateN, _StateN);
  gsl_matrix_memcpy(_K_k_k1, &K_k_k1.matrix);

  cout << "K_k_k1" << endl;
  _printMatrix(_K_k_k1);

  _increment();
  return _vector;
}



// ----- methods for class `PlaneWaveSimulator' -----
//
const gsl_complex PlaneWaveSimulator::_ComplexZero = gsl_complex_rect(0.0, 0.0);

PlaneWaveSimulator::PlaneWaveSimulator(const VectorComplexFeatureStreamPtr& source, ModalDecompositionPtr& modalDecomposition,
				       unsigned channelX, double theta, double phi, const String& nm)
  : VectorComplexFeatureStream(modalDecomposition->subbandsN(), nm),
    _subbandsN(modalDecomposition->subbandsN()), _subbandsN2(modalDecomposition->subbandsN2()), _channelX(channelX),
    _theta(theta), _phi(phi),
    _source(source), _modalDecomposition(modalDecomposition),
    _subbandCoefficients(gsl_vector_complex_calloc(_subbandsN2 + 1))
{
  // cout << "Initializing 'PlaneWaveSimulator' ... " << endl;
  for (unsigned subbandX = 0; subbandX <= _subbandsN2; subbandX++) {
    gsl_complex coefficient = _ComplexZero;
    for (int n = 0; n <= _modalDecomposition->orderN(); n++) {			// order
      gsl_complex bn = _modalDecomposition->modalCoefficient(n, subbandX);	// scaled modal coefficient
      gsl_complex coeff_n = _ComplexZero;
      for (int m = -n ; m <= n; m++) {						// degree
	coeff_n = gsl_complex_add(coeff_n, gsl_complex_mul(_modalDecomposition->harmonic(n, m, _channelX),
							   _modalDecomposition->harmonic(n, m, _theta, _phi)));
      }
      coefficient = gsl_complex_add(coefficient, gsl_complex_mul(bn, coeff_n));
    }
    gsl_vector_complex_set(_subbandCoefficients, subbandX, coefficient);
  }
  // cout << "Done." << endl;
}

PlaneWaveSimulator::~PlaneWaveSimulator()
{
  gsl_vector_complex_free(_subbandCoefficients);
}

const gsl_vector_complex* PlaneWaveSimulator::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  const gsl_vector_complex* block = _source->next(_frameX + 1);
  for (unsigned subbandX = 0; subbandX <= _subbandsN2; subbandX++) {
    gsl_complex component = gsl_complex_mul(gsl_vector_complex_get(_subbandCoefficients, subbandX), gsl_vector_complex_get(block, subbandX));
    gsl_vector_complex_set(_vector, subbandX, component);
    if (subbandX != 0 && subbandX != _subbandsN2)
      gsl_vector_complex_set(_vector, _subbandsN - subbandX, gsl_complex_conjugate(component));
  }

  _increment();
  return _vector;
}

void PlaneWaveSimulator::reset()
{
  _source->reset();  VectorComplexFeatureStream::reset();
}
