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

#include <gsl/gsl_linalg.h>

#include <math.h>
#include <iostream>
#include <algorithm>

#include "common/mach_ind_io.h"
#include "train/estimateAdapt.h"
#include <gsl/gsl_linalg.h>


// ----- global trace variables -----
//
static       int Trace           = 0x0000;
static const int Top             = 0x0001;
static const int Sequences       = 0x0002;
static const int Bound           = 0x0004;
static const int Likelihood      = 0x0008;
static const int Iterates        = 0x0010;
static const int Determinant     = 0x0020;
static const int SearchDirection = 0x0040;
static const int TraceGradient   = 0x0080;
static const int SequenceDerivs  = 0x0100;

BiasType getBiasType(const char* ctype)
{
  if (ctype == NULL) return NoBias;

  String type(ctype);
  if (type == "NONE" || type == "None")
    return NoBias;
  if (type == "CEPSTRAONLY" || type == "CepstraOnly")
    return CepstralFeat;
  if (type == "ALLCOMPONENTS" || type == "AllComponents")
    return AllFeat;

  throw jtype_error("Unrecognized bias type %s", type.chars());
  return AllFeat;
}


// ----- methods for class `GaussListTrain' -----
//
GaussListTrain::GaussListTrain(CodebookSetTrainPtr& cb, unsigned idx)
{
  for (CodebookSetTrain::GaussianIterator itr(cb); itr.more(); itr++) {
    CodebookTrain::GaussDensity mix(itr.mix());

    if (idx == 0 || isAncestor(idx, mix.regClass()))
      _glist.push_back(mix);
  }
}


// ----- methods for interface class `EstimatorBase' -----
//
const unsigned EstimatorBase::MaxItns      = 100;
const double   EstimatorBase::Tolerance    = 1.0e-02;
const double   EstimatorBase::ConstGold    = 0.3819660;
const double   EstimatorBase::Gold         = 1.618034;
const double   EstimatorBase::GoldLimit    = 10.0;
const double   EstimatorBase::Tiny         = 1.0e-20;
const double   EstimatorBase::ZEpsilon     = 1.0e-10;

const double   EstimatorBase::BarrierScale = 1.0e-06;
const double   EstimatorBase::RhoEpsilon   = 1.0e-03;
const double   EstimatorBase::MaximumRho   = 0.90;

// enforce upper and lower bounds on step size
void EstimatorBase::bound(double min, double max, double& q)
{
  if (q > max) {
    if(Trace & Bound)
      printf("Bounding %g to %g", q, max);
    q = max;
  }
  if (q < min) {
    if (Trace & Bound)
      printf("Bounding %g to %g", q, min);
    q = min;
  }
}

// bracket the extremum of the likelihood function
bool EstimatorBase::bracket(double& ax, double& bx, double& cx)
{
  double fa, fb, fc;
  double ulim, u, r, q, fu;

  if (Trace & Iterates)
     cout << "    bracketing ..." << endl;

  double minim, maxim;
  findRange(MaximumRho-RhoEpsilon, minim, maxim);

  bound(minim, maxim, ax);

  fa = -calcLogLhood(ax);  if (isnan(fa)) return false;
  fb = -calcLogLhood(bx);  if (isnan(fb)) return false;
  if (fb > fa) {
    _swap(ax, bx);
    _swap(fb, fa);
  }
  cx = bx + Gold * (bx - ax);
  bound(minim, maxim, cx);
  fc = -calcLogLhood(cx);  if (isnan(fc)) return false;

  while (fb > fc) {
    r = (bx-ax) * (fb-fc);
    q = (bx-cx) * (fb-fa);
    u = bx - ((bx - cx) * q - (bx - ax)*r) /
      (2.0*_sign(max(fabs(q-r),Tiny),q-r));
    bound(minim, maxim, u);

    ulim = bx + GoldLimit * (cx - bx);
    bound(minim, maxim, ulim);

    if ((bx-u)*(u-cx) > 0.0) {
      fu = -calcLogLhood(u);
      if (fu < fc) {
	ax = bx;
	bx = u;
	fa = fb;
	fb = fu;
	return true;
      } else if (fu > fb) {
	cx = u;
	fc = fu;
	return true;
      }
      u=cx + Gold * (cx-bx);
      bound(minim, maxim, u);
      fu = -calcLogLhood(u);

    } else if ((cx-u)*(u-ulim) > 0.0) {
      fu = -calcLogLhood(u);

      if (fu < fc) {
	_shift(bx,cx,u,cx+Gold*(cx-bx));
	bound(minim, maxim, u);
	_shift(fb,fc,fu,-calcLogLhood(u));
      }
    } else if ((u-ulim)*(ulim-cx) >= 0.0) {
      u  = ulim;
      fu = -calcLogLhood(u);
    } else {
      u=cx + Gold * (cx - bx);
      bound(minim, maxim, u);
      fu = -calcLogLhood(u);
    }
    _shift(ax,bx,cx,u);
    _shift(fa,fb,fc,fu);
  }

  return true;
}

// perform a line search for the extremum of a function
double EstimatorBase::
brentsMethod(double ax, double bx, double cx, double& xmin)
{
  double a,b,d = 0.0,etemp,fu,fv,fw,fx,p,q,r,tol1,tol2,u,v,w,x,xm;
  double e=0.0;

  if (Trace & Iterates)
     cout << "    solving ..." << endl;

  a = min(ax, cx);
  b = max(ax, cx);
  x = w = v = bx;

  fw = fv = fx = -calcLogLhood(x);
  for (unsigned iter = 0; iter < MaxItns; iter++) {
    xm=0.5*(a+b);
    tol2=2.0*(tol1= Tolerance *fabs(x)+ZEpsilon);
    if (fabs(x-xm) <= (tol2-0.5*(b-a))) { xmin=x; return(-fx); }
    if (fabs(e) > tol1) {
      r=(x-w)*(fx-fv);
      q=(x-v)*(fx-fw);
      p=(x-v)*q-(x-w)*r;
      q=2.0*(q-r);
      if (q > 0.0) p = -p;
      q=fabs(q);
      etemp=e;
      e=d;
      if (fabs(p) >= fabs(0.5*q*etemp) || p <= q*(a-x) || p >= q*(b-x))
	d=ConstGold*(e=(x >= xm ? a-x : b-x));
      else {
	d=p/q;
	u=x+d;
	if (u-a < tol2 || b-u < tol2)
	  d=_sign(tol1,xm-x);
      }
    } else {
      d=ConstGold*(e=(x >= xm ? a-x : b-x));
    }
    u=(fabs(d) >= tol1 ? x+d : x+_sign(tol1,d));

    fu = -calcLogLhood(u);
    if (fu <= fx) {
      if (u >= x) a=x; else b=x;
      _shift(v,w,x,u);
      _shift(fv,fw,fx,fu);
    } else {
      if (u < x) a=u; else b=u;
      if (fu <= fw || w == x) {
	v=w; w=u; fv=fw; fw=fu;
      } else if (fu <= fv || v == x || v == w) {
	v=u; fv=fu;
      }
    }
  }

  throw jconsistency_error("Too many iterations of Brent's method");
  return(-fx);
}


// ----- methods for class `BLTEstimatorBase' -----
//
void BLTEstimatorBase::_writeParm(FILE* fp)
{
  if (fp == NULL) return;

  RAPTParam(_alpha, _bias).write(fp);
}

void BLTEstimatorBase::findRange(double rho, double& min, double& max)
{
  min = -rho; max = rho;
}


// ----- methods for implementation base class `RAPTSeriesDerivs' -----
//
void RAPTSeriesDerivs::a_Alpha(double alpha, LaurentSeries& _sequence)
{
  _sequence = 0.0;

  _sequence[0] = -1.0;

  if (alpha == 0.0) {
    _sequence[2] = 1.0;
    return;
  }
  
  double alpha_n = 1.0;
  for (int n = 1; n < _sequence.len(); n++) {
    alpha_n *= alpha;
    _sequence[n] = alpha_n * ((n - 1) / (alpha * alpha) - (n + 1));
  }
}

bool RAPTSeriesDerivs::
b_RhoSmallTheta(double _rho, double _theta, LaurentSeries& _sequence,
		bool _invFlag)
{
  SmallTheta smallTheta = isThetaSmall(_theta);

  if (smallTheta == NotSmall) return false;

  double rho2   = _rho * _rho;
  double rho4   = rho2 * rho2;
  double theta2 = _theta * _theta;

  // coefficients of z^n for all n >= 2
  double rho_n_3 = 1.0 / _rho;
  for (int n = 2; n < _sequence.len(); n++) {
    int     n2   = n * n;
    int    index = (_invFlag) ?  -n : +n;
    double term  = 
      (  (6.0 - n*(n+2)*theta2) *(n+1) * (n+2) * rho4 / 6.0
	 - (6.0 - (n2+2)*theta2)  * n2 * rho2 / 3.0
	 + (6.0 - (n-2)*n*theta2) * (n-2) * (n-1) / 6.0) * rho_n_3;

    _sequence[index] =
      (smallTheta == ThetaZero || (n%2) == 0) ? term : -term;
    rho_n_3 *= _rho;
  }

  if (Trace & SequenceDerivs)
    cout << "Coefficients of (small) "
	 << ((_invFlag) ? "G_Rho(z):" : "B_Rho(z):") << endl
	 << _sequence << endl;
  
  return true;
}

void RAPTSeriesDerivs::
b_Rho(double _rho, double _theta, LaurentSeries& _sequence, bool _invFlag)
{
  if (_rho <= 0.0 || _rho >= 1.0)
    throw jparameter_error("Rho value (%g) is inappropriate", _rho);

  _sequence = 0.0;

  double cos_theta = cos(_theta);
  double sin_theta = sin(_theta);

  double rho2 = _rho * _rho;
  double rho4 = rho2 * rho2;

  // calculate coefficient of z^0
  _sequence[0] = 2.0 * _rho;
  
  // calculate coefficient of z^1
  int index  = (_invFlag) ? -1 : +1;
  _sequence[index] = 2.0 * (3.0 * rho2 - 1.0) * cos_theta;

  if (b_RhoSmallTheta(_rho, _theta, _sequence, _invFlag)) return;

  // calculate coefficients of z^n for n >= 2
  double rho_n_2 = 1.0;
  double rho_n_3 = 1.0 / _rho;
  for (int n = 2; n < _sequence.len(); n++) {
    index = (_invFlag) ?  -n : +n;

    _sequence[index] = (rho_n_3 / sin_theta) *
      ((n + 2) * rho4 * sin(_theta * (n + 1))
       - 2.0 * n * rho2 * cos_theta * sin(_theta * n)
       + (n - 2) * sin(_theta * (n - 1)));

    rho_n_3 *= _rho; rho_n_2 *= _rho;
  }

  if (Trace & SequenceDerivs)
    cout << "Coefficients of "
	 << ((_invFlag) ? "G_Rho(z):" : "B_Rho(z):") << endl
	 << _sequence << endl;
}

bool RAPTSeriesDerivs::
b_ThetaSmallTheta(double _rho, double _theta, LaurentSeries& _sequence,
		  bool _invFlag)
{
  SmallTheta smallTheta = isThetaSmall(_theta);

  if (smallTheta == NotSmall) return false;

  double rho2   = _rho * _rho;
  double rho4   = rho2 * rho2;
  double theta2 = _theta * _theta;

  // coefficients of z^n for all n >= 2
  double rho_n_2 = 1.0;
  for (int n = 2; n < _sequence.len(); n++) {
    int    n2    = n * n;
    int    n4    = n2 * n2;
    int    index = (_invFlag) ? -n : +n;
    double term  =
      (-(30.0 - (3*n2 + 6*n - 4)*theta2)*n*(n+1)*(n+2)*rho4 / 90.0
       +(30.0*(n2+2) - (3*n4+ 20*n2 -8)*theta2)*n*rho2 / 45.0
       -(30.0-(3*n2 - 6*n - 4)*theta2)*(n-2)*(n-1)*n / 90.0)*_theta*rho_n_2;

    _sequence[index] =
      (smallTheta == ThetaZero || (n%2) == 0) ? term : -term;
    rho_n_2 *= _rho;
  }

  if (Trace & SequenceDerivs)
    cout << "Coefficients of (small) "
	 << ((_invFlag) ? "G_Theta(z):" : "B_Theta(z):") << endl
	 << _sequence << endl;

  return true;
}

void RAPTSeriesDerivs::
b_Theta(double _rho, double _theta, LaurentSeries& _sequence, bool _invFlag)
{
  if (_rho <= 0.0 || _rho >= 1.0)
    throw jparameter_error("Rho value (%g) is inappropriate", _rho);

  _sequence = 0.0;

  double cos_theta = cos(_theta);
  double sin_theta = sin(_theta);

  double rho2 = _rho * _rho;
  double rho4 = rho2 * rho2;

  // calculate coefficient of z^0
  _sequence[0] = 0.0;

  // calculate coefficient of z^1
  int index  = (_invFlag) ? -1 : +1;
  _sequence[index] =  2.0 * _rho * (1.0 - rho2) * sin_theta;

  if (b_ThetaSmallTheta(_rho, _theta, _sequence, _invFlag)) return;

  // calculate coefficients of z^n for n >= 2
  double rho_n_2 = 1.0;
  double rho_n_3 = 1.0 / _rho;
  for (int n = 2; n < _sequence.len(); n++) {
    index = (_invFlag) ?  -n : +n;

    _sequence[index] = (rho_n_2 / sin_theta) *
      ((n + 1) * rho4 * cos(_theta * (n + 1))
       + 2.0 * rho2 * (sin_theta * sin(_theta * n)
		       - n * cos_theta * cos(_theta * n))
       + (n - 1) * cos(_theta * (n - 1)))
      - (cos_theta * rho_n_2 / (sin_theta * sin_theta)) *
      (rho4 * sin(_theta * (n + 1))
       - 2.0 * rho2 * cos_theta * sin(_theta * n)
       + sin(_theta * (n - 1)));
    rho_n_3 *= _rho; rho_n_2 *= _rho;
  }

  if (Trace & SequenceDerivs)
    cout << "Coefficients of "
	 << ((_invFlag) ? "G_Theta(z):" : "B_Theta(z):") << endl
	 << _sequence << endl;
}

void RAPTSeriesDerivs::
a_AlphaAlpha(double alpha, LaurentSeries& _sequence)
{
  _sequence = 0.0;

  _sequence[0] = 0.0;

  double alpha_n_1 = 1.0;
  double alpha_n_3 = 1.0 / (alpha * alpha);
  for (int n = 1; n < _sequence.len(); n++) {
    _sequence[n] = ((n-1)*(n-2)*alpha_n_3 - (n+1)*n*alpha_n_1);
    alpha_n_1 *= alpha;
    alpha_n_3 *= alpha;
  }

  if (Trace & SequenceDerivs)
    cout << "Coefficients of A_alpha_alpha(z):" << endl
	 << _sequence << endl;
}

bool RAPTSeriesDerivs::
b_RhoRhoSmallTheta(double _rho, double _theta, LaurentSeries& _sequence,
		   bool _invFlag)
{
  SmallTheta smallTheta = isThetaSmall(_theta);

  if (smallTheta == NotSmall) return false;

  double rho2   = _rho * _rho;
  double rho4   = rho2 * rho2;
  double theta2 = _theta * _theta;

  // coefficients of z^n for all n >= 2
  double rho_n_4 = 1.0 / rho2;
  for (int n = 2; n < _sequence.len(); n++) {
    int    n2    = n * n;
    int    index = (_invFlag) ? -n : +n;
    double term =
      ( (6.0 - (n+2)*n*theta2)*(n+1)*(n+1)*(n+2)*rho4/6.0
	-(6.0 - (n2+2)*theta2)*(n-1)*n2*rho2/3.0
	+(6.0 - (n-2)*n*theta2)*(n-3)*(n-2)*(n-1)/6.0)*rho_n_4;

    _sequence[index] =
      (smallTheta == ThetaZero || (n%2) == 0) ? term : -term;
    rho_n_4 *= _rho;
  }

  if (Trace & SequenceDerivs)
    cout << "Coefficients of (small)"
	 << ((_invFlag) ? "G_RhoRho(z):" : "B_RhoRho(z):") << endl
	 << _sequence << endl;

  return true;
}

void RAPTSeriesDerivs::
b_RhoRho(double _rho, double _theta, LaurentSeries& _sequence, bool _invFlag)
{
  if (_rho <= 0.0 || _rho >= 1.0)
    throw jparameter_error("Rho value (%g) is inappropriate", _rho);

  _sequence = 0.0;

  double cos_theta = cos(_theta);
  double sin_theta = sin(_theta);

  // coefficient of z^0
  _sequence[0] = 2.0;

  // coefficient of z^1
  int index  = (_invFlag) ? -1 : +1;
  _sequence[index] = 12.0 * _rho * cos_theta;

  if (b_RhoRhoSmallTheta(_rho, _theta, _sequence, _invFlag)) return;

  // coefficient(s) of z^n
  double rho_n   = _rho * _rho;
  double rho_n_2 = 1.0;
  double rho_n_4 = 1.0 / rho_n;
  for (int n = 2; n < _sequence.len(); n++) {
    index = (_invFlag) ?  -n : +n;
    _sequence[index] = (1.0 / sin_theta) *
      ((n+2)*(n+1)*rho_n*sin(_theta*(n+1))
       -2*n*(n-1)*rho_n_2*cos_theta*sin(_theta*n)
       +(n-2)*(n-3)*rho_n_4*sin(_theta*(n-1)));

    rho_n *= _rho; rho_n_2 *= _rho; rho_n_4 *= _rho;
  }

  if (Trace & SequenceDerivs)
    cout << "Coefficients of "
	 << ((_invFlag) ? "G_RhoRho(z):" : "B_RhoRho(z):") << endl
	 << _sequence << endl;
}

bool RAPTSeriesDerivs::
b_ThetaThetaSmallTheta(double _rho, double _theta, LaurentSeries& _sequence,
		       bool _invFlag)
{
  SmallTheta smallTheta = isThetaSmall(_theta);

  if (smallTheta == NotSmall) return false;

  double rho2   = _rho * _rho;
  double rho4   = rho2 * rho2;
  double theta2 = _theta * _theta;

  // coefficients of z^n for all n >= 2
  double rho_n_2 = 1.0;
  for (int n = 2; n < _sequence.len(); n++) {
    int     n2 = n * n;
    int     n4 = n2 * n2;
    int    index = (_invFlag) ?  -n : +n;
    double term  =
      (-(10.0 - (3*n2 + 6*n - 4)*theta2)*n*(n+1)*(n+2)*rho4 / 30.0
       +(10.0*(n2+2) - (3*n4 + 20*n2 - 8)*theta2)*n*rho2 / 15.0
       -(10.0-(3*n2 - 6*n - 4)*theta2)*(n-2)*(n-1)*n / 30.0) * rho_n_2;

    _sequence[index] =
      (smallTheta == ThetaZero || (n%2) == 0) ? term : -term;
    rho_n_2 *= _rho;
  }

  if (Trace & SequenceDerivs)
    cout << "Coefficients of (small) "
	 << ((_invFlag) ? "G_ThetaTheta(z):" : "B_ThetaTheta(z):") << endl
	 << _sequence << endl;

  return true;
}

void RAPTSeriesDerivs::
b_ThetaTheta(double _rho, double _theta, LaurentSeries& _sequence,
	     bool _invFlag)
{
  if (_rho <= 0.0 || _rho >= 1.0)
    throw jparameter_error("Rho value (%g) is inappropriate", _rho);

  _sequence = 0.0;

  double cos_theta = cos(_theta);
  double sin_theta = sin(_theta);

  // coefficient of z^0
  _sequence[0] = 0.0;

  // coefficient of z^1
  int index  = (_invFlag) ? -1 : +1;
  _sequence[index] = 2.0 * _rho * (1.0 - _rho * _rho) * cos_theta;

  if (b_ThetaThetaSmallTheta(_rho, _theta, _sequence, _invFlag)) return;

  // coefficient(s) of z^n for all n >= 2
  double rho_n   = _rho * _rho;
  double rho_n2  = rho_n * rho_n;
  double rho_n_2 = 1.0;
  for (int n = 2; n < _sequence.len(); n++) {
    index = (_invFlag) ?  -n : +n;
    _sequence[index] = -2.0*(cos_theta / (sin_theta * sin_theta))
      *((n+1)*rho_n2*cos(_theta*(n+1))
	+2.0*rho_n*(sin_theta*sin(_theta*n) - n*cos_theta*cos(_theta*n))
	+(n-1)*rho_n_2*cos(_theta*(n-1)))
      +((1.0 + cos_theta * cos_theta) / (sin_theta*sin_theta*sin_theta))
      *(rho_n2*sin(_theta*(n+1)) - 2.0*rho_n*cos_theta*sin(_theta*n)
	+rho_n_2*sin(_theta*(n-1)))
      +(1.0/sin_theta)*
      (-(n+1)*(n+1)*rho_n2*sin(_theta*(n+1))
       +2.0*rho_n*(2.0*n*cos(_theta*n)*sin_theta + (n*n + 1)*cos_theta*sin(_theta*n))
       -(n-1)*(n-1)*rho_n_2*sin(_theta*(n-1)));

    rho_n *= _rho; rho_n2 *= _rho; rho_n_2 *= _rho;
  }

  if (Trace & SequenceDerivs)
    cout << "Coefficients of "
	 << ((_invFlag) ? "G_ThetaTheta(z):" : "B_ThetaTheta(z):") << endl
	 << _sequence << endl;
}

bool RAPTSeriesDerivs::
b_RhoThetaSmallTheta(double _rho, double _theta, LaurentSeries& _sequence,
		     bool _invFlag)
{
  SmallTheta smallTheta = isThetaSmall(_theta);

  if (smallTheta == NotSmall) return false;

  double rho2   = _rho * _rho;
  double rho4   = rho2 * rho2;
  double theta2 = _theta * _theta;

  // coefficients of z^n for all n >= 2
  double rho_n_3 = 1.0 / _rho;
  for (int n = 2; n < _sequence.len(); n++) {
    int     n2    = n * n;
    int     n4    = n2 * n2;
    int    index = (_invFlag) ? -n : +n;
    double term  =
      (-(30.0 - (3*n2 + 6*n - 4)*theta2)*(n+1)*(n+2)*(n+2)*rho4 / 90.0
       +(30.0*(n2+2) - (3*n4 + 20*n2 - 8)*theta2)*n*rho2 /45.0
       -(30.0 - (3*n2 - 6*n - 4)*theta2)*(n-2)*(n-2)*(n-1) / 90.0)
      * n * _theta * rho_n_3;

    _sequence[index] = 
      (smallTheta == ThetaZero || (n%2) == 0) ? term : -term;
    rho_n_3 *= _rho;
  }

  if (Trace & SequenceDerivs)
    cout << "Coefficients of (small) "
	 << ((_invFlag) ? "G_RhoTheta(z):" : "B_RhoTheta(z):") << endl
	 << _sequence << endl;

  return true;
}

void RAPTSeriesDerivs::
b_RhoTheta(double _rho, double _theta, LaurentSeries& _sequence, bool _invFlag)
{
  if (_rho <= 0.0 || _rho >= 1.0)
    throw jparameter_error("Rho value (%g) is inappropriate", _rho);

  _sequence = 0.0;

  double cos_theta = cos(_theta);
  double sin_theta = sin(_theta);

  // coefficient of z^0
  _sequence[0] = 0.0;

  // coefficient of z^1
  int index = (_invFlag) ? -1 : +1;
  _sequence[index] = -2.0 * (3.0*_rho*_rho - 1.0) * sin_theta;

  if (b_RhoThetaSmallTheta(_rho, _theta, _sequence, _invFlag)) return;

  // coefficient(s) of z^n
  double rho_n1  = _rho * _rho * _rho;
  double rho_n_1 = _rho;
  double rho_n_3 = 1.0 / _rho;
  for (int n = 2; n < _sequence.len(); n++) {
    index = (_invFlag) ?  -n : +n;
    _sequence[index] = (1.0 / sin_theta) *
      ((n+2)*(n+1)*rho_n1*cos(_theta*(n+1))
       +2*n*rho_n_1*(sin_theta*sin(_theta*n) - n*cos_theta*cos(_theta*n))
       +(n-1)*(n-2)*rho_n_3*cos((n-1)*_theta))
      -(cos_theta/(sin_theta*sin_theta)) *
      ((n+2)*rho_n1*sin(_theta*(n+1))
       - 2.0*n*rho_n_1*cos_theta*sin(_theta*n)
       +(n-2)*rho_n_3*sin(_theta*(n-1)));

    rho_n1 *= _rho; rho_n_1 *= _rho; rho_n_3 *= _rho;
  }

  if (Trace & SequenceDerivs)
    cout << "Coefficients of "
	 << ((_invFlag) ? "G_RhoTheta(z):" : "B_RhoTheta(z):") << endl
	 << _sequence << endl;
}

// utility procedures
void matrixNegate(gsl_matrix* mat)
{
  unsigned m = mat->size1;
  unsigned n = mat->size2;
  
  for (unsigned i = 0; i < m; i++)
    for (unsigned j = 0; j < n; j++)
      gsl_matrix_set(mat, i, j, -gsl_matrix_get(mat, i, j));
}

static const double MaxEigenValueRatio = 1.0e+06;

static void makePositiveDefinite(gsl_vector* eigenVals)
{
  unsigned n = eigenVals->size;

  double  maxEvalue = -HUGE;
  for (unsigned i = 0; i < n; i++)
    if (gsl_vector_get(eigenVals, i) > maxEvalue)
      maxEvalue = gsl_vector_get(eigenVals, i);

  double minEvalue = maxEvalue / MaxEigenValueRatio;
  for (unsigned i = 0; i < n; i++) {
    gsl_vector_set(eigenVals, i, fabs(gsl_vector_get(eigenVals, i)));
    if (gsl_vector_get(eigenVals, i) < minEvalue)
      gsl_vector_set(eigenVals, i, minEvalue);
  }
}


// ----- methods for class `APTEstimatorBase' -----
//
const double APTEstimatorBase::GradientTolerance = 0.1;
const double APTEstimatorBase::SearchTolerance   = 5.0e-05;
unsigned     APTEstimatorBase::MaxIterations     = 10;

APTEstimatorBase::APTEstimatorBase()
{
  throw jconsistency_error("Incorrect constructor.");
}

APTEstimatorBase::APTEstimatorBase(unsigned noItns)
  : _xi(NoAllPassParams), _p1(NoAllPassParams),
    _dbh_daj(NULL), _d2bh_daj_dai(NULL),
    direction(NoAllPassParams)
{
  _hessian = gsl_matrix_alloc(NoAllPassParams, NoAllPassParams);

  MaxIterations = noItns;

  for (UnShrt i = 0; i < NoAllPassParams; i++)
    _xi[i] = 0.0;
    
  _gradMatrix    = new gsl_matrix*[NoAllPassParams];
  _cepGradMatrix = new gsl_matrix*[NoAllPassParams];
  for (UnShrt i = 0; i < NoAllPassParams; i++) {
    _gradMatrix[i]    = gsl_matrix_alloc(_subFeatLen, _orgSubFeatLen);
    _cepGradMatrix[i] = gsl_matrix_alloc(_cepSubFeatLen, _cepOrgSubFeatLen);
  }

  _hessMatrix    = new gsl_matrix**[NoAllPassParams];
  _cepHessMatrix = new gsl_matrix**[NoAllPassParams];
  for (UnShrt i = 0; i < NoAllPassParams; i++) {
    _hessMatrix[i]    = new gsl_matrix*[NoAllPassParams];
    _cepHessMatrix[i] = new gsl_matrix*[NoAllPassParams];
    for (UnShrt j = 0; j < NoAllPassParams; j++) {
      _hessMatrix[i][j] =
	(j < i) ? (gsl_matrix*) NULL : gsl_matrix_alloc(_subFeatLen, _orgSubFeatLen);
      _cepHessMatrix[i][j] =
	(j < i) ? (gsl_matrix*) NULL : gsl_matrix_alloc(_cepSubFeatLen, _cepOrgSubFeatLen);
    }
  }
}

APTEstimatorBase::~APTEstimatorBase()
{
   gsl_matrix_free(_hessian);

   for (UnShrt i = 0; i < NoAllPassParams; i++) {
      gsl_matrix_free(_gradMatrix[i]);  gsl_matrix_free(_cepGradMatrix[i]);
   }
   delete[] _gradMatrix;  delete[] _cepGradMatrix;

   for (UnShrt i = 0; i < NoAllPassParams; i++) {
     for (UnShrt j = i; j < NoAllPassParams; j++) {
       gsl_matrix_free(_hessMatrix[i][j]);  gsl_matrix_free(_cepHessMatrix[i][j]);
     }
     delete[] _hessMatrix[i];  delete[] _cepHessMatrix[i];
   }
   delete[] _hessMatrix;  delete[] _cepHessMatrix;
}

bool APTEstimatorBase::
converged(double newFp, double oldFp)
{
  if (newFp < oldFp) {
    printf("Objective at new point (%g) is worse than at old (%g)",
    newFp, oldFp);
    return true;
  }

  if (2.0*fabs(oldFp - newFp)
      > SearchTolerance * (fabs(oldFp)+fabs(newFp))) return false;

  double sum = 0.0;
  for (UnShrt i = 0; i < NoAllPassParams; i++)
    sum += _xi[i] * _xi[i];

  return sqrt(sum) <= GradientTolerance;
}

double APTEstimatorBase::optimize()
{
  double fret = -HUGE;

  /*
  p[0] = 0.03;
  p[1] = 0.4; p[2] =  4.0e-03;
  p[3] = 0.8; p[4] = -8.0e-03;
  */

  if (NoAllPassParams == 1) {  // hack to handle 1-parameter SLAPT
    derivFlag = None;
    _xi[0] = 1.0; p[0] = 0.0;
    double ax = -0.05, bx = 0.0, cx, xmin;
    bracket(ax, bx, cx);
    double fp = brentsMethod(ax, bx, cx, xmin);
    p[0] = xmin;

    return fp;
  }

  for (unsigned iter = 0; iter < (MaxIterations * (NoAllPassParams / 4)); iter++) {
    derivFlag = Hessian;
    fret = calcLogLhood(/* rho= */ 0.0);

    direction.solve(_hessian, _xi, _xi);

    if (Trace & SearchDirection) {
      cout << " Newton search direction: " << endl;
      for (UnShrt j = 0; j < NoAllPassParams; j++)
	cout << "  " << _xi[j];
      cout << endl;
    }

    derivFlag = None;
    double ax = 0.8, bx = 0.0, cx, xmin;
    while(bracket(ax, bx, cx) == false) { ax /= 4; bx = 0.0; }
    brentsMethod(ax, bx, cx, xmin);
    for (UnShrt j = 0; j < NoAllPassParams; j++)
      p[j] += xmin * _xi[j];

    derivFlag = Gradient;
    double fp = calcLogLhood(/* rho= */ 0.0);
    if (converged(fp, fret)) return fp;

    if (Trace & Iterates) {
      cout << "New point:" << endl;
      for (UnShrt j = 0; j < NoAllPassParams; j++)
	cout << "  " << p[j];
      cout << endl;
    }
  }

  printf("Too many iterations of Newton's method.");

  return fret;
}

void APTEstimatorBase::calcGradientMatrix(const CoeffSequence& allPassParams)
{
  int last = _laurentDerivs.last();
  for (UnShrt ipar = 0; ipar < NoAllPassParams; ipar++) {
    gsl_matrix* cepGradMatrix = _cepGradMatrix[ipar];
    gsl_matrix_set_zero(cepGradMatrix);
    calcSeriesDerivs(allPassParams, _laurentDerivs, ipar);

    for (UnShrt m = 1; m < _cepOrgSubFeatLen; m++) {

      // calculate n = 0 component
      double sum = 0.0;
      for (int k = -last; k <= last; k++)
	sum += _qmn[m-1][-k] * _laurentDerivs[k];
      gsl_matrix_set(cepGradMatrix, 0, m, m * sum);

      // calculate n >= 1 components
      for (UnShrt n = 1; n < _cepSubFeatLen; n++) {
	sum = 0.0;
	for (int k = n - last; k <= -n + last; k++)
	  sum += (_qmn[m-1][(n-k)] + _qmn[m-1][(-n-k)]) *
	    _laurentDerivs[k];
	gsl_matrix_set(cepGradMatrix, n, m, m * sum);
      }
    }
    /*
    cout << endl << " Gradient matrix " << ipar << endl;
    m_output(cepGradMatrix);
    */

    _ldaMatrix(cepGradMatrix, _gradMatrix[ipar]);
  }
}

void APTEstimatorBase::calcHessianMatrix(const CoeffSequence& allPassParams)
{
  int last = q_Alpha.last();
  for (UnShrt ipar = 0; ipar < NoAllPassParams; ipar++) {
    for (UnShrt jpar = ipar; jpar < NoAllPassParams; jpar++) {
      gsl_matrix* cepHessMatrix = _cepHessMatrix[ipar][jpar];

      gsl_matrix_set_zero(cepHessMatrix);

      calcSeriesDerivs(allPassParams, q_Alpha, ipar);
      calcSeriesDerivs(allPassParams, q_Rho,   jpar);
      CauchyProduct(q_Alpha, q_Rho, qp_AlphaRho);
      calcSeries2ndDerivs(allPassParams, q_AlphaRho, ipar, jpar);

      for (UnShrt m = 1; m < _cepOrgSubFeatLen; m++) {

	// calculate n = 0 component
	double sum1 = 0.0, sum2 = 0.0;
	if (m >= 2) {
	  for (int k = -last; k <= last; k++)
	    sum1 += _qmn[m-2][-k] * qp_AlphaRho[k];
	  sum1 *= m * (m-1);
	}

	for (int k = -last; k <= last; k++)
	  sum2 += _qmn[m-1][-k] * q_AlphaRho[k];
	sum2 *= m;

	gsl_matrix_set(cepHessMatrix, 0, m, sum1 + sum2);

	// calculate n >= 1 components
	for (UnShrt n = 1; n < _cepSubFeatLen; n++) {
	  sum1 = sum2 = 0.0;

	  if (m >= 2) {
	    for (int k = n - last; k <= -n + last; k++)
	      sum1 += (_qmn[m-2][(n-k)] + _qmn[m-2][(-n-k)])
		 * qp_AlphaRho[k];
	    sum1 *= m * (m-1);
	  }

	  for (int k = n - last; k <= -n + last; k++)
	    sum2 += (_qmn[m-1][(n-k)] + _qmn[m-1][(-n-k)]) * q_AlphaRho[k];
	  sum2 *= m;

	  gsl_matrix_set(cepHessMatrix, n, m, sum1 + sum2);
	}
      }

      /*
      cout << endl << " Hessian matrix " << ipar << " " << jpar << endl;
      m_output(cepHessMatrix);
      */

      _ldaMatrix(cepHessMatrix, _hessMatrix[ipar][jpar]);
    }
  }
}

void APTEstimatorBase::
xformGradVec(const NaturalVector& initFeat, NaturalVector& parDeriv,
	     unsigned ipar, bool useBias)
{
  const gsl_matrix* gradient = _gradMatrix[ipar];
  for (UnShrt isub = 0; isub < _nSubFeat; isub++) {
    for (UnShrt n = 0; n < _subFeatLen; n++) {
      double sum = 0.0;

      if ((initFeat.nSubFeat() == parDeriv.nSubFeat()) &&
	  (initFeat.nSubFeat() == _nSubFeat)) {

	// Normal Case: _gradMatrix is only as large as a single
	//              cepstral sub-feature
	for (int m = 1; m < _orgSubFeatLen; m++)
	  sum += gsl_matrix_get(gradient, n, m) * initFeat(isub, m);

      } else if (initFeat.featLen() == orgFeatLen()) {

	// Normal LDA Case: _gradMatrix is as large as entire original mean
	const UnShrt olen = initFeat.featLen();
	for (int m = 0; m < olen; m++)
	  sum += gsl_matrix_get(gradient, n, m) * initFeat(m);

      } else
	throw jconsistency_error("Cannot handle extended mean case.");

      parDeriv(isub, n) = sum;
    }
  }

  UnShrt bSize = _bias.featLen();
  if (bSize == 0 || useBias == false) return;

  for (UnShrt n = 0; n < bSize; n++)
    parDeriv(n) += gsl_matrix_get(_dbh_daj, n, ipar);
}

void APTEstimatorBase::
xformHessVec(const NaturalVector& initFeat, NaturalVector& parDeriv,
	     unsigned ipar, unsigned jpar, bool useBias)
{
  assert(ipar <= jpar);

  const gsl_matrix* hessian = _hessMatrix[ipar][jpar];
  for (UnShrt isub = 0; isub < _nSubFeat; isub++) {
    for (UnShrt n = 0; n < _subFeatLen; n++) {
      double sum = 0.0;

      if ((initFeat.nSubFeat() == parDeriv.nSubFeat()) &&
	  (initFeat.nSubFeat() == _nSubFeat)) {

	// Normal Case: _hessMatrix is only as large as a single
	//              cepstral sub-feature
	for (int m = 1; m < _orgSubFeatLen; m++)
	  sum += gsl_matrix_get(hessian, n, m) * initFeat(isub, m);

      } else if (initFeat.featLen() == orgFeatLen()) {

	// Normal LDA Case: _hessMatrix is as large as entire original mean
	const UnShrt olen = initFeat.featLen();
	for (int m = 0; m < olen; m++)
	  sum += gsl_matrix_get(hessian, n, m) * initFeat(m);

      } else
	throw jconsistency_error("Cannot handle extended mean case.");

      parDeriv(isub, n) = sum;
    }
  }

  UnShrt bSize = _bias.featLen();
  if (bSize == 0 || useBias == false) return;

  for (UnShrt n = 0; n < bSize; n++)
    parDeriv(n) += gsl_matrix_get(_d2bh_daj_dai[n], ipar, jpar);
}

void APTEstimatorBase::copyLowerTriangle(gsl_matrix* mat)
{
  assert(mat->size1 == mat->size2);

  u_int m = mat->size1;
  for (u_int i = 1; i < m; i++)
    for (u_int j = 0; j < i; j++)
      gsl_matrix_set(mat, i, j, gsl_matrix_get(mat, j, i));
}


// --- methods for nested class `APTEstimatorBase::NewtonDirection' ---
//
APTEstimatorBase::NewtonDirection::NewtonDirection()
  : nParms(0)
{
  throw jconsistency_error("Incorrect constructor.");
}

APTEstimatorBase::NewtonDirection::NewtonDirection(UnShrt n)
  : nParms(n)
{
  _eigenVecs  = gsl_matrix_alloc(nParms, nParms);
  _tempVector = gsl_vector_alloc(nParms);
  _eigenVals  = gsl_vector_alloc(nParms);
  _newGrad    = gsl_vector_alloc(nParms);
  _pk         = gsl_vector_alloc(nParms);
  _workSpace  = gsl_eigen_symmv_alloc(nParms);
}

APTEstimatorBase::NewtonDirection::~NewtonDirection()
{
  gsl_vector_free(_newGrad);
  gsl_vector_free(_pk);
  gsl_vector_free(_tempVector);
  gsl_vector_free(_eigenVals);
  gsl_matrix_free(_eigenVecs);
  gsl_eigen_symmv_free(_workSpace);
}

void APTEstimatorBase::NewtonDirection::
solve(gsl_matrix* hess, CoeffSequence& grad, CoeffSequence& newDirection)
{
  for (UnShrt i = 0; i < nParms; i++)
    gsl_vector_set(_newGrad, i, grad[i]);

  matrixNegate(hess);
  gsl_eigen_symmv (hess, _eigenVals, _eigenVecs, _workSpace);
  makePositiveDefinite(_eigenVals);

  gsl_blas_dgemv(CblasTrans, 1.0, _eigenVecs, _newGrad, 0.0, _tempVector);
  for (UnShrt i = 0; i < nParms; i++)
    gsl_vector_set(_tempVector, i,
		   gsl_vector_get(_tempVector, i)
		   / gsl_vector_get(_eigenVals, i));
  gsl_blas_dgemv(CblasNoTrans, 1.0, _eigenVecs, _tempVector, 0.0, _pk);

  for (UnShrt i = 0; i < nParms; i++)
    newDirection[i] = gsl_vector_get(_pk, i);
}


// ----- methods for class `RAPTEstimatorBase' -----
//
// initialize all-pass parameters to default values
//
RAPTEstimatorBase::RAPTEstimatorBase()
{
   _convMatrix   = gsl_matrix_alloc(NoAllPassParams, NoAllPassParams);
   _rectHessian  = gsl_matrix_alloc(NoAllPassParams, NoAllPassParams);
}

RAPTEstimatorBase::~RAPTEstimatorBase()
{
   gsl_matrix_free(_convMatrix);
   gsl_matrix_free(_rectHessian);
}

void RAPTEstimatorBase::_resetParams(UnShrt nParams)
{
  if (nParams == 0) {
    p[0] = 0.001;
    nParams++;
  }

  short NoDiffParams = NoAllPassParams - nParams;
  if (NoDiffParams < 0)
    throw jparameter_error("Cannot reduce the number of RAPT parameters from %d to %d.",
			   nParams, NoAllPassParams);
  else if (NoDiffParams == 0)
    return;

  if (NoDiffParams % 4 != 0)
    throw jparameter_error("No. of different parameters (%d) is not correct.", NoDiffParams);
  UnShrt NoDiffPairs    = NoDiffParams  / 4;
  UnShrt NoCurrentPairs = (nParams - 1) / 4;

  const double InitialRho = 0.03;
  const double DeltaArg   = M_PI / NoDiffPairs;
  const double InitialArg = M_PI / (2 * NoDiffPairs);

  for (UnShrt ipair = 0; ipair < NoDiffPairs; ipair++) {
    p[4*(NoCurrentPairs+ipair)+1] = p[4*(NoCurrentPairs+ipair)+3] =
      InitialRho * cos((ipair * DeltaArg) + InitialArg);
    p[4*(NoCurrentPairs+ipair)+2] = p[4*(NoCurrentPairs+ipair)+4] =
      InitialRho * sin((ipair * DeltaArg) + InitialArg);
  }

  /*
    These values were used to validate the series expansions, etc.
    p[0] = 0.05;
    p[1] = p[2] = 0.1;
    p[3] = -0.2; p[4] = 0.2;
  */
}

void RAPTEstimatorBase::_writeParm(FILE* fp)
{
  if (fp == NULL) return;

  RAPTParam(p, _bias).write(fp);
}

void RAPTEstimatorBase::
calcSeriesDerivs(const CoeffSequence& _params, LaurentSeries& _sequence,
		 unsigned derivIndex)
{
  double alpha = _params[0];    // always unpack in the same order
  if (derivIndex == 0)
    a_Alpha(alpha, _sequence);
  else
    bilinear(alpha, _sequence);

  for (unsigned ipair = 0; ipair < NoComplexPairs; ipair++) {
    for (unsigned i = 0; i < 2; i++) {
      bool invFlag = (i == 0) ? false : true;
      unsigned rhoIndex   = (4*ipair) + (2*i) + 1;
      unsigned thetaIndex = (4*ipair) + (2*i) + 2;

      double x  = _params[rhoIndex];
      double y  = _params[thetaIndex];

      double _rho, _theta;
      rect2polar(x,  y,  _rho, _theta);

      _temp1 = _sequence;

      if (derivIndex == rhoIndex)
	b_Rho(_rho, _theta, _temp2, invFlag);
      else if (derivIndex == thetaIndex)
	b_Theta(_rho, _theta, _temp2, invFlag);
      else
	allPass(_rho, _theta, _temp2, invFlag);

      CauchyProduct(_temp1, _temp2, _sequence);
    }
  }

  if (Trace & SequenceDerivs)
    cout << "Sequence Derivatives: " << derivIndex << endl
	 << _sequence << endl;
}

void RAPTEstimatorBase::
calcSeries2ndDerivs(const CoeffSequence& _params, LaurentSeries& _sequence,
		    unsigned derivIndex1, unsigned derivIndex2)
{
  assert(derivIndex1 <= derivIndex2);

  double alpha = _params[0];    // always unpack in the same order
  if (derivIndex1 == 0)
    if (derivIndex2 == 0)
      a_AlphaAlpha(alpha, _sequence);
    else
      a_Alpha(alpha, _sequence);
  else
    bilinear(alpha, _sequence);

  for (unsigned ipair = 0; ipair < NoComplexPairs; ipair++) {
    for (unsigned i = 0; i < 2; i++) {
      bool invFlag = (i == 0) ? false : true;
      unsigned rhoIndex   = (4*ipair) + (2*i) + 1;
      unsigned thetaIndex = (4*ipair) + (2*i) + 2;

      double x = _params[rhoIndex];
      double y = _params[thetaIndex];

      double _rho, _theta;
      rect2polar(x,  y,  _rho, _theta);

      _temp1 = _sequence;

      // perhaps not the cleanest logic, but correct nonetheless
      if (derivIndex1 == rhoIndex)
	if (derivIndex2 == rhoIndex)
	  b_RhoRho(_rho, _theta, _temp2, invFlag);
	else if (derivIndex2 == thetaIndex)
	  b_RhoTheta(_rho, _theta, _temp2, invFlag);
	else
	  b_Rho(_rho, _theta, _temp2, invFlag);
      else if (derivIndex1 == thetaIndex)
	if (derivIndex2 == thetaIndex)
	  b_ThetaTheta(_rho, _theta, _temp2, invFlag);
	else
	  b_Theta(_rho, _theta, _temp2, invFlag);
      else if (derivIndex2 == rhoIndex)
	b_Rho(_rho, _theta, _temp2, invFlag);
      else if (derivIndex2 == thetaIndex)
	b_Theta(_rho, _theta, _temp2, invFlag);
      else
	allPass(_rho, _theta, _temp2, invFlag);

      CauchyProduct(_temp1, _temp2, _sequence);
    }
  }

  if (Trace & SequenceDerivs)
    cout << "Derivatives " << derivIndex1 << " " << derivIndex2 << endl
	 << _sequence << endl;
}

void RAPTEstimatorBase::calcConvMatrix()
{
  gsl_matrix_set_zero(_convMatrix);
  gsl_matrix_set(_convMatrix, 0, 0, 1.0);
  for (unsigned ipair = 0; ipair < NoComplexPairs; ipair++) {
    for (int i = 0; i < 2; i++) {
      unsigned  realIndex = (4*ipair) + (2*i) + 1;
      unsigned  imagIndex = (4*ipair) + (2*i) + 2;

      double x    = _p1[realIndex];
      double y    = _p1[imagIndex];
      double rho2 = (x * x) + (y * y);
      double rho  = sqrt(rho2);

      // _convMatrix[alpha_i][x_j]
      gsl_matrix_set(_convMatrix, realIndex, realIndex,   x / rho);  // = drho_dx
      gsl_matrix_set(_convMatrix, realIndex, imagIndex,   y / rho);  // = drho_dy
      gsl_matrix_set(_convMatrix, imagIndex, realIndex, - y / rho2); // = dtheta_dx
      gsl_matrix_set(_convMatrix, imagIndex, imagIndex,   x / rho2); // = dtheta_dy
    }
  }
}

// convert partial derivatives in 'xi' from polar
// to rectangular coordinates
//
void RAPTEstimatorBase::convertGrad2Rect()
{
  calcConvMatrix();

  for (unsigned ipair = 0; ipair < NoComplexPairs; ipair++) {
    for (int i = 0; i < 2; i++) {
      unsigned realIndex = (4*ipair) + (2*i) + 1;
      unsigned imagIndex = (4*ipair) + (2*i) + 2;

      double dRho_dx   = gsl_matrix_get(_convMatrix, realIndex, realIndex);
      double dRho_dy   = gsl_matrix_get(_convMatrix, realIndex, imagIndex);
      double dTheta_dx = gsl_matrix_get(_convMatrix, imagIndex, realIndex);
      double dTheta_dy = gsl_matrix_get(_convMatrix, imagIndex, imagIndex);

      double dLhood_dRho   = _xi[realIndex];
      double dLhood_dTheta = _xi[imagIndex];

      // dLhood_dx :=
      _xi[realIndex] =
	(dLhood_dRho * dRho_dx) + (dLhood_dTheta * dTheta_dx);

      // dLhood_dy :=
      _xi[imagIndex] =
	(dLhood_dRho * dRho_dy) + (dLhood_dTheta * dTheta_dy);
    }
  }
}

void RAPTEstimatorBase::convertHess2Rect()
{
  calcConvMatrix();

  for (UnShrt i = 0; i < NoAllPassParams; i++) {
    for (UnShrt j = i; j < NoAllPassParams; j++) {

      double sum = 0.0;
      for (UnShrt kpar = 0; kpar < NoAllPassParams; kpar++)
	for (UnShrt lpar = 0; lpar < NoAllPassParams; lpar++)
	  sum += gsl_matrix_get(_hessian, kpar, lpar) * gsl_matrix_get(_convMatrix, kpar, i) * gsl_matrix_get(_convMatrix, lpar, j);
      gsl_matrix_set(_rectHessian, i, j, sum);
    }
  }

  for (unsigned ipair = 0; ipair < NoComplexPairs; ipair++) {
    for (int i = 0; i < 2; i++) {
      unsigned realIndex = (4*ipair) + (2*i) + 1;
      unsigned imagIndex = (4*ipair) + (2*i) + 2;

      double L_rho    =   _xi[realIndex];  // assumes gradient is
      double L_theta  =   _xi[imagIndex];  // still in polar form

      double x        =   _p1[realIndex];
      double y        =   _p1[imagIndex];

      double rho      =   sqrt((x * x) + (y * y));
      double rho3     =   rho * rho * rho;
      double rho4     =   rho3 * rho;

      double rho_xx   =   y * y / rho3;
      double rho_yy   =   x * x / rho3;
      double rho_xy   = - x * y / rho3;

      double theta_xx =   2.0 * x * y / rho4;
      double theta_yy = - theta_xx;
      double theta_xy =   (y*y - x*x) / rho4;

      // L_xx +=
      gsl_matrix_set(_rectHessian, realIndex, realIndex,
		     gsl_matrix_get(_rectHessian, realIndex, realIndex)
		     + (L_rho * rho_xx) + (L_theta * theta_xx));

      // L_yy +=
      gsl_matrix_set(_rectHessian, imagIndex, imagIndex,
		     gsl_matrix_get(_rectHessian, imagIndex, imagIndex)
		     + (L_rho * rho_yy) + (L_theta * theta_yy));
  
      // L_xy +=
      gsl_matrix_set(_rectHessian, realIndex, imagIndex,
		     gsl_matrix_get(_rectHessian, realIndex, imagIndex)
		     + (L_rho * rho_xy) + (L_theta * theta_xy));
    }
  }

  copyLowerTriangle(_rectHessian);
  gsl_matrix_memcpy(_hessian, _rectHessian);
}

void RAPTEstimatorBase::
boundRange(double x, double y, double dx, double dy, double rho,
	   double& min, double& max)
{
  double q1 = (x * dx)  + (y * dy);
  double q2 = (dx * dx) + (dy * dy);
  double q3 = (rho * rho) - (x * x) - (y * y);

  max = (-q1 + sqrt((q1 * q1) + (q2 * q3))) / q2;
  min = (-q1 - sqrt((q1 * q1) + (q2 * q3))) / q2;
}

// determine the upper and lower bounds on step size
//
void RAPTEstimatorBase::findRange(double rho, double& mmin, double& mmax)
{
  double alpha  =  p[0];
  double d_alph = _xi[0];
  double mn     = (-rho - alpha ) / d_alph;
  double mx     = ( rho - alpha ) / d_alph;

  mmin = min(mn, mx);  mmax = max(mn, mx);

  for (unsigned ipair = 0; ipair < NoComplexPairs; ipair++) {
    for (unsigned i = 0; i < 2; i++) {
      unsigned realIndex = (4*ipair) + (2*i) + 1;
      unsigned imagIndex = (4*ipair) + (2*i) + 2;

      double x = p[realIndex];
      double y = p[imagIndex];

      double dx = _xi[realIndex];
      double dy = _xi[imagIndex];

      double mn, mx;
      boundRange(x, y, dx, dy, rho, mn, mx);
      mmin = max(mmin, mn);
      mmax = min(mmax, mx);
    }
  }
}

double RAPTEstimatorBase::barrier()
{
  double _rho =  _p1[0];
  double diff =  MaximumRho - _rho;
  double sum  = -BarrierScale / (diff * diff);

  for (unsigned ipair = 0; ipair < NoComplexPairs; ipair++) {
    for (unsigned i = 0; i < 2; i++) {
      unsigned realIndex = (4*ipair) + (2*i) + 1;
      unsigned imagIndex = (4*ipair) + (2*i) + 2;

      double x = _p1[realIndex];
      double y = _p1[imagIndex];

      _rho = sqrt(x*x + y*y);
      diff = MaximumRho - _rho;
      sum -= BarrierScale / (diff * diff);
    }
  }
  return sum;
}

void RAPTEstimatorBase::addGradBarrier()
{
  double _rho =  _p1[0];
  double diff =  MaximumRho - _rho;
  _xi[0] -= 2.0 * BarrierScale / (diff * diff * diff);

  for (unsigned ipair = 0; ipair < NoComplexPairs; ipair++) {
    for (unsigned i = 0; i < 2; i++) {
      unsigned realIndex = (4*ipair) + (2*i) + 1;
      unsigned imagIndex = (4*ipair) + (2*i) + 2;

      double x = _p1[realIndex];
      double y = _p1[imagIndex];

      _rho = sqrt(x*x + y*y);
      diff = MaximumRho - _rho;
      _xi[realIndex] -= 2.0 * BarrierScale / (diff * diff * diff);
    }
  }
}

void RAPTEstimatorBase::addHessBarrier()
{
  double _rho =  _p1[0];
  double diff =  MaximumRho - _rho;
  gsl_matrix_set(_hessian, 0, 0,
		 gsl_matrix_get(_hessian, 0, 0)
		 - 6.0 * BarrierScale / (diff * diff * diff * diff));

  for (unsigned ipair = 0; ipair < NoComplexPairs; ipair++) {
    for (unsigned i = 0; i < 2; i++) {
      unsigned realIndex = (4*ipair) + (2*i) + 1;
      unsigned imagIndex = (4*ipair) + (2*i) + 2;

      double x = _p1[realIndex];
      double y = _p1[imagIndex];

      _rho = sqrt(x*x + y*y);
      diff = MaximumRho - _rho;
      gsl_matrix_set(_hessian, realIndex, realIndex,
		     gsl_matrix_get(_hessian, realIndex, realIndex)
		     - 6.0 * BarrierScale / (diff * diff * diff * diff));
    }
  }
}


// ----- methods for class `SLAPTEstimatorBase' -----
//
void SLAPTEstimatorBase::_writeParm(FILE* fp)
{
  if (fp == NULL) return;

  SLAPTParam(p, _bias).write(fp);
}

void SLAPTEstimatorBase::
calcSeriesDerivs(const CoeffSequence& _params, LaurentSeries& _sequence,
		 unsigned derivIndex)
{
  _calcSequence(_params, _temp1);

  _temp2 = 0.0;
  _temp2[derivIndex+1]    =  M_PI_2;
  _temp2[-(derivIndex+1)] = -M_PI_2;

  CauchyProduct(_temp1, _temp2, _sequence);

  if (Trace & SequenceDerivs)
    cout << "Sequence Derivatives: " << derivIndex << endl
	 << _sequence << endl;
}

void SLAPTEstimatorBase::
calcSeries2ndDerivs(const CoeffSequence& _params, LaurentSeries& _sequence,
		    unsigned derivIndex1, unsigned derivIndex2)
{
  _calcSequence(_params, _temp1);

  _temp2 = 0.0;
  _temp2[derivIndex1+1]    =  M_PI_2;
  _temp2[-(derivIndex1+1)] = -M_PI_2;

  CauchyProduct(_temp1, _temp2, _temp3);

  _temp2 = 0.0;
  _temp2[derivIndex2+1]    =  M_PI_2;
  _temp2[-(derivIndex2+1)] = -M_PI_2;

  CauchyProduct(_temp3, _temp2, _sequence);

  if (Trace & SequenceDerivs)
    cout << "Derivatives " << derivIndex1 << " " << derivIndex2 << endl
	 << _sequence << endl;
}

void SLAPTEstimatorBase::_resetParams(UnShrt nParams)
{
  if (nParams > NoAllPassParams)
    throw jdimension_error("Cannot reduce the number of SLAPT parameters from %d to %d.",
			   nParams, NoAllPassParams);

  for (UnShrt ipar = nParams; ipar < NoAllPassParams; ipar++)
    p[ipar] = 0.0;
}

void SLAPTEstimatorBase::findRange(double rho, double& min, double& max)
{
  if (rho > 1.0) throw jparameter_error("Incorrect 'rho'.");
  min = -100.0; max = 100.0;
}


// ----- methods for interface class `EstimatorAdapt' -----
//
EstimatorAdapt::
EstimatorAdapt(ParamTreePtr& pt, CodebookSetTrainPtr& cb, unsigned idx, BiasType bt)
  : _paramTree(pt), _nodeIndex(idx), _glist(cb, idx), _biasType(bt)
{
  if (_biasType == NoBias) return;

  UnShrt flen = (_biasType == CepstralFeat) ? _subFeatLen : _featLen;
  UnShrt nsub = (_biasType == CepstralFeat) ?           1 : _nSubFeat;

  _bias.resize(flen, nsub);  _denom.resize(flen, nsub);
}

double EstimatorAdapt::ttlFrames() const 
{
  double ttl = 0.0;
  for (GaussListTrain::ConstIterator itr(_glist); itr.more(); itr++)
    ttl += itr.mix().postProb();

  return ttl;
}

void EstimatorAdapt::_calcBias()
{
  UnShrt bSize = _bias.featLen();

  if (bSize == 0) return;

  for (UnShrt n = 0; n < bSize; n++) {
    _bias(n) = _denom(n) = 0.0;
  }

  for (GaussListTrain::ConstIterator itr(_glist); itr.more(); itr++) {
    const CodebookTrain::GaussDensity& pdf(itr.mix());
    float c_k = pdf.postProb();

    if (c_k == 0.0) continue;

    transform(pdf.origMean(), _transFeat, /* useBiasFlag= */ false);

    for (UnShrt n = 0; n < bSize; n++) {
      float invVar = pdf.invVar(n);
      _bias(n)  += (pdf.sumO(n) - c_k * _transFeat(n)) * invVar;
      _denom(n) += c_k * invVar;
    }
  }

  for (UnShrt n = 0; n < bSize; n++)
    _bias(n) /= _denom(n);
}

double EstimatorAdapt::
_calcMixLhood(const CodebookTrain::GaussDensity& pdf, double& nCounts)
{
  float c_k = pdf.postProb();

  if (c_k == 0.0) return 0.0;

  transform(pdf.origMean(), _transFeat);

  double sum = 0.0;
  for (UnShrt k = 0; k < _featLen; k++) {
    float trnMn = _transFeat(k);
    sum += (pdf.sumO(k) - 0.5 * c_k * trnMn) * trnMn * pdf.invVar(k);
  }

  nCounts += c_k;
  return sum;
}


// ----- methods for class `BLTEstimatorAdapt' -----
//
BLTEstimatorAdapt::
BLTEstimatorAdapt(ParamTreePtr& pt, CodebookSetTrainPtr& cb, unsigned idx, BiasType bt,
		  int trace)
  : TransformBase((cb->ldaFile() == "") ? cb->subFeatLen()    : cb->featLen(),
		  (cb->ldaFile() == "") ? cb->nSubFeat()      : 1,
		  (cb->ldaFile() == "") ? cb->orgSubFeatLen() : cb->orgFeatLen(), trace),
    APTTransformerBase(cb->cepSubFeatLen(), cb->cepNSubFeat(), cb->cepOrgSubFeatLen(),
		       /* noParams= */ 1,
		       trace, cb->ldaFile(), cb->featLen()),
    EstimatorAdapt(pt, cb, idx, bt),
    BLTEstimatorBase(cb->subFeatLen(), cb->nSubFeat())
{ Trace = trace; }

double BLTEstimatorAdapt::calcLogLhood(double mu)
{
  calcTransMatrix(mu);
  _calcBias();

  // loop over all training samples
  double nCounts = 0.0, sum = 0.0;

  for (GaussListTrain::ConstIterator itr(_glist); itr.more(); itr++)
    sum += _calcMixLhood(itr.mix(), nCounts);

  sum /= nCounts;      // normalize by total counts

  if (Trace & Likelihood)
    cout << "    mu = "
	 << setw(12) << setprecision(6) << mu  << " : log-lhood = "
	 << setw(12) << setprecision(6) << sum << endl;

  return sum;
}

EstimatorAdapt& BLTEstimatorAdapt::estimate(unsigned rc)
{
  double b = 0.0;
  double a = b + 0.10, c;
   
  bracket(a, b, c);
  double logLhood = brentsMethod(a, b, c, _alpha);

  // update transformation matrix one last time
  calcTransMatrix(_alpha);  _calcBias();

  // copy parameters back to `par'
  APTParamBasePtr& par = _paramTree->findAPT(rc, RAPT);
  if (par->size() != 1 && par->size() != 0)
    throw jconsistency_error(" No. of all-pass parameters (%d) is incorrect.", par->size());
  par->_conf = _alpha;  par->_bias = _bias;

  if (Trace & Top) {
    cout << "    Best alpha = " << setprecision(4) << setw(10) << _alpha
	 << " : Log-lhood = " << setw(10) << logLhood << endl;

    UnShrt sz = _bias.subFeatLen();
    if (sz != 0) {
      cout << "    Best bias = " << setprecision(4);
      for (UnShrt n = 0; n < sz; n++)
	cout << setw(12) << _bias(n);
      cout << endl;
    }
  }

  return *this;
}


// ----- methods for class `APTEstimatorAdaptBase' -----
//
APTEstimatorAdaptBase::
APTEstimatorAdaptBase(ParamTreePtr& pt, CodebookSetTrainPtr& cb, unsigned idx, BiasType bt,
		      int trace)
  : EstimatorAdapt(pt, cb, idx, bt)
{
  paramDerivs.resize(_featLen, _nSubFeat);

  _dmu_daj = new NaturalVector[NoAllPassParams];
  for (UnShrt i = 0; i < NoAllPassParams; i++)
    _dmu_daj[i].resize(_featLen, _nSubFeat);

  Trace = trace;

  if (_biasType == NoBias) return;

  UnShrt flen =
    (_biasType == CepstralFeat) ? _subFeatLen : _featLen;

  _dbh_daj = gsl_matrix_alloc(flen, NoAllPassParams);
  _d2bh_daj_dai = new gsl_matrix*[flen];
  for (UnShrt n = 0; n < flen; n++)
    _d2bh_daj_dai[n] = gsl_matrix_alloc(NoAllPassParams, NoAllPassParams);
}

APTEstimatorAdaptBase::~APTEstimatorAdaptBase()
{
  delete[] _dmu_daj;

  if (_biasType == NoBias) return;

  gsl_matrix_free(_dbh_daj);

  UnShrt flen =
    (_biasType == CepstralFeat) ? _subFeatLen : _featLen;

  for (UnShrt n = 0; n < flen; n++)
    gsl_matrix_free(_d2bh_daj_dai[n]);
  delete[] _d2bh_daj_dai;
}

void APTEstimatorAdaptBase::calcGradient(const CodebookTrain::GaussDensity& pdf)
{
  float c_k = pdf.postProb();

  if (c_k == 0.0) return;

  transform(pdf.origMean(), _transFeat);
  for (UnShrt ipar = 0; ipar < NoAllPassParams; ipar++) {
    xformGradVec(pdf.origMean(), paramDerivs, ipar);

    double sum = 0.0;
    for (UnShrt k = 0; k < _featLen; k++)
      sum += (pdf.sumO(k) - c_k * _transFeat(k)) * paramDerivs(k) * pdf.invVar(k);
    _xi[ipar] += sum;
  }
}

void APTEstimatorAdaptBase::calcHessian(const CodebookTrain::GaussDensity& pdf)
{
  float  c_k = pdf.postProb();

  if (c_k == 0.0) return;

  transform(pdf.origMean(), _transFeat);
  for (UnShrt jpar = 0; jpar < NoAllPassParams; jpar++)
    xformGradVec(pdf.origMean(), _dmu_daj[jpar], jpar);

  for (UnShrt ipar = 0; ipar < NoAllPassParams; ipar++) {
    for (UnShrt jpar = ipar; jpar < NoAllPassParams; jpar++) {
      xformHessVec(pdf.origMean(), paramDerivs, ipar, jpar);

      double sum = 0.0;
      for (UnShrt k = 0; k < _featLen; k++)
	sum += ((pdf.sumO(k) - c_k * _transFeat(k)) * paramDerivs(k)
		- c_k * (_dmu_daj[ipar][k] * _dmu_daj[jpar][k])) * pdf.invVar(k);
      gsl_matrix_set(_hessian, ipar, jpar,
		     gsl_matrix_get(_hessian, ipar, jpar) + sum);
    }
  }
}

void APTEstimatorAdaptBase::_calcBiasGradient()
{
  if (_dbh_daj == NULL) return;

  UnShrt bSize = _bias.featLen();
  for (UnShrt n = 0; n < bSize; n++)
    _denom(n) = 0.0;
  gsl_matrix_set_zero(_dbh_daj);

  for (GaussListTrain::ConstIterator itr(_glist); itr.more(); itr++) {
    const CodebookTrain::GaussDensity& pdf(itr.mix());

    float c_k = pdf.postProb();

    if (c_k == 0.0) continue;

    for (UnShrt n = 0; n < bSize; n++)
      _denom(n) += c_k * pdf.invVar(n);

    for (UnShrt jpar = 0; jpar < NoAllPassParams; jpar++) {
      xformGradVec(pdf.origMean(), paramDerivs, jpar,
		   /* useBiasFlag= */ false);

      for (UnShrt n = 0; n < bSize; n++)
	gsl_matrix_set(_dbh_daj, n, jpar,
		       gsl_matrix_get(_dbh_daj, n, jpar)
		       - c_k * pdf.invVar(n) * paramDerivs(n));
    }
  }

  for (UnShrt jpar = 0; jpar < NoAllPassParams; jpar++)
    for (UnShrt n = 0; n < bSize; n++)
      gsl_matrix_set(_dbh_daj, n, jpar,
		     gsl_matrix_get(_dbh_daj, n, jpar) / _denom(n));
}

void APTEstimatorAdaptBase::_calcBiasHessian()
{
  if (_d2bh_daj_dai == NULL) return;

  UnShrt bSize = _bias.featLen();
  for (UnShrt n = 0; n < bSize; n++) {
    _denom(n) = 0.0;
    gsl_matrix_set_zero(_d2bh_daj_dai[n]);
  }

  for (GaussListTrain::ConstIterator itr(_glist); itr.more(); itr++) {
    const CodebookTrain::GaussDensity& pdf(itr.mix());
    double c_k = pdf.postProb();

    if (c_k == 0.0) continue;

    for (UnShrt n = 0; n < bSize; n++)
      _denom(n) += c_k * pdf.invVar(n);

    for (UnShrt jpar = 0; jpar < NoAllPassParams; jpar++) {
      for (UnShrt ipar = jpar; ipar < NoAllPassParams; ipar++) {
	xformHessVec(pdf.origMean(), paramDerivs, jpar, ipar,
		     /* useBiasFlag= */ false);

	for (UnShrt n = 0; n < bSize; n++)
	  gsl_matrix_set(_d2bh_daj_dai[n], jpar, ipar,
			 gsl_matrix_get(_d2bh_daj_dai[n], jpar, ipar)
			 - c_k * pdf.invVar(n) * paramDerivs(n));
      }
    }
  }

  for (unsigned jpar = 0; jpar < NoAllPassParams; jpar++)
    for (unsigned ipar = jpar; ipar < NoAllPassParams; ipar++)
      for (UnShrt n = 0; n < bSize; n++)
	gsl_matrix_set(_d2bh_daj_dai[n], jpar, ipar,
		       gsl_matrix_get(_d2bh_daj_dai[n], jpar, ipar) / _denom(n));
}

double APTEstimatorAdaptBase::_accumLogLhood(double rho, double& nCounts)
{
  // set series coefficients and calculate transformation matrix
  for (unsigned i = 0; i < NoAllPassParams; i++)
    _p1[i] = p[i] + rho * _xi[i];

  _calcSequence(_p1, _laurentCoeffs);
  _calcTransMatrix(_laurentCoeffs);  _calcBias();

  if (derivFlag == Gradient || derivFlag == Hessian) {
    for (UnShrt i = 0; i < NoAllPassParams; i++)
      _xi[i] = 0.0;
    calcGradientMatrix(_p1);
    _calcBiasGradient();
  }
  if (derivFlag == Hessian) {
    calcHessianMatrix(_p1);
    gsl_matrix_set_zero(_hessian);
    _calcBiasHessian();
  }

  // loop over all training samples
  double sum = 0.0;  nCounts = 0.0;
  for (GaussListTrain::ConstIterator itr(_glist); itr.more(); itr++) {
    const CodebookTrain::GaussDensity& pdf(itr.mix());
    sum += _calcMixLhood(pdf, nCounts);
    if (derivFlag == Gradient || derivFlag == Hessian)
      calcGradient(pdf);
    if (derivFlag == Hessian)
      calcHessian(pdf);
  }

  return sum / nCounts;      // normalize by total counts
}

void APTEstimatorAdaptBase::_dumpDebugInfo()
{
  if ((Trace & TraceGradient)
      && (derivFlag == Gradient || derivFlag == Hessian))
  {
     cout << "Gradient:" << endl;
     for (unsigned ipar = 0; ipar < NoAllPassParams; ipar++)
       cout << setw(14) << setprecision(6) << _xi[ipar];
     cout << endl;
  }
}

EstimatorAdapt& APTEstimatorAdaptBase::estimate(unsigned rc)
{
  APTParamBasePtr& aptParam = _paramTree->findAPT(rc, _type());
  if (aptParam->isZero()) {
    _resetParams();
  } else {
    unsigned pSize = aptParam->size();

    _resetParams(pSize);
    for (unsigned i = 0; i < pSize; i++)
      p[i] = (Cast<APTParamBase>(*aptParam))[i];
  }

  unsigned bSize = aptParam->biasSize();
  if (bSize != 0 && bSize != _bias.featLen())
    throw jdimension_error("Feature lengths (%d vs. %d) do not match.", bSize, _bias.featLen());
  for (UnShrt i = 0; i < bSize; i++)
    _bias(i) = aptParam->bias(i);

  if (Trace & Iterates) {
    cout << "    Initial point: ";
    for (unsigned i = 0; i < NoAllPassParams; i++)
      cout << setw(12) << setprecision(4) << p[i];
    cout << endl;
  }

  double logLhood = optimize();

  _calcTransMatrix(_laurentCoeffs);  _calcBias();

  // copy parameters back to `aptParam'
  aptParam->_conf = p;  aptParam->_bias = _bias;

  if (Trace & Top) {
    cout << "    Optimal params: " << setprecision(4);
    for (unsigned i = 0; i < NoAllPassParams; i++)
      cout << setw(12) << p[i];
    cout << endl
	 << "    Log-lhood = " << setw(10) << logLhood << endl << endl;

    unsigned sz = _bias.subFeatLen();
    if (sz != 0) {
      cout << "    Best bias = " << setprecision(4);
      for (unsigned n = 0; n < sz; n++)
	cout << setw(12) << _bias(n);
      cout << endl;
    }
  }

  return *this;
}


// ----- methods for class `RAPTEstimatorAdapt' -----
//
RAPTEstimatorAdapt::
RAPTEstimatorAdapt(ParamTreePtr& pt, CodebookSetTrainPtr& cb, unsigned idx, UnShrt noParams,
		   BiasType bt, unsigned noItns, int trace)
  : TransformBase((cb->ldaFile() == "") ? cb->subFeatLen()    : cb->featLen(),
		  (cb->ldaFile() == "") ? cb->nSubFeat()      : 1,
		  (cb->ldaFile() == "") ? cb->orgSubFeatLen() : cb->orgFeatLen(), trace),
    APTTransformerBase(cb->cepSubFeatLen(), cb->cepNSubFeat(), cb->cepOrgSubFeatLen(),
		       noParams, trace, cb->ldaFile(), cb->featLen()),
    APTEstimatorBase(noItns),
    APTEstimatorAdaptBase(pt, cb, idx, bt, trace) { }

void RAPTEstimatorAdapt::_writeParm(FILE* fp) {
  RAPTEstimatorBase::_writeParm(fp);
}

void RAPTEstimatorAdapt::_resetParams(UnShrt nParams) {
  RAPTEstimatorBase::_resetParams(nParams);
}

double RAPTEstimatorAdapt::calcLogLhood(double rho)
{
  double nCounts;
  double sum = _accumLogLhood(rho, nCounts) + barrier();
  if (derivFlag == Gradient || derivFlag == Hessian) {
    for (unsigned ipar = 0; ipar < NoAllPassParams; ipar++)
      _xi[ipar] /= nCounts;
    addGradBarrier();
  }

  if (derivFlag == Hessian) {
    for (unsigned jpar = 0; jpar < NoAllPassParams; jpar++)
      for (unsigned lpar = jpar; lpar < NoAllPassParams; lpar++)
	gsl_matrix_set(_hessian, jpar, lpar,
		       gsl_matrix_get(_hessian, jpar, lpar) / nCounts);
    addHessBarrier();
    copyLowerTriangle(_hessian);
    convertHess2Rect();
  }

  if (derivFlag == Gradient || derivFlag == Hessian)
    convertGrad2Rect();

  _dumpDebugInfo();

  if (Trace & Likelihood)
    cout << "    rho = "      << setw(12) << setprecision(6) << rho
	 << " : log-lhood = " << setw(12) << setprecision(6) << sum << endl;

  return sum;
}


// ----- methods for class `SLAPTEstimatorAdapt' -----
//
SLAPTEstimatorAdapt::
SLAPTEstimatorAdapt(ParamTreePtr& pt, CodebookSetTrainPtr& cb, unsigned idx, UnShrt noParams,
		    BiasType bt, unsigned noItns, int trace)
  : TransformBase((cb->ldaFile() == "") ? cb->subFeatLen()    : cb->featLen(),
		  (cb->ldaFile() == "") ? cb->nSubFeat()      : 1,
		  (cb->ldaFile() == "") ? cb->orgSubFeatLen() : cb->orgFeatLen(), trace),
    APTTransformerBase(cb->cepSubFeatLen(), cb->cepNSubFeat(), cb->cepOrgSubFeatLen(),
		       noParams, trace, cb->ldaFile(), cb->featLen()),
    APTEstimatorBase(noItns),
    APTEstimatorAdaptBase(pt, cb, idx, bt, trace) { }

void SLAPTEstimatorAdapt::_writeParm(FILE* fp) {
  SLAPTEstimatorBase::_writeParm(fp);
}

void SLAPTEstimatorAdapt::_resetParams(UnShrt nParams) {
  SLAPTEstimatorBase::_resetParams(nParams);
}

double SLAPTEstimatorAdapt::calcLogLhood(double rho)
{
  double nCounts;
  double sum = _accumLogLhood(rho, nCounts);
  if (derivFlag == Gradient || derivFlag == Hessian)
    for (unsigned ipar = 0; ipar < NoAllPassParams; ipar++)
      _xi[ipar] /= nCounts;

  if (derivFlag == Hessian) {
    for (unsigned jpar = 0; jpar < NoAllPassParams; jpar++)
      for (unsigned lpar = jpar; lpar < NoAllPassParams; lpar++)
	gsl_matrix_set(_hessian, jpar, lpar,
		       gsl_matrix_get(_hessian, jpar, lpar) / nCounts);
    copyLowerTriangle(_hessian);
  }

  _dumpDebugInfo();

  if (Trace & Likelihood)
    cout << "    rho = "      << setw(12) << setprecision(6) << rho
	 << " : log-lhood = " << setw(12) << setprecision(6) << sum << endl;

  return sum;
}


// ----- methods for class `MLLREstimatorAdapt' -----
//
unsigned MLLREstimatorAdapt::nUse         = 0;
gsl_matrix*     MLLREstimatorAdapt::U            = NULL;
gsl_matrix*     MLLREstimatorAdapt::V            = NULL;
gsl_vector*     MLLREstimatorAdapt::singularVals = NULL;
gsl_vector*     MLLREstimatorAdapt::tempVec      = NULL;
gsl_vector*     MLLREstimatorAdapt::_workSpace   = NULL;

MLLREstimatorAdapt::
MLLREstimatorAdapt(ParamTreePtr& pt, CodebookSetTrainPtr& cb,
		   unsigned idx, BiasType bt, int trace)
  : TransformBase(cb->subFeatLen(), cb->nSubFeat(), cb->subFeatLen()),
    EstimatorAdapt(pt, cb, idx, bt),
    MLLRTransformer(cb->subFeatLen(), cb->nSubFeat())
{
  Trace = trace;

  TransformMatrix::_initialize(_featLen, /* nSubFt= */ 1, _featLen);

  Gi = gsl_matrix_alloc(_featLen + 1, _featLen + 1);
  W  = gsl_matrix_alloc(_featLen,     _featLen + 1);
  Z  = gsl_matrix_alloc(_featLen,     _featLen + 1);

  if (nUse++ > 0) return;

  U            = gsl_matrix_alloc(_featLen + 1, _featLen + 1);
  V            = gsl_matrix_alloc(_featLen + 1, _featLen + 1);

  singularVals = gsl_vector_alloc(_featLen + 1);
  tempVec      = gsl_vector_alloc(_featLen + 1);
  _workSpace   = gsl_vector_alloc(_featLen + 1);
}

MLLREstimatorAdapt::~MLLREstimatorAdapt()
{
  gsl_matrix_free(Gi);  gsl_matrix_free(Z);  gsl_matrix_free(W);

  if (--nUse > 0) return;

  gsl_matrix_free(U);             U = NULL;
  gsl_matrix_free(V);             V = NULL;

  gsl_vector_free(singularVals);  singularVals = NULL;
  gsl_vector_free(tempVec);       tempVec      = NULL;
  gsl_vector_free(_workSpace);    _workSpace   = NULL;
}

// assume a full-feature bias for the time being
void MLLREstimatorAdapt::_calcZ()
{
  gsl_matrix_set_zero(Z);

  for (GaussListTrain::ConstIterator itr(_glist); itr.more(); itr++) {
    const CodebookTrain::GaussDensity mix(itr.mix());
    const NaturalVector origMean(mix.origMean());
    float ck = mix.postProb();

    if (ck == 0.0) continue;

    for (UnShrt m = 0; m < _featLen; m++) {
      float term = mix.sumO(m) * mix.invVar(m);
      for (UnShrt n = 0; n <= _featLen; n++) {
	float mn = (n == _featLen) ? 1.0 : origMean[n];
	gsl_matrix_set(Z, m, n,
		       gsl_matrix_get(Z, m, n) + term * mn);
      }
    }
  }
}

void MLLREstimatorAdapt::_copyLowerTriangle()
{
  for (UnShrt m = 0; m <= _featLen; m++)
    for (UnShrt n = m+1; n <= _featLen; n++)
      gsl_matrix_set(Gi, n, m, gsl_matrix_get(Gi, m, n));
}

void MLLREstimatorAdapt::_calcGi(UnShrt dim)
{
  gsl_matrix_set_zero(Gi);

  for (GaussListTrain::ConstIterator itr(_glist); itr.more(); itr++) {
    const CodebookTrain::GaussDensity mix(itr.mix());
    float ck = mix.postProb();

    if (ck == 0.0) continue;

    const NaturalVector origMean(mix.origMean());
    float term = ck * mix.invVar(dim);

    for (UnShrt m = 0; m <= _featLen; m++) {
      float mn_m = (m == _featLen) ? 1.0 : origMean[m];
      for (UnShrt n = m; n <= _featLen; n++) {
	float mn_n = (n == _featLen) ? 1.0 : origMean[n];
	gsl_matrix_set(Gi, m, n,
		       gsl_matrix_get(Gi, m, n) + term * mn_m * mn_n);
      }
    }
  }
  _copyLowerTriangle();
}


const double MLLREstimatorAdapt::MaxSingularValueRatio = 1.0e-07;

EstimatorAdapt& MLLREstimatorAdapt::estimate(unsigned rc)
{
  gsl_matrix* mat = (gsl_matrix*) matrix();
  gsl_vector* vec = (gsl_vector*) offset();

  _calcZ();
  for (UnShrt dim = 0; dim < _featLen; dim++) {
    _calcGi(dim);
    gsl_vector_view       wi = gsl_matrix_row(W, dim);
    gsl_vector_const_view zi = gsl_matrix_const_row(Z, dim);

    // use SVD to solve for new mean
    gsl_linalg_SV_decomp(Gi, V, singularVals, _workSpace);
    gsl_blas_dgemv(CblasTrans, 1.0, Gi, &(zi.vector), 0.0, tempVec);
    double largestValue = gsl_vector_get(singularVals, 0);
    for (UnShrt n = 0; n <= _featLen; n++) {
      if ((gsl_vector_get(singularVals, n) / largestValue) >= MaxSingularValueRatio)
	gsl_vector_set(tempVec, n,
		       gsl_vector_get(tempVec, n) / gsl_vector_get(singularVals, n));
      else
	gsl_vector_set(tempVec, n, 0.0);
    }
    gsl_blas_dgemv(CblasNoTrans, 1.0, V, tempVec, 0.0, &(wi.vector));

    // copy over new parameters values
    gsl_vector_set(vec, dim, gsl_vector_get(&(wi.vector), _featLen));
    for (UnShrt n = 0; n < _featLen; n++)
      gsl_matrix_set(mat, dim, n, gsl_vector_get(&(wi.vector), n));
  }

  MLLRParamPtr& par(_paramTree->findMLLR(rc));
  Cast<MLLRParam>(*par) = W;

  MLLRParamPtr& parNode(_paramTree->findMLLR(_nodeIndex));
  Cast<MLLRParam>(*parNode) = W;

  return *this;
}

void MLLREstimatorAdapt::_writeParm(FILE* fp)
{
  throw j_error("Not implemented as yet.");
}


// ----- methods for class `STCEstimator' -----
//
STCEstimator::
STCEstimator(VectorFloatFeatureStreamPtr& src, ParamTreePtr& pt, DistribSetTrainPtr& dss,
	     unsigned idx, BiasType bt, bool cascade, bool mmiEstimation, int trace, const String& nm)
  : TransformBase(dss->cbs()->subFeatLen(), dss->cbs()->nSubFeat()),
    EstimatorAdapt(pt, dss->cbs(), idx, bt),
    STCTransformer(src, dss->cbs()->subFeatLen(), dss->cbs()->nSubFeat(), nm),
    _cascade(cascade), _mmiEstimation(mmiEstimation),
    _cofactor(gsl_vector_alloc(_subFeatLen)),
    _scratchMatrix(gsl_matrix_alloc(_subFeatLen, _subFeatLen)),
    _cascadeMatrix(gsl_matrix_alloc(_subFeatLen, _subFeatLen)),
    _scratchVector(gsl_vector_alloc(_subFeatLen)),
    _stcInv(gsl_matrix_alloc(_subFeatLen, _subFeatLen)),
    _transMatrix(gsl_matrix_float_alloc(_subFeatLen, _subFeatLen)),
    _permutation(gsl_permutation_alloc(_subFeatLen)),
    _GiInv(new gsl_matrix*[_subFeatLen]),
    _transFeature(gsl_vector_float_alloc(_featLen)),
    _dss(dss), _accu(new Accu(dss->cbs()->subFeatLen(), dss->cbs()->nSubFeat()))
{
  // _initialize(_subFeatLen, _nSubFeat, _orgSubFeatLen);
  _identity();

  for (unsigned dimX = 0; dimX < _subFeatLen; dimX++)
    _GiInv[dimX] = gsl_matrix_alloc(_subFeatLen, _subFeatLen);

  unsigned maxRefN = 0;
  for (DistribSetTrain::Iterator itr(_dss); itr.more(); itr++)
    if (itr.dst()->cbk()->refN() > maxRefN)
      maxRefN = itr.dst()->cbk()->refN();
  
  _addCount = new float[maxRefN];
}

STCEstimator::~STCEstimator()
{
  gsl_vector_free(_cofactor);
  gsl_matrix_free(_scratchMatrix);
  gsl_vector_free(_scratchVector);
  gsl_matrix_free(_stcInv);
  gsl_matrix_float_free(_transMatrix);
  gsl_permutation_free(_permutation);

  for (unsigned dimX = 0; dimX < _subFeatLen; dimX++)
    gsl_matrix_free(_GiInv[dimX]);
  delete[] _GiInv;

  delete[] _addCount;
  gsl_vector_float_free(_transFeature);
}

void STCEstimator::_identity()
{
  gsl_matrix* mat = (gsl_matrix*) matrix();
  gsl_matrix_set_zero(mat);  gsl_matrix_set_zero(_stcInv);

  for (unsigned i = 0; i < _subFeatLen; i++) {
    gsl_matrix_set(mat, i, i, 1.0);
    gsl_matrix_set(_stcInv, i, i, 1.0);
  }
}

void STCEstimator::accumulate(DistribPathPtr& path, float factor)
{
  double   totalPr = 0.0;
  unsigned frameX  = 0;
  for (DistribPath::Iterator itr(path); itr.more(); itr++) {
    const DistribBasicPtr& ds(_dss->find(itr.name()));
    const CodebookBasicPtr& cb(ds->cbk());

    totalPr += cb->_scoreOpt(frameX, ds->val(), _addCount);

    const gsl_vector_float* block = _src->next(frameX);
    if (_cascade) {
      _transform(block, _transFeature);  block = _transFeature;
    }

    _accu->accumulateW1(cb, block,   _addCount, factor);
    _accu->accumulateW2(cb, _stcInv, _addCount, factor);
    frameX++;
  }
  if (factor > 0.0)
    _accu->increment(totalPr, frameX);
  else
    _accu->increment(-totalPr);

  cout << "STCEstimator: accumulated " << frameX;
  if (factor > 0.0)
    cout << " numerator frames." << endl;
  else
    cout << " denominator frames." << endl;
}

EstimatorAdapt& STCEstimator::estimate(unsigned rc, double minE, double multE)
{
  if (_cascade) {

    gsl_matrix_memcpy(_cascadeMatrix, matrix());
    _update(_cascadeMatrix, minE, multE);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, _cascadeMatrix, matrix(), 0.0, _scratchMatrix);
    gsl_matrix_memcpy((gsl_matrix*) matrix(), _scratchMatrix);

    int signum;
    gsl_linalg_LU_decomp(_scratchMatrix, _permutation, &signum);
    gsl_linalg_LU_invert(_scratchMatrix, _permutation, _stcInv);

  } else {

    _update((gsl_matrix*) matrix(), minE, multE);

  }

  return *this;
}

const gsl_matrix_float* STCEstimator::transMatrix()
{
  for (unsigned i = 0; i < _subFeatLen; i++)
    for (unsigned j = 0; j < _subFeatLen; j++)
      gsl_matrix_float_set(_transMatrix, i, j, gsl_matrix_get(matrix(), i, j));

  return _transMatrix;
}

void STCEstimator::save(const String& fileName, const gsl_matrix_float* trans)
{
  gsl_matrix_float* scratchMatrix = gsl_matrix_float_alloc(_subFeatLen, _subFeatLen);

  for (unsigned i = 0; i < _subFeatLen; i++)
    for (unsigned j = 0; j < _subFeatLen; j++)
      gsl_matrix_float_set(_transMatrix, i, j, gsl_matrix_get(matrix(), i, j));

  if (trans == NULL)
    gsl_matrix_float_memcpy(scratchMatrix, _transMatrix);
  else
    gsl_blas_sgemm(CblasNoTrans, CblasNoTrans, 1.0, _transMatrix, trans,
		   0.0, scratchMatrix);

  FILE* fp = fileOpen(fileName, "w");
  gsl_matrix_float_fwrite(fp, scratchMatrix);
  fileClose(fileName, fp);

  gsl_matrix_float_free(scratchMatrix);
}

void STCEstimator::load(const String& fileName)
{
  STCTransformer::load(fileName);

  gsl_matrix_memcpy(_scratchMatrix, matrix());
  int signum;
  gsl_linalg_LU_decomp(_scratchMatrix, _permutation, &signum);
  gsl_linalg_LU_invert(_scratchMatrix, _permutation, _stcInv);
}

void STCEstimator::zeroAccu()
{
  _accu->zero();
}

void STCEstimator::saveAccu(FILE* fp) const
{
  _accu->save(fp);
}

void STCEstimator::loadAccu(FILE* fp)
{
  _accu->load(fp);
}

void STCEstimator::saveAccu(const String& fileName) const
{
  FILE* fp = fileOpen(fileName, "w");
  saveAccu(fp);
  fileClose(fileName, fp);
}

void STCEstimator::loadAccu(const String& fileName)
{
  FILE* fp = fileOpen(fileName, "r");
  loadAccu(fp);
  fileClose(fileName, fp);
}

void STCEstimator::_update(gsl_matrix* stcMatrix, double minE, double multE)
{
  static unsigned MaxItns = 10;

  _accu->makePosDef(minE, multE);

  /*
  cout << "Initial STC Transformation Matrix:" << endl;
  gsl_matrix_fprintf(stdout, stcMatrix, "%10.4");
  */

  cout << "Using E = " << _accu->E() << endl;

  for (unsigned dimX = 0; dimX < _subFeatLen; dimX++) {
    for (unsigned idim = 0; idim < _subFeatLen; idim++)
      for (unsigned jdim = 0; jdim < _subFeatLen; jdim++)
	gsl_matrix_set(_scratchMatrix, idim, jdim,
		       (gsl_matrix_get(_accu->sumOsq(dimX), idim, jdim)
			+ _accu->E() * gsl_matrix_get(_accu->regSq(dimX), idim, jdim)));

    int signum;
    gsl_linalg_LU_decomp(_scratchMatrix, _permutation, &signum);
    gsl_linalg_LU_invert(_scratchMatrix, _permutation, _GiInv[dimX]);
  }

  double auxVal = _accu->calcAuxFunc(stcMatrix, _mmiEstimation);

  cout << "Initial auxiliary function value = " << auxVal << endl;

  for (unsigned itn = 0; itn < MaxItns; itn++) {
    for (unsigned dimX = 0; dimX < _subFeatLen; dimX++) {

      // calculate the row 'dimX' of the cofactor matrix
      gsl_matrix_memcpy(_scratchMatrix, stcMatrix);
      int signum;
      gsl_linalg_LU_decomp(_scratchMatrix, _permutation, &signum);
      gsl_linalg_LU_invert(_scratchMatrix, _permutation, _stcInv);
      double determinant = gsl_linalg_LU_det(_scratchMatrix, signum);
      for (unsigned idim = 0; idim < _subFeatLen; idim++)
	gsl_vector_set(_cofactor, idim, determinant * gsl_matrix_get(_stcInv, idim, dimX));

      // calculate the "direction" of the udpated row
      gsl_blas_dgemv(CblasTrans, 1.0, _GiInv[dimX], _cofactor, 0.0, _scratchVector);

      // calculate the scale factor
      double factor;
      gsl_blas_ddot(_scratchVector, _cofactor, &factor);
      if (factor <= 0.0)
	throw jnumeric_error("Inner product %f <= 0.0 for dimension %d.", factor, dimX);

      double gamma = _mmiEstimation ? (_accu->denCount() * _accu->E()) : _accu->count();
      factor = sqrt(gamma / factor);

      // update the row 'dimX' of the STC transformation matrix
      for (unsigned idim = 0; idim < _subFeatLen; idim++)
	gsl_matrix_set(stcMatrix, dimX, idim, factor * gsl_vector_get(_scratchVector, idim));
    }

    auxVal = _accu->calcAuxFunc(stcMatrix, _mmiEstimation);

    cout << "After iteration " << (itn+1) << " auxiliary function value = " << auxVal << endl;
  }

  gsl_matrix_memcpy(_scratchMatrix, stcMatrix);
  int signum;
  gsl_linalg_LU_decomp(_scratchMatrix, _permutation, &signum);
  gsl_linalg_LU_invert(_scratchMatrix, _permutation, _stcInv);

  gsl_linalg_LU_decomp(_scratchMatrix, _permutation, &signum);
  double determinant = gsl_linalg_LU_lndet(_scratchMatrix);

  cout << "Final log-determinant = " << determinant << endl;

  /*
  cout << "Final STC Transformation Matrix:" << endl;
  gsl_matrix_fprintf(stdout, stcMatrix, "%10.4");
  */
}

void STCEstimator::_writeParm(FILE* fp)
{
  throw jconsistency_error("Method not yet finished.");
}


// ----- methods for class `STCEstimator::Accu' -----
//
double STCEstimator::Accu::CountThreshold = 1.0E-02;

STCEstimator::Accu::Accu(unsigned subFeatLen, unsigned nSubFeat)
  : _subFeatLen(subFeatLen), _nSubFeat(nSubFeat),
    _count(0.0), _denCount(0.0), _totalPr(0.0), _totalT(0),
    _obs(new double[_subFeatLen]), _E(0.0),
    _scratchMatrix(gsl_matrix_alloc(_subFeatLen, _subFeatLen)),
    _scratchVector(gsl_vector_alloc(_subFeatLen)),
    _permutation(gsl_permutation_alloc(_subFeatLen)),
    _workSpace(gsl_eigen_symm_alloc(_subFeatLen)),
    _sumOsq(new gsl_matrix*[_subFeatLen]),
    _regSq(new gsl_matrix*[_subFeatLen])
{
  for (unsigned n = 0; n < _subFeatLen; n++) {
    _sumOsq[n] = gsl_matrix_alloc(_subFeatLen, _subFeatLen);
    _regSq[n]  = gsl_matrix_alloc(_subFeatLen, _subFeatLen);
  }

  zero();
}

STCEstimator::Accu::~Accu()
{
  delete[] _obs;

  gsl_matrix_free(_scratchMatrix);
  gsl_vector_free(_scratchVector);
  gsl_permutation_free(_permutation);
  gsl_eigen_symm_free(_workSpace);

  for (unsigned n = 0; n < _subFeatLen; n++) {
    gsl_matrix_free(_sumOsq[n]);
    gsl_matrix_free(_regSq[n]);
  }
  delete[] _sumOsq;  delete[] _regSq;
}

void STCEstimator::Accu::zero()
{
  _count = _denCount = _totalPr = 0.0;  _totalT = 0;

  for (unsigned n = 0; n < _subFeatLen; n++) {
    gsl_matrix_set_zero(_sumOsq[n]);
    gsl_matrix_set_zero(_regSq[n]);
  }
}

void STCEstimator::Accu::save(FILE* fp, unsigned index)
{
  write_int(fp,   _subFeatLen);
  write_int(fp,   _nSubFeat);
  write_float(fp, _count);
  write_float(fp, _denCount);

  write_float(fp, _totalPr);
  write_int(fp,   _totalT);

  for (unsigned n = 0; n < _subFeatLen; n++)
    gsl_matrix_fwrite(fp, sumOsq(n));

  for (unsigned n = 0; n < _subFeatLen; n++)
    gsl_matrix_fwrite(fp, regSq(n));
}

void STCEstimator::Accu::load(FILE* fp, unsigned index)
{
  unsigned sublen = read_int(fp);
  if (_subFeatLen != sublen)
    throw jdimension_error("Sub-feature lengths (%d vs. %d) do not match.", _subFeatLen, sublen);

  unsigned nsub = read_int(fp);
  if (_nSubFeat != nsub)
    throw jdimension_error("Number of sub-features (%d vs. %d) do not match.", _nSubFeat, nsub);

  _count    += read_float(fp);
  _denCount += read_float(fp);

  _totalPr  += read_float(fp);
  _totalT   += read_int(fp);

  gsl_matrix* temp = gsl_matrix_alloc(_subFeatLen, _subFeatLen);
  for (unsigned n = 0; n < _subFeatLen; n++) {
    gsl_matrix_fread(fp, temp);
    gsl_matrix_add(_sumOsq[n], temp);
  }

  for (unsigned n = 0; n < _subFeatLen; n++) {
    gsl_matrix_fread(fp, temp);
    gsl_matrix_add(_regSq[n], temp);
  }

  gsl_matrix_free(temp);
}

// accumulate 'W^(1)'
void STCEstimator::Accu::accumulateW1(const CodebookBasicPtr& cb, const gsl_vector_float* pattern,
				      const float* addCount, float factor)
{
  unsigned refN = cb->refN();

  for (unsigned refX = 0; refX < refN; refX++) {
    float count = addCount[refX];

    if (count < CountThreshold) continue;

    count *= factor;

    if ( count > 0.0 )
      _count    += count;
    else if ( count < 0.0 )
      _denCount -= count;

    for (unsigned iblk = 0; iblk < _nSubFeat; iblk++) {
      unsigned offset = iblk * _subFeatLen;

      // accumulate 'W^(1)'
      for (unsigned dimX = 0; dimX < _subFeatLen; dimX++)
	_obs[dimX] = gsl_vector_float_get(pattern, offset+dimX) - cb->mean(refX, offset+dimX);

      for (unsigned dimX = 0; dimX < _subFeatLen; dimX++) {
	float wgt = cb->invCov(refX, offset+dimX) * count;
	gsl_matrix*  cov = _sumOsq[dimX];

	for (unsigned i = 0; i < _subFeatLen; i++) {
	  for (unsigned j = 0; j <= i; j++) {
	    double outer = wgt * _obs[i] * _obs[j];
	    gsl_matrix_set(cov, i, j, gsl_matrix_get(cov, i, j) + outer);
	  }
	}
      }
    }
  }
}

// accumulate 'W^(2)'
void STCEstimator::Accu::accumulateW2(const CodebookBasicPtr& cb, const gsl_matrix* invMatrix,
				      const float* addCount, float factor)
{
  if (factor > 0.0) return;

  unsigned refN = cb->refN();

  for (unsigned refX = 0; refX < refN; refX++) {
    float count = addCount[refX];

    if (count < CountThreshold) continue;

    count *= -factor;

    for (unsigned iblk = 0; iblk < _nSubFeat; iblk++) {
      unsigned offset = iblk * _subFeatLen;

      // calculate sum of outer products over columns of 'P'
      gsl_matrix_set_zero(_scratchMatrix);
      for (unsigned icol = 0; icol < _subFeatLen; icol++) {
	for (unsigned j = 0; j < _subFeatLen; j++)
	  _obs[j] = gsl_matrix_get(invMatrix, j, icol);

	float wgt = count / cb->invCov(refX, offset+icol);
	for (unsigned i = 0; i < _subFeatLen; i++) {
	  for (unsigned j = 0; j <= i; j++) {
	    double outer = wgt * _obs[i] * _obs[j];
	    gsl_matrix_set(_scratchMatrix, i, j, gsl_matrix_get(_scratchMatrix, i, j) + outer);
	  }
	}
      }

      // add sum of outer products to 'W^(2)'
      for (unsigned dimX = 0; dimX < _subFeatLen; dimX++) {
	gsl_matrix*  cov = _regSq[dimX];
	float wgt = cb->invCov(refX, offset+dimX);
	for (unsigned i = 0; i < _subFeatLen; i++)
	  for (unsigned j = 0; j <= i; j++)
	    gsl_matrix_set(cov, i, j, gsl_matrix_get(cov, i, j) + wgt * gsl_matrix_get(_scratchMatrix, i, j));
      }
    }
  }
}

double STCEstimator::Accu::logLhood()
{
  return (_totalPr / _totalT);
}

void STCEstimator::Accu::copyUpperTriangle(gsl_matrix* cov)
{
  for (unsigned i = 0; i < _subFeatLen; i++)
    for (unsigned j = i+1; j < _subFeatLen; j++)
      gsl_matrix_set(cov, i, j, gsl_matrix_get(cov, j, i));
}

double STCEstimator::Accu::calcAuxFunc(const gsl_matrix* stcMatrix, bool mmiEstimation)
{
  int signum;
  gsl_matrix_memcpy(_scratchMatrix, stcMatrix);
  gsl_linalg_LU_decomp(_scratchMatrix, _permutation, &signum);
  double gamma = mmiEstimation ? (_denCount * _E) : _count;
  double auxFunc = 2.0 * gamma *
    _nSubFeat * gsl_linalg_LU_lndet(_scratchMatrix);

  for (unsigned dimX = 0; dimX < _subFeatLen; dimX++) {
    for (unsigned idim = 0; idim < _subFeatLen; idim++) {
      double sum = 0.0;
      for (unsigned jdim = 0; jdim < _subFeatLen; jdim++)
	sum += gsl_matrix_get(stcMatrix, dimX, jdim) *
	  gsl_matrix_get(_sumOsq[dimX], idim, jdim) +
	  _E * gsl_matrix_get(_regSq[dimX], idim, jdim);
      auxFunc -= gsl_matrix_get(stcMatrix, dimX, idim) * sum;
    }
  }

  return auxFunc;
}

// makePosDef: start with E = minE and set E *= 2 until all W_j are p.d.
void STCEstimator::Accu::makePosDef(double minE, double multE)
{
  _E = minE;
  bool flag;

  do {
    for (unsigned dimX = 0; dimX < _subFeatLen; dimX++) {
      for (unsigned idim = 0; idim < _subFeatLen; idim++)
	for (unsigned jdim = 0; jdim < _subFeatLen; jdim++)
	  gsl_matrix_set(_scratchMatrix, idim, jdim,
	    (gsl_matrix_get(sumOsq(dimX), idim, jdim) + _E * gsl_matrix_get(regSq(dimX), idim, jdim)));

      flag = true;
      gsl_eigen_symm(_scratchMatrix, _scratchVector, _workSpace);
      for (unsigned idim = 0; idim < _subFeatLen; idim++)
	if (gsl_vector_get(_scratchVector, idim) <= 0.0) { flag = false;  break; }

      if (flag == false) { _E *= 2.0;  break; }
    }
  } while (flag == false);

  _E *= multE;
}


// ----- methods for class `LDAEstimator' -----
//
LDAEstimator::
LDAEstimator(VectorFloatFeatureStreamPtr& src, ParamTreePtr& pt,
	     DistribSetTrainPtr& dss, const CodebookBasicPtr& globalCodebook,
	     unsigned idx, BiasType bt, int trace)
  : TransformBase(dss->cbs()->subFeatLen(), dss->cbs()->nSubFeat()),
    EstimatorAdapt(pt, dss->cbs(), idx, bt),
    LDATransformer(src, dss->cbs()->orgSubFeatLen(), dss->cbs()->nSubFeat()),
    _globalCodebook(globalCodebook),
    _ldaMatrix(gsl_matrix_alloc(_subFeatLen, _subFeatLen)),
    _scatterMatrix(gsl_matrix_alloc(_subFeatLen, _subFeatLen)),
    _scratchMatrix(gsl_matrix_alloc(_subFeatLen, _subFeatLen)),
    _eigenValue(gsl_vector_alloc(_subFeatLen)),
#if 1
    _whitening(gsl_matrix_alloc(_subFeatLen, _subFeatLen)),
#else
    _squareRootB(gsl_matrix_alloc(_subFeatLen, _subFeatLen)),
#endif
    _eigenVector(gsl_matrix_alloc(_subFeatLen, _subFeatLen)),
    _workSpace(gsl_eigen_symmv_alloc(_subFeatLen)),
    _accu(new Accu(dss->cbs()->subFeatLen(), dss->cbs()->nSubFeat())), _dss(dss) { }

LDAEstimator::~LDAEstimator()
{
  gsl_matrix_free(_ldaMatrix);
  gsl_matrix_free(_scatterMatrix);
  gsl_matrix_free(_scratchMatrix);
  gsl_vector_free(_eigenValue);
#if 1
  gsl_matrix_free(_whitening);
#else
  gsl_matrix_free(_squareRootB);
#endif
  gsl_matrix_free(_eigenVector);
  gsl_eigen_symmv_free(_workSpace);
}

EstimatorAdapt& LDAEstimator::estimate(unsigned rc)
{
  _accu->scaleScatterMatrices();
  for (unsigned iblk = 0; iblk < _nSubFeat; iblk++)
    _update(iblk);

  return *this;
}

double LDAEstimator::accumulate(DistribPathPtr& dspath, float factor)
{
  int    frameX = 0;
  double score  = 0.0;
  for (DistribPath::Iterator itr(dspath); itr.more(); itr++) {

    const DistribBasicPtr ds(_dss->find(itr.name()));

    if (ds.isNull())
      throw jconsistency_error("Distribution is NULL.");

    const CodebookBasicPtr cb(ds->cbk());

    static unsigned maxRefN  = 0;
    static float*   addCount = NULL;

    if (cb->refN() > maxRefN) {
      delete[] addCount;
      addCount = new float[cb->refN()];
      maxRefN  = cb->refN();
    }

    score += cb->_scoreOpt(frameX, ds->val(), addCount);
    const gsl_vector_float* pattern = _src->next(frameX++);

    _accu->accumulate(cb, _globalCodebook, pattern, addCount, factor);
  }

  return score;
}

// simultaneous diagonalization to solve for LDA matrix
void LDAEstimator::_update(unsigned iblk)
{
  gsl_matrix* ldaMatrix = (gsl_matrix*) matrix(iblk);

  // calculate whitening transformation of 'within' class scatter matrix
  gsl_matrix_memcpy(_scratchMatrix, _accu->within(iblk));
  gsl_eigen_symmv(_scratchMatrix, _eigenValue, _whitening, _workSpace);
  for (unsigned i = 0; i < _subFeatLen; i++) {
    double evalue = gsl_vector_get(_eigenValue, i);
    if (evalue <= 0.0) {
      for (unsigned j = 0; j < _subFeatLen; j++)
	gsl_matrix_set(_whitening, j, i, 0.0);
    } else {
      evalue = 1.0 / sqrt(evalue);
      for (unsigned j = 0; j < _subFeatLen; j++)
	gsl_matrix_set(_whitening, j, i, evalue * gsl_matrix_get(_whitening, j, i));
    }
  }

  // transform 'between' class scatter matrix
  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, _whitening, _accu->between(iblk),
		 0.0, _scratchMatrix);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, _scratchMatrix, _whitening,
		 0.0, _scatterMatrix);

  // calculate diagonalizing transform for transformed 'within' class scatter matrix
  gsl_eigen_symmv(_scatterMatrix, _eigenValue, _eigenVector, _workSpace);

  // sort the unordered eigenvalues and eigenvectors
  vector<_SortItem> sortedValues(_subFeatLen);
  for (unsigned i = 0; i < _subFeatLen; i++) {
    sortedValues[i].evalue = gsl_vector_get(_eigenValue, i);
    sortedValues[i].index  = i;
  }
  sort(sortedValues.begin(), sortedValues.end(), _GreaterThan());

  for (unsigned i = 0; i < _subFeatLen; i++)
    for (unsigned j = 0; j < _subFeatLen; j++)
      gsl_matrix_set(_scratchMatrix, j, i, gsl_matrix_get(_eigenVector, j, sortedValues[i].index));

  // calculate final transformation and transpose
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, _whitening, _scratchMatrix, 0.0, ldaMatrix);
  gsl_matrix_transpose(ldaMatrix);
}

void LDAEstimator::zeroAccu()
{
  _accu->zero();
}

void LDAEstimator::saveAccu(FILE* fp) const
{
  _accu->save(fp);
}

void LDAEstimator::loadAccu(FILE* fp)
{
  _accu->load(fp);
}

void LDAEstimator::saveAccu(const String& fileName) const
{
  FILE* fp = fileOpen(fileName, "w");
  saveAccu(fp);
  fileClose(fileName, fp);
}

void LDAEstimator::loadAccu(const String& fileName)
{
  FILE* fp = fileOpen(fileName, "r");
  loadAccu(fp);
  fileClose(fileName, fp);
}

void LDAEstimator::_writeParm(FILE* fp)
{
  throw jconsistency_error("Method not yet finished.");
}


// ----- methods for class `LDAEstimator::Accu' -----
//
double LDAEstimator::Accu::CountThreshold = 1.0E-02;

LDAEstimator::Accu::Accu(unsigned subFeatLen, unsigned nSubFeat)
  : _subFeatLen(subFeatLen), _nSubFeat(nSubFeat),
    _obsW(new double[_subFeatLen]),
    _obsB(new double[_subFeatLen]),
    _obsM(new double[_subFeatLen]),
    _within(new gsl_matrix*[_nSubFeat]),
    _between(new gsl_matrix*[_nSubFeat]),
    _mixture(new gsl_matrix*[_nSubFeat])
{
  printf("Allocating %d matrices (%d x %d)\n",
	 _nSubFeat, _subFeatLen, _subFeatLen);  fflush(stdout);

  for (unsigned iblk = 0; iblk < _nSubFeat; iblk++) {
    _within[iblk]  = gsl_matrix_alloc(_subFeatLen, _subFeatLen);
    _between[iblk] = gsl_matrix_alloc(_subFeatLen, _subFeatLen);
    _mixture[iblk] = gsl_matrix_alloc(_subFeatLen, _subFeatLen);
  }
}

LDAEstimator::Accu::~Accu()
{
  delete[] _obsW;  delete[] _obsB;  delete[] _obsM;

  for (unsigned iblk = 0; iblk < _nSubFeat; iblk++) {
    gsl_matrix_free(_within[iblk]);
    gsl_matrix_free(_between[iblk]);
    gsl_matrix_free(_mixture[iblk]);
  }
  delete[] _within;  delete[] _mixture;
}

void LDAEstimator::Accu::zero()
{
  printf("Clearing %d matrices (%d x %d)\n",
	 _nSubFeat, _subFeatLen, _subFeatLen);  fflush(stdout);

  _count = 0.0;
  for (unsigned iblk = 0; iblk < _nSubFeat; iblk++) {
    gsl_matrix_set_zero(_within[iblk]);
    gsl_matrix_set_zero(_between[iblk]);
    gsl_matrix_set_zero(_mixture[iblk]);
  }
}

void LDAEstimator::Accu::scaleScatterMatrices()
{
  for (unsigned iblk = 0; iblk < _nSubFeat; iblk++) {
    for (unsigned rowX = 0; rowX < _subFeatLen; rowX++) {
      for (unsigned colX = 0; colX < _subFeatLen; colX++) {
	gsl_matrix_set(_within[iblk],  rowX, colX, gsl_matrix_get(_within[iblk],  rowX, colX) / _count);
	gsl_matrix_set(_between[iblk], rowX, colX, gsl_matrix_get(_between[iblk], rowX, colX) / _count);
	gsl_matrix_set(_mixture[iblk], rowX, colX, gsl_matrix_get(_mixture[iblk], rowX, colX) / _count);
      }
    }
  }
}

void LDAEstimator::Accu::save(FILE* fp, unsigned index)
{
  write_int(fp, _subFeatLen);
  write_int(fp, _nSubFeat);

  for (unsigned iblk = 0; iblk < _nSubFeat; iblk++)
    gsl_matrix_fwrite(fp, within(iblk));

  for (unsigned iblk = 0; iblk < _nSubFeat; iblk++)
    gsl_matrix_fwrite(fp, between(iblk));

  for (unsigned iblk = 0; iblk < _nSubFeat; iblk++)
    gsl_matrix_fwrite(fp, mixture(iblk));

  write_float(fp, _count);
}

void LDAEstimator::Accu::load(FILE* fp, unsigned index)
{
  unsigned sublen = read_int(fp);
  if (_subFeatLen != sublen)
    throw jdimension_error("Sub-feature lengths (%d vs. %d) do not match.", _subFeatLen, sublen);

  unsigned nsub = read_int(fp);
  if (_nSubFeat != nsub)
    throw jdimension_error("Number of sub-features (%d vs. %d) do not match.", _nSubFeat, nsub);

  gsl_matrix* temp = gsl_matrix_alloc(_subFeatLen, _subFeatLen);
  for (unsigned iblk = 0; iblk < _nSubFeat; iblk++) {
    gsl_matrix_fread(fp, temp);
    gsl_matrix_add(_within[iblk], temp);
  }

  for (unsigned iblk = 0; iblk < _nSubFeat; iblk++) {
    gsl_matrix_fread(fp, temp);
    gsl_matrix_add(_between[iblk], temp);
  }

  for (unsigned iblk = 0; iblk < _nSubFeat; iblk++) {
    gsl_matrix_fread(fp, temp);
    gsl_matrix_add(_mixture[iblk], temp);
  }

  _count += read_float(fp);

  gsl_matrix_free(temp);
}

void LDAEstimator::Accu::copyUpperTriangle(gsl_matrix* cov)
{
  for (unsigned i = 0; i < _subFeatLen; i++)
    for (unsigned j = i+1; j < _subFeatLen; j++)
      gsl_matrix_set(cov, i, j, gsl_matrix_get(cov, j, i));
}

void LDAEstimator::Accu::accumulate(const CodebookBasicPtr& cb, const CodebookBasicPtr& globalCodebook,
				    const gsl_vector_float* pattern,
				    const float* addCount, float addFactor)
{
  unsigned refN = cb->refN();

  if (refN != globalCodebook->refN())
    throw jdimension_error("Codebook sizes (%d and %d) do not match.", refN, globalCodebook->refN());

  // accumulate within class statistics
  for (unsigned refX = 0; refX < refN; refX++) {
    float count = addCount[refX];

    if (count < CountThreshold) continue;

    count *= addFactor;  _count += count;

    for (unsigned iblk = 0; iblk < _nSubFeat; iblk++) {
      gsl_matrix*     win    = _within[iblk];
      gsl_matrix*     bet    = _between[iblk];
      gsl_matrix*     mix    = _mixture[iblk];
      unsigned offset = iblk * _subFeatLen;

      for (unsigned dimX = 0; dimX < _subFeatLen; dimX++) {
	_obsW[dimX] = gsl_vector_float_get(pattern, offset+dimX) - cb->mean(refX, offset+dimX);
	_obsB[dimX] = cb->mean(refX, offset+dimX) - globalCodebook->mean(/*refX=*/ 0, offset+dimX);
	_obsM[dimX] = gsl_vector_float_get(pattern, offset+dimX) - globalCodebook->mean(/*refX=*/ 0, offset+dimX);
      }

      for (unsigned i = 0; i < _subFeatLen; i++) {
	for (unsigned j = 0; j <= i; j++) {

	  // accumulate 'within' statistics
	  double outer = count * _obsW[i] * _obsW[j];
	  gsl_matrix_set(win, i, j, gsl_matrix_get(win, i, j) + outer);

	  // accumulate 'between' statistics
	  outer = count * _obsB[i] * _obsB[j];
	  gsl_matrix_set(bet, i, j, gsl_matrix_get(bet, i, j) + outer);

	  // accumulate 'mixture' statistics
	  outer = count * _obsM[i] * _obsM[j];
	  gsl_matrix_set(mix, i, j, gsl_matrix_get(mix, i, j) + outer);
	}
      }
    }
  }
}


// ----- methods for class `EstimatorTree' -----
//
float EstimatorTree::EstimateThreshold;

EstimatorTree::
EstimatorTree(CodebookSetTrainPtr& cb, ParamTreePtr& paramTree,
	      int trace, float threshold)
  : TransformerTree(cb, paramTree)
{
  Trace = trace;
  EstimateThreshold = threshold;
}

double EstimatorTree::ttlFrames()
{
  double ttl = 0.0;

  for (_NodeListIter itr = _nodeList.begin(); itr != _nodeList.end(); itr++) {
    NodePtr& node(Cast<NodePtr>((*itr).second));
    ttl += node->ttlFrames();
  }

  return ttl;
}

void estimateAdaptationParameters(EstimatorTreePtr& tree)
{
  if (tree->paramTree()->type() == MLLR)
    tree->paramTree()->clear();

  for (EstimatorTree::LeafIterator itr(tree); itr.more(); itr++) {
    if (Trace & Top)
      cout << endl
	   << "Estimating Parameters for Regression Class "
	   << itr.regClass() << endl
	   << "Total Frames = " << itr.ttlFrames() << endl;

    itr.estimate();
  }
}

void EstimatorTree::writeParams(const String& fileName)
{
   _paramTree->write(fileName);
}

EstimatorTree::RCSet EstimatorTree::leafNodes()
{
  // determine the set of leaf nodes
  RCSet regClassSet;
  for (CodebookSetTrain::GaussianIterator itr(cbs()); itr.more(); itr++) {
    CodebookTrain::GaussDensity mix(itr.mix());
    regClassSet.insert(mix.regClass());
  }

  cout << endl << "Indices of Nodes: ";
  for (RCSetIter itr = regClassSet.begin(); itr != regClassSet.end(); itr++)
    cout << " " << (*itr) << " ";
  cout << endl << endl;

  return regClassSet;
}


// ----- methods for class `EstimatorTree::LeafIterator' -----
//
void EstimatorTree::LeafIterator::estimate()
{
  unsigned rClass;
  NodePtr  treeNode;

  for (rClass = regClass(); rClass > 0; rClass /= 2) {
    treeNode = tree()->node(rClass);
    if (treeNode->ttlFrames() > EstimateThreshold)
      break;
  }

  if (rClass == 0) {
    printf("Node %d has no ancestor with sufficient frame count (%g).",
	   regClass(), EstimateThreshold);  fflush(stdout);
    rClass = 1;
    treeNode = tree()->node(rClass);
  }

  if (regClass() != rClass) {
    if (tree()->paramTree()->type() == MLLR && tree()->paramTree()->hasIndex(rClass)) {
      printf("Copying parameters for node %d from node %d.\n", regClass(), rClass);  fflush(stdout);
      tree()->paramTree()->find(regClass(), /* useAncestor= */ true);
      return;
    }
    printf("Backing off from node %d to node %d.\n", regClass(), rClass);  fflush(stdout);
  }

  treeNode->estimate(regClass());
}

void initializeNaturalIndex(UnShrt nsub = 3)
{
  NaturalIndex::initialize(nsub);
}


// ----- methods for class `EstimatorTreeMLLR' -----
//
EstimatorTreeMLLR::
EstimatorTreeMLLR(CodebookSetTrainPtr& cb, ParamTreePtr& paramTree, int trace, float threshold)
  : EstimatorTree(cb, paramTree, trace, threshold)
{
  RCSet regClassSet = leafNodes();

  // create leaf and internal nodes
  for (RCSetIter itr = regClassSet.begin(); itr != regClassSet.end(); itr++) {
    int rClass = (*itr);

    NodePtr next(new Node(*this, rClass, Leaf, new MLLREstimatorAdapt(paramTree, cb, rClass)));
    _setNode(rClass, next);

    for (int ancIndex = rClass / 2; ancIndex > 0; ancIndex /= 2) {
      if (_nodePresent(ancIndex)) continue;

      NodePtr ancNode(new Node(*this, ancIndex, Internal, new MLLREstimatorAdapt(paramTree, cb, ancIndex)));
      _setNode(ancIndex, ancNode);
    }
  }

  _validateNodeTypes();
}


// ----- methods for class `EstimatorTreeSLAPT' -----
//
EstimatorTreeSLAPT::
EstimatorTreeSLAPT(CodebookSetTrainPtr& cb, ParamTreePtr& paramTree,
		   const String& biasType, UnShrt paramN, unsigned noItns, int trace, float threshold)
  : EstimatorTree(cb, paramTree, trace, threshold)
{
  BiasType bt          = getBiasType(biasType);
  RCSet    regClassSet = leafNodes();

  // create leaf and internal nodes
  for (RCSetIter itr = regClassSet.begin(); itr != regClassSet.end(); itr++) {
    int rClass = (*itr);

    NodePtr next(new Node(*this, rClass, Leaf,
			  new SLAPTEstimatorAdapt(paramTree, cb, rClass, paramN, bt, noItns, trace)));
    _setNode(rClass, next);

    for (int ancIndex = rClass / 2; ancIndex > 0; ancIndex /= 2) {
      if (_nodePresent(ancIndex)) continue;

      NodePtr ancNode(new Node(*this, ancIndex, Internal,
			       new SLAPTEstimatorAdapt(paramTree, cb, ancIndex, paramN, bt, noItns, trace)));
      _setNode(ancIndex, ancNode);
    }
  }

  _validateNodeTypes();
}


// ----- methods for class `EstimatorTreeSTC' -----
//
EstimatorTreeSTC::
EstimatorTreeSTC(VectorFloatFeatureStreamPtr& src, DistribSetTrainPtr& dss, ParamTreePtr& paramTree,
		 const String& biasType, int trace, float threshold)
  : EstimatorTree(dss->cbs(), paramTree, trace, threshold)
{
  BiasType bt          = getBiasType(biasType);
  RCSet    regClassSet = leafNodes();

  // create leaf and internal nodes
  for (RCSetIter itr = regClassSet.begin(); itr != regClassSet.end(); itr++) {
    int rClass = (*itr);

    NodePtr next(new Node(*this, rClass, Leaf,
			  new STCEstimator(src, paramTree, dss, rClass, bt, trace)));
    _setNode(rClass, next);

    for (int ancIndex = rClass / 2; ancIndex > 0; ancIndex /= 2) {
      if (_nodePresent(ancIndex)) continue;

      NodePtr ancNode(new Node(*this, ancIndex, Internal,
			       new STCEstimator(src, paramTree, dss, ancIndex, bt, trace)));
      _setNode(ancIndex, ancNode);
    }
  }

  _validateNodeTypes();
}


// ----- methods for class `EstimatorTreeLDA' -----
//
EstimatorTreeLDA::
EstimatorTreeLDA(VectorFloatFeatureStreamPtr& src, DistribSetTrainPtr& dss, ParamTreePtr& paramTree,
		 const CodebookBasicPtr& globalCodebook, const String& biasType, int trace, float threshold)
  : EstimatorTree(dss->cbs(), paramTree, trace, threshold)
{
  BiasType bt = getBiasType(biasType);

  unsigned rClass = 1;
  NodePtr global(new Node(*this, rClass, Leaf,
			  new LDAEstimator(src, paramTree, dss, globalCodebook, rClass, bt, trace)));
  _setNode(rClass, global);

  _validateNodeTypes();
}
