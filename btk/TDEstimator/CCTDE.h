//                              -*- C++ -*-
//
//                          Beamforming Toolkit
//                                 (btk)
//
//  Module:  btk.TDEstimator
//  Purpose: 
//  Author:  ABC
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

#ifndef _cctde_h_
#define _cctde_h_

#ifdef BTK_MEMDEBUG
#include "memcheck/memleakdetector.h"
#endif
#include <stdio.h>
#include <assert.h>
#include <float.h>

#include <gsl/gsl_block.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_complex.h>
#include <common/refcount.h>
#include "common/jexception.h"

#include "stream/stream.h"
#include "feature/feature.h"
#include "modulated/modulated.h"
#include "btk.h"

// ----- definition for class `CCTDE' -----
// 
/**
   @class find the time difference which provides the maximum correlation between tow signals.
   @usage
   1. constrct sample feature objects :
       samp1 = 
       samp2 =
   2. construct this object and feed the sample features into it:
   
*/
class CCTDE : public VectorFeatureStream {
public:
  /**
     @brief 
     @param 
     @param 
     @param 
     @param 
   */
  CCTDE( SampleFeaturePtr& samp1, SampleFeaturePtr& samp2, int fftLen=512, unsigned nHeldMaxCC=1, int freqLowerLimit=-1, int freqUpperLimit=-1, const String& nm= "CCTDE" );
  ~CCTDE();

  void setTargetFrequencyRange( int freqLowerLimit, int freqUpperLimit ){
    _freqLowerLimit = freqLowerLimit;
    _freqUpperLimit = freqUpperLimit;
  }
  void  allsamples( int fftLen = -1 );
  virtual const gsl_vector* next(int frameX = -5 );
  virtual const gsl_vector* nextX( unsigned chanX = 0, int frameX = -5);
  virtual void  reset();
  
  const unsigned *getSampleDelays(){return (const unsigned *)_sampleDelays;}
  const double *getCCValues(){return (const double *)_ccValues;}

private:
  const gsl_vector* detectPeaksOfCCFunction( double **samples, size_t stride );
	
protected:
  //typedef list<VectorFloatFeatureStreamPtr>	_ChannelList; 
  typedef list<SampleFeaturePtr>	_ChannelList; 
  typedef _ChannelList::iterator		_ChannelIterator;
  _ChannelList	_channelList;   // must be 2.
  unsigned	*_sampleDelays; // sample delays
  double	*_ccValues;     // cross-correlation (CC) values
  unsigned	_nHeldMaxCC;    // how many CC values are held 
  unsigned	_fftLen;
  int		_sampleRate;
  gsl_vector	*_window;
  int		_freqLowerLimit;
  int		_freqUpperLimit;
  vector<unsigned > _frameCounter;
};

typedef Inherit<CCTDE, VectorFeatureStreamPtr> CCTDEPtr;

#endif

