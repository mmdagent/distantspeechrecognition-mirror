//                              -*- C++ -*-
//
//                          Beamforming Toolkit
//                                 (btk)
//
//  Module:  btk.dereverberation
//  Purpose: Single- and multi-channel dereverberation base on linear
//	     prediction in the subband domain.
//  Author:  John McDonough

#ifndef _dereverberation_h_
#define _dereverberation_h_

#include <stdio.h>
#include <assert.h>

#include <gsl/gsl_block.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_linalg.h>
#include "common/jexception.h"

#include "stream/stream.h"
#include "feature/feature.h"

/*
#include <Eigen/Cholesky>
#include <iostream>
#include <Eigen/Dense>
*/


// ----- definition for class `SingleChannelWPEDereverberationFeature' -----
//
class SingleChannelWPEDereverberationFeature : public VectorComplexFeatureStream {
  typedef vector<gsl_vector_complex*>			_Samples;
  typedef _Samples::iterator				_SamplesIterator;

 public:
  SingleChannelWPEDereverberationFeature(VectorComplexFeatureStreamPtr& samples, unsigned lowerN, unsigned upperN, unsigned iterationsN = 2, double loadDb = -20.0, double bandWidth = 0.0,  double sampleRate = 16000.0,
					 const String& nm = "SingleChannelWPEDereverberationFeature");

  ~SingleChannelWPEDereverberationFeature();

  virtual const gsl_vector_complex* next(int frameX = -5);

  virtual void reset();

  void nextSpeaker();

private:
  static const double					_SubbandFloor;

  void _fillBuffer();
  void _estimateGn();
  void _calculateRr(unsigned subbandX);
  void _calculateThetan();
  void _loadR();
  unsigned _setBandWidthN(double bandWidth, double sampleRate);

  const gsl_vector_complex* _getLags(unsigned subbandX, unsigned sampleX);
  VectorComplexFeatureStreamPtr				_samples;

  const unsigned					_lowerN;
  const unsigned					_upperN;
  const unsigned					_predictionN;
  const unsigned					_iterationsN;
  bool							_firstFrame;
  unsigned						_framesN;
  const double						_loadFactor;
  const unsigned					_lowerBandWidthN;
  const unsigned					_upperBandWidthN;

  _Samples						_yn;
  gsl_matrix*						_thetan;
  gsl_vector_complex**					_gn;
  gsl_matrix_complex*					_R;
  gsl_vector_complex*					_r;
  gsl_vector_complex*					_lagSamples;
};

typedef Inherit<SingleChannelWPEDereverberationFeature, VectorComplexFeatureStreamPtr> SingleChannelWPEDereverberationFeaturePtr;


// ----- definition for class `MultiChannelWPEDereverberation' -----
//
class MultiChannelWPEDereverberation : public Countable {
  typedef vector<VectorComplexFeatureStreamPtr>		_SourceList;
  typedef _SourceList::iterator				_SourceListIterator;

  typedef vector<gsl_vector_complex*>			_FrameBrace;
  typedef _FrameBrace::iterator				_FrameBraceIterator;

  typedef vector<_FrameBrace>				_FrameBraceList;
  typedef _FrameBraceList::iterator			_FrameBraceListIterator;

 public:
  MultiChannelWPEDereverberation(unsigned subbandsN, unsigned channelsN, unsigned lowerN, unsigned upperN, unsigned iterationsN = 2, double loadDb = -20.0, double bandWidth = 0.0,  double sampleRate = 16000.0);

  ~MultiChannelWPEDereverberation();

  void reset();

  unsigned size() const { return _subbandsN; }

  void setInput(VectorComplexFeatureStreamPtr& samples);

  void nextSpeaker();

  const gsl_vector_complex* getOutput(unsigned channelX, int frameX = -5);

private:
  static const double					_SubbandFloor;

  void _fillBuffer();
  void _estimateGn();
  void _calculateRr(unsigned subbandX);
  void _calculateThetan();
  void _loadR();
  unsigned _setBandWidthN(double bandWidth, double sampleRate);

  void _increment() { _frameX++; }
  const gsl_vector_complex* _getLags(unsigned subbandX, unsigned sampleX);

  _SourceList						_sources;
  const unsigned					_subbandsN;
  const unsigned					_channelsN;
  const unsigned					_lowerN;
  const unsigned					_upperN;
  const unsigned					_predictionN;
  const unsigned					_iterationsN;
  const unsigned					_totalPredictionN;

  bool							_firstFrame;
  unsigned						_framesN;
  const double						_loadFactor;
  const unsigned					_lowerBandWidthN;
  const unsigned					_upperBandWidthN;

  _FrameBraceList					_frames;
  gsl_matrix**						_thetan;
  gsl_vector_complex***					_Gn;
  gsl_matrix_complex**					_R;
  gsl_vector_complex**					_r;
  gsl_vector_complex*					_lagSamples;
  gsl_vector_complex**					_output;

  const int						FrameResetX;
  int							_frameX;
};

typedef refcountable_ptr<MultiChannelWPEDereverberation> MultiChannelWPEDereverberationPtr;


// ----- definition for class `MultiChannelWPEDereverberationFeature' -----
//
class MultiChannelWPEDereverberationFeature : public VectorComplexFeatureStream {
public:
  MultiChannelWPEDereverberationFeature(MultiChannelWPEDereverberationPtr& source, unsigned channelX, const String& nm = "MultiChannelWPEDereverberationFeature");

  ~MultiChannelWPEDereverberationFeature();

  virtual const gsl_vector_complex* next(int frameX = -5);

  virtual void reset();

private:
  MultiChannelWPEDereverberationPtr			_source;
  const unsigned					_channelX;
};

typedef Inherit<MultiChannelWPEDereverberationFeature, VectorComplexFeatureStreamPtr> MultiChannelWPEDereverberationFeaturePtr;

#endif // _dereverberation_h_
