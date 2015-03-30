//                              -*- C++ -*-
//
//                              Millennium
//                  Automatic Speech Recognition System
//                                 (asr)
//
//  Module:  asr.train
//  Purpose: Speaker adaptation parameter estimation and conventional
//           ML and discriminative HMM training.
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


#ifndef _distribTrain_h_
#define _distribTrain_h_

#include "common/refcount.h"
#include "gaussian/distribBasic.h"
#include "decoder/decoder.h"
#include "train/codebookTrain.h"

/**
* \addtogroup Distribution Distribution
*/
/*@{*/

/**
* \defgroup DistribTrainGroup Distribution Train Group
* This group of classes provides the capability for performing ML estimation of Gaussian mixture weights.
*/
/*@{*/

/**
* \defgroup DistribTrain Distrib Train
*/
/*@{*/

// ----- definition for class `DistribTrain' -----
//
class DistribTrain : public DistribBasic {
 public:
  DistribTrain(const String& nm, CodebookTrainPtr& cb);
  DistribTrain(const DistribTrain& ds);

  virtual ~DistribTrain();

  DistribTrain& operator=(const DistribTrain& ds);

  class Accu;  friend class Accu;

  typedef refcount_ptr<Accu> AccuPtr;

  // accumulate one frame of data
  float accumulate(int frameX, float factor = 1.0);

  // save accumulator
  void  saveAccu(FILE* fp) const;

  // load accumulator
  void  loadAccu(FILE* fp, float addFactor = 1.0);

  // zero accumulator
  void	zeroAccu();

  // update mixture weights with ML criterion
  void	update();

  // update mixture weights with MMI criterion
  void updateMMI(bool merialdo = false);

  // split Gaussian in codebook with highest count
  void split(float minCount = 0.0, float splitFactor = 0.2);

	AccuPtr& accu()       { return _accu; }
  const AccuPtr& accu() const { return _accu; }

        CodebookTrainPtr&  cbk()       { return Cast<CodebookTrainPtr>(_cbk()); }
  const CodebookTrainPtr&  cbk() const { return Cast<CodebookTrainPtr>(_cbk()); }

 private:
  float* 		_mixVal;	// scratch space for Gaussian posterior probabilities
  AccuPtr		_accu;		// accumulator for parameter re-estimation
};

typedef Inherit<DistribTrain, DistribBasicPtr> DistribTrainPtr;


// ----- definition for class `DistribTrain::Accu' -----
//
class DistribTrain::Accu {
 public:
  Accu(UnShrt valN, UnShrt subN = 1);
  ~Accu();

  void accumulate(float factor, float* mixVal);
  void update(float* val, float& count);
  void updateMMI(float* val, float& count, bool merialdo = false);
  void zero();
  bool zeroOccupancy() const;
  void save(FILE* fp) const;
  void load(FILE* fp, float addFactor);
  unsigned size() const;
  double counts() const;
  void add(AccuPtr& ac, double factor = 1.0);

 private:
  static const int MarkerMagic;

  void		_dontRead(FILE* fp, int valN, int subN);
  void		_dumpMarker(FILE* fp) const;
  void		_checkMarker(FILE* fp) const;

  void _updateMMI(float* val, float& count);
  void _updateMMIAlternate(float* val, float& count);

  UnShrt	_subN;		// how many subaccumulators are we using
  UnShrt	_valN;		// number of Gaussians in distribution
  double**	_count;		// a "numerator" count for each distribution value
  double**	_denCount;	// a "denominator" count for each distribution value
  UnShrt	_subX;		// dummy index for sub-accumulator
};


// ----- definition for class `DistibSetTrain' -----
//
class DistribSetTrain : public DistribSetBasic {
  friend class AccMap;
 public:
  DistribSetTrain(CodebookSetTrainPtr& cb, const String& descFile, const String& distFile = "");
  virtual ~DistribSetTrain() { }

  class Iterator;  friend class Iterator;

  	DistribTrainPtr& find(const String& key)       { return Cast<DistribTrainPtr>(_find(key)); }
  const DistribTrainPtr& find(const String& key) const { return Cast<DistribTrainPtr>(_find(key)); }

        DistribTrainPtr& find(unsigned dsX)            { return Cast<DistribTrainPtr>(_find(dsX)); }
  const DistribTrainPtr& find(unsigned dsX)      const { return Cast<DistribTrainPtr>(_find(dsX)); }

  	CodebookSetTrainPtr& cbs()                     { return Cast<CodebookSetTrainPtr>(_cbs()); }
  const CodebookSetTrainPtr& cbs()               const { return Cast<CodebookSetTrainPtr>(_cbs()); }

  // accumulate forward-backward statistics over path
  double accumulate(DistribPathPtr& path, float factor = 1.0);

  // accumulate forward-backward statistics over all edges in lattice
  void accumulateLattice(LatticePtr& lat, float factor = 1.0);

  // save accumulators
  void saveAccus(const String& fileName) const;

  // load accumulators
  void loadAccus(const String& fileName, unsigned nParts = 1, unsigned part = 1);

  // clear all accumulators
  void zeroAccus(int nParts = 1, int part = 1);

  // update mixture weights with ML criterion
  void update(int nParts = 1, int part = 1);

  // update mixture weights with MMI criterion
  void updateMMI(int nParts = 1, int part = 1, bool merialdo = false);

 protected:
  virtual void _addDist(const String& name, const String& cbname);
};

typedef Inherit<DistribSetTrain, DistribSetBasicPtr> DistribSetTrainPtr;


// ----- definition for class `DistibSetTrain::Iterator' -----
//
class DistribSetTrain::Iterator : public DistribSetBasic::Iterator {
 public:
  Iterator(DistribSetTrainPtr dss) : DistribSetBasic::Iterator(dss) { }

  DistribTrainPtr 	dst()       { return Cast<DistribTrainPtr>(_dst()); }
  const DistribTrainPtr dst() const { return Cast<const DistribTrainPtr>(_dst()); }

  inline DistribTrainPtr next();
};

// needed for Python iterator
//
DistribTrainPtr DistribSetTrain::Iterator::next()
{
  if (!more())
    throw jiterator_error("end of distributions!");

  DistribTrainPtr ds(dst());
  operator++(1);
  return ds;
}


// ----- definition of class `AccMap' -----
//
class CodebookSetFastSAT;
class AccMap {
  typedef map<String, long int, less<String> >	_AccumMap;
  typedef _AccumMap::const_iterator		_AccumMapIter;
  typedef _AccumMap::value_type			_ValueType;

 public:
  AccMap(FILE* fp);
  AccMap(const CodebookSetTrain& cb, bool countsOnlyFlag = false);
  AccMap(const CodebookSetFastSAT& cb, bool fastAccusFlag = false, bool countsOnlyFlag = false);
  AccMap(const DistribSetTrain& dss);

  void write(FILE* fp);

  long int state(const String& name);

  static const long int NotPresent;

 protected:
  static const String& endOfAccs() {
    static const String EndOfAccsString("EndOfAccumulators");
    return EndOfAccsString;
  }

  bool _readName(FILE* fp, char* name);
  void _stampEndOfAccs(FILE* fp);

  _AccumMap _stateMap;
};

/*@}*/

/*@}*/

/*@}*/

#endif
