//                              -*- C++ -*-
//
//                              Millennium
//                   Distant Speech Recognition System
//                                 (dsr)
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


#ifndef _distribBasic_h_
#define _distribBasic_h_

#include "gaussian/codebookBasic.h"

/**
* \defgroup Distribution Manipulation of Gaussian mixture weights
* This group of classes provides the capability to manipulate Gaussian mixture weights.
*/
/*@{*/

/**
* \defgroup DistribBasic Classes for calculating likelihoods with Gaussian mixtures.
*/
/*@{*/

// ----- definition for class `Distrib' -----
//
class Distrib {
public:
  virtual ~Distrib() { }

  float score(int frameX) {
    return _scoreFunction(this, frameX);
  }

  const String&	name() const { return _name; }

  inline CodebookBasicPtr cbk();

protected:
  typedef float (*_ScoreFunction)(Distrib* ds, int frameX);

  Distrib(const String& nm)
    : _name(nm), _scoreFunction(NULL) { }

  String					_name;		// name of distribution
  _ScoreFunction				_scoreFunction;
};

typedef refcount_ptr<Distrib> DistribPtr;


// ----- definition for class `DistribBasic' -----
//
class DistribBasic;
typedef Inherit<DistribBasic, DistribPtr> DistribBasicPtr;

class DistribBasic : public Distrib {
 public:
  DistribBasic(const String& nm, CodebookBasicPtr& cb);
  DistribBasic(const DistribBasic& ds);

  virtual ~DistribBasic();

  String puts() const;

  DistribBasic& operator=(const DistribBasic& ds);

  void save(FILE* fp) const;
  void load(FILE* fp, bool readName = false);
  void initializeFromCodebook();

  void write(FILE* fp) const;

  inline LogLhoodIndexPtr logLhood(const gsl_vector* frame) const;

        CodebookBasicPtr& cbk()       { return _cbk(); }
  const CodebookBasicPtr& cbk() const { return _cbk(); }

  	float* val()       { return _val; }
  const float* val() const { return _val; }
  UnShrt valN() const { return _valN; }
  void setWght(unsigned compX, float wgt) { _val[compX] = wgt; }

  void copyWeights(const DistribBasicPtr& ds);

 protected:
        CodebookBasicPtr& _cbk()       { return _codebook; }
  const CodebookBasicPtr& _cbk() const { return _codebook; }

  float& count() { return _count; }

  // private:
  static float _score(Distrib* ds, int frameX)
  {
    DistribBasic* ptr = (DistribBasic*) ds;
    return ptr->cbk()->score(frameX, ptr->val());
  }

  static const unsigned MaxNameLength;
  static const float SmallCount;

  float*					_val;		// vector of distribution values
  UnShrt					_valN;		// number of values in this distribution
  float						_count;		// total counts received in last epoch
  CodebookBasicPtr				_codebook;	// codebook for this distribution
};

LogLhoodIndexPtr DistribBasic::logLhood(const gsl_vector* frame) const
{
  return cbk()->logLhood(frame, _val);
}

CodebookBasicPtr Distrib::cbk() { return ((DistribBasic*)this)->cbk(); }


// ----- definition for class `DistribMultiStream' -----
//
class DistribMultiStream : public Distrib {
public:
  DistribMultiStream(const String& nm, DistribPtr& audioDist, DistribPtr& videoDist, double audioWeight = 0.95, double videoWeight = 0.05);
  virtual ~DistribMultiStream() { }

  double audioWeight() const { return _audioWeight; }
  double videoWeight() const { return _videoWeight; }
  void   setWeights( double audioWeight, double videoWeight ){
    _audioWeight = audioWeight; _videoWeight = videoWeight;
    /*fprintf(stderr,"stream weights %e %e\n", _audioWeight, _videoWeight);*/
  }
  
protected:
  static float _score(Distrib* ds, int frameX)
  {
    DistribMultiStream* ptr = (DistribMultiStream*) ds;
    return ptr->audioWeight() * ptr->_audioDist->score(frameX) + ptr->videoWeight() * ptr->_videoDist->score(frameX);
  }

  DistribPtr&					_audioDist;
  DistribPtr&					_videoDist;
  double                                        _audioWeight;
  double				        _videoWeight;
};

typedef Inherit<DistribMultiStream, DistribPtr> DistribMultiStreamPtr;


// ----- definition for container class `DistibSet' -----
//
class DistribSet {
 protected:
  typedef List<DistribPtr>			_DistribList;
  typedef _DistribList::Iterator		_Iterator;
  typedef _DistribList::ConstIterator		_ConstIterator;

 public:
  virtual ~DistribSet() { }

  class Iterator;  	friend class Iterator;
  class ConstIterator;  friend class ConstIterator;

  virtual void resetCache() = 0;
  virtual void resetFeature() = 0;

  unsigned ndists() const { return _list.size(); }

  unsigned index(const String& key) const { return _list.index(key); }

        DistribPtr& find(const String& key)        { return _find(key); }
  const DistribPtr& find(const String& key)  const { return _find(key); }

        DistribPtr& find(unsigned dsX)             { return _find(dsX); }
  const DistribPtr& find(unsigned dsX)       const { return _find(dsX); }

 protected:
  DistribSet(): _list("Distribution Set") { }

        DistribPtr&	_find(const String& key)       { return _list[key]; }
  const DistribPtr&	_find(const String& key) const { return _list[key]; }

        DistribPtr& 	_find(unsigned dsX)       { return _list[dsX]; }
  const DistribPtr&	_find(unsigned dsX) const { return _list[dsX]; }

 protected:
  _DistribList					_list;
};

typedef refcount_ptr<DistribSet> DistribSetPtr;


// ----- definition for class `DistibSet::Iterator' -----
//
class DistribSet::Iterator {
  friend class DistribSet;
 public:
  Iterator(DistribSetPtr& dss)
    : _iter(dss->_list) { }

  void operator++(int) { _iter++; }
  bool more() const { return _iter.more(); }

  inline DistribPtr next();

        DistribPtr&	dst()       { return _dst(); }
  const DistribPtr&	dst() const { return _dst(); }

 protected:
        DistribPtr&	_dst()       { return *_iter; }
  const DistribPtr&	_dst() const { return *_iter; }

 private:
  _Iterator		_iter;
};

// needed for Python iterator
//
DistribPtr DistribSet::Iterator::next()
{
  if (!more())
    throw jiterator_error("end of distributions!");

  DistribPtr ds(dst());
  operator++(1);
  return ds;
}


// ----- definition for class `DistibSet::ConstIterator' -----
//
class DistribSet::ConstIterator {
  friend class DistribSet;
 public:
  ConstIterator(const DistribSetPtr& dss)
    : _iter(dss->_list) { }

  void operator++(int) { _iter++; }
  bool more() const { return _iter.more(); }

  inline const DistribPtr next();

  const DistribPtr&	dst() const { return _dst(); }

 protected:
  const DistribPtr&	_dst() const { return *_iter; }

 private:
  _ConstIterator	_iter;
};

// needed for Python iterator
//
const DistribPtr DistribSet::ConstIterator::next()
{
  if (more()) {
    operator++(1);
    return dst();
  } else {
    throw jiterator_error("end of codebook!");
  }
}

/**
* \defgroup DistibSetBasic Basic representation and manipulation of Gaussian mixture weights.
*/
/*@{*/

// ----- definition for container class `DistibSetBasic' -----
//
class DistribSetBasic : public DistribSet {
 public:
  DistribSetBasic(CodebookSetBasicPtr& cbs, const String& descFile = "", const String& distFile = "");
  virtual ~DistribSetBasic();

  class Iterator;  	friend class Iterator;
  class ConstIterator;  friend class ConstIterator;

  void load(const String& distFile);
  void save(const String& distFile) const;

  void write(const String& fileName, const String& time = "") const;

  void resetCache() { _codebookSet->resetCache(); }
  void resetFeature() { _codebookSet->resetFeature(); }

        DistribBasicPtr& find(const String& key)        { return Cast<DistribBasicPtr>(_find(key)); }
  const DistribBasicPtr& find(const String& key)  const { return Cast<DistribBasicPtr>(_find(key)); }

        DistribBasicPtr& find(unsigned dsX)             { return Cast<DistribBasicPtr>(_find(dsX)); }
  const DistribBasicPtr& find(unsigned dsX)       const { return Cast<DistribBasicPtr>(_find(dsX)); }

        CodebookSetBasicPtr& cbs()                      { return _cbs(); }
  const CodebookSetBasicPtr& cbs()                const { return _cbs(); }

  void __add(const String& s);

 protected:
  virtual void _addDist(const String& name, const String& cbname);

        CodebookSetBasicPtr&	_cbs()       { return _codebookSet; }
  const CodebookSetBasicPtr&	_cbs() const { return _codebookSet; }

  CodebookSetBasicPtr	_codebookSet;

  void _initializeFromCodebooks();

 private:
  static const unsigned MaxNameLength;
  static const int      Magic;
};

typedef Inherit<DistribSetBasic, DistribSetPtr> DistribSetBasicPtr;


// ----- definition for class `DistibSetBasic::Iterator' -----
//
class DistribSetBasic::Iterator : public DistribSet::Iterator {
  friend class DistribSetBasic;
 public:
  Iterator(DistribSetBasicPtr& dss)
    : DistribSet::Iterator(dss) { }

  inline DistribBasicPtr next();

        DistribBasicPtr& dst()       { return Cast<DistribBasicPtr>(_dst()); }
  const DistribBasicPtr& dst() const { return Cast<DistribBasicPtr>(_dst()); }
};

// needed for Python iterator
//
DistribBasicPtr DistribSetBasic::Iterator::next()
{
  if (!more())
    throw jiterator_error("end of distributions!");

  DistribBasicPtr ds(dst());
  operator++(1);
  return ds;
}


// ----- definition for class `DistibSetBasic::ConstIterator' -----
//
class DistribSetBasic::ConstIterator : public DistribSet::ConstIterator {
  friend class DistribSetBasic;
 public:
  ConstIterator(const DistribSetBasicPtr& dss)
    : DistribSet::ConstIterator(dss) { }

  inline const DistribBasicPtr next();

  const DistribBasicPtr& dst() const { return Cast<const DistribBasicPtr>(_dst()); }
};

// needed for Python iterator
//
const DistribBasicPtr DistribSetBasic::ConstIterator::next()
{
  if (more()) {
    operator++(1);
    return dst();
  } else {
    throw jiterator_error("end of codebook!");
  }
}

/*@}*/

/**
* \defgroup DistribMap Mapping across distribution sets for multiple stream decoding
*/
/*@{*/

// ----- definition for container class `DistribMap' -----
//
class DistribMap : public Countable {
  typedef map<String, String>			_Mapping;
  typedef _Mapping::iterator			_MappingIterator;
  typedef _Mapping::const_iterator		_MappingConstIterator;
public:
  DistribMap() { }
  ~DistribMap() { }

  const String& mapped(const String& name) const;
  void add(const String& fromName, const String& toName) { _mapping[fromName] = toName; }

  void read(const String& fileName);
  void write(const String& fileName) const;

private:
  _Mapping					_mapping;
};

typedef refcount_ptr<DistribMap> DistribMapPtr;

/*@}*/

/**
* \defgroup DistribSetMultiStream Multiple stream distribution sets
*/
/*@{*/

// ----- definition for container class `DistribSetMultiStream' -----
//
class DistribSetMultiStream : public DistribSet {
 public:
  DistribSetMultiStream(DistribSetBasicPtr& audioDistSet, DistribSetBasicPtr& videoDistSet, double audioWeight = 0.95, double videoWeight = 0.05);
  DistribSetMultiStream(DistribSetBasicPtr& audioDistSet, DistribSetBasicPtr& videoDistSet, DistribMapPtr& distribMap, double audioWeight = 0.95, double videoWeight = 0.05);
  virtual ~DistribSetMultiStream() { }

  class Iterator;  	friend class Iterator;
  class ConstIterator;  friend class ConstIterator;

  void resetCache()   { _audioDistribSet->resetCache();   _videoDistribSet->resetCache();   }
  void resetFeature() { _audioDistribSet->resetFeature(); _videoDistribSet->resetFeature(); }

        DistribMultiStreamPtr& find(const String& key)        { return Cast<DistribMultiStreamPtr>(_find(key)); }
  const DistribMultiStreamPtr& find(const String& key)  const { return Cast<DistribMultiStreamPtr>(_find(key)); }

        DistribMultiStreamPtr& find(unsigned dsX)             { return Cast<DistribMultiStreamPtr>(_find(dsX)); }
  const DistribMultiStreamPtr& find(unsigned dsX)       const { return Cast<DistribMultiStreamPtr>(_find(dsX)); }

  DistribSetBasicPtr& audioDistribSet() { return _audioDistribSet; }
  DistribSetBasicPtr& videoDistribSet() { return _videoDistribSet; }
  
  void setWeights( double audioWeight, double videoWeight );

 private:
  DistribSetBasicPtr				_audioDistribSet;
  DistribSetBasicPtr				_videoDistribSet;
};

typedef Inherit<DistribSetMultiStream, DistribSetPtr> DistribSetMultiStreamPtr;


// ----- definition for class `DistibSetMultiStream::Iterator' -----
//
class DistribSetMultiStream::Iterator : public DistribSet::Iterator {
  friend class DistribSetMultiStream;
 public:
  Iterator(DistribSetMultiStreamPtr& dss)
    : DistribSet::Iterator(dss) { }

  inline DistribMultiStreamPtr next();

        DistribMultiStreamPtr& dst()       { return Cast<DistribMultiStreamPtr>(_dst()); }
  const DistribMultiStreamPtr& dst() const { return Cast<DistribMultiStreamPtr>(_dst()); }
};

// needed for Python iterator
//
DistribMultiStreamPtr DistribSetMultiStream::Iterator::next()
{
  if (!more())
    throw jiterator_error("end of distributions!");

  DistribMultiStreamPtr ds(dst());
  operator++(1);
  return ds;
}


// ----- definition for class `DistibSetMultiStream::ConstIterator' -----
//
class DistribSetMultiStream::ConstIterator : public DistribSet::ConstIterator {
  friend class DistribSetMultiStream;
 public:
  ConstIterator(const DistribSetMultiStreamPtr& dss)
    : DistribSet::ConstIterator(dss) { }

  inline const DistribMultiStreamPtr next();

  const DistribMultiStreamPtr& dst() const { return Cast<const DistribMultiStreamPtr>(_dst()); }
};

// needed for Python iterator
//
const DistribMultiStreamPtr DistribSetMultiStream::ConstIterator::next()
{
  if (more()) {
    operator++(1);
    return dst();
  } else {
    throw jiterator_error("end of codebook!");
  }
}

/*@}*/

/*@}*/

/*@}*/

#endif
