//
//			          Millenium
//                    Automatic Speech Recognition System
//                                  (asr)
//
//  Module:  asr.sat
//  Purpose: Speaker-adapted ML and discriminative HMM training.
//  Author:  John McDonough.
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


#ifndef _cluster_h_
#define _cluster_h_

using namespace std;

#include <list>
#include <map>

#include "train/estimateAdapt.h"
#include "train/distribTrain.h"
#include "sat/sat.h"

class BaseRegClassKMeans;
typedef CodebookFastSAT::GaussDensity GaussDensity;

// ----- definition of class `GaussListFastSAT' -----
//
class GaussListFastSAT : public GaussListTrain {
 public:
  GaussListFastSAT() { }
  GaussListFastSAT(CodebookSetFastSATPtr& cb, unsigned idx = 0)
    : GaussListTrain(cb, idx) { }

  class Iterator;       friend class Iterator;
  class ConstIterator;  friend class ConstIterator;
};


// ----- definition of class `GaussListFastSAT::Iterator' -----
//
class GaussListFastSAT::Iterator : public GaussListTrain::Iterator {
 public:
  Iterator(GaussListFastSAT& l)
    : GaussListTrain::Iterator(l) { }

  CodebookFastSAT::GaussDensity& mix() {
    return Cast<CodebookFastSAT::GaussDensity>(_mix());
  }
};


// ----- definition of class `GaussListFastSAT::ConstIterator' -----
//
class GaussListFastSAT::ConstIterator : public GaussListTrain::ConstIterator {
 public:
  ConstIterator(const GaussListFastSAT& l)
    : GaussListTrain::ConstIterator(l) { }

  const CodebookFastSAT::GaussDensity& mix() {
    return Cast<CodebookFastSAT::GaussDensity>(_mix());
  }
};


// ----- definition for class `RegClassTree' -----
//
class RegClassTree;
ostream& operator<<(ostream&, const RegClassTree&);

class RegClassTree : public BaseTree {
  friend ostream& operator<<(ostream& os, RegClassTree& rct);
 public:
  RegClassTree(CodebookSetFastSATPtr& cb, const String& gCovFile = "", int trace = 0x0041,
	       const String& spkrParamFile = "",
	       const String& spkrMeanVarStats = "",
	       const String& spkrListFile = "");
  ~RegClassTree();

  class Node;		friend class Node;
  class LeafIterator;	friend class LeafIterator;

  typedef Inherit<Node, BaseTree::NodePtr> NodePtr;

        CodebookSetFastSATPtr& cbs()       { return Cast<CodebookSetFastSATPtr>(_cbs()); }
  const CodebookSetFastSATPtr& cbs() const { return Cast<CodebookSetFastSATPtr>(_cbs()); }

  NodePtr& operator[](short idx);
  void print(FILE *fp);
  void increment(UnShrt toAdd, bool onlyTopOne = false);

  NodePtr& leftChild(const NodePtr& p)  { return Cast<NodePtr>(_leftChild(p));  }
  NodePtr& rightChild(const NodePtr& p) { return Cast<NodePtr>(_rightChild(p)); }

  NodePtr bestLeaf2Split();

 private:
  static const double SplitFactor;
  static SpeakerList* spkrList;
  static TransformerTreeList* spkrTransList;
  static BaseRegClassKMeans* kMeans;

  NodePtr _bestLeaf(const NodePtr& node, float& score, NodePtr best);
};

typedef Inherit<RegClassTree, BaseTreePtr> RegClassTreePtr;

void incrementRegClasses(CodebookSetFastSATPtr& cbs, unsigned toAdd = 1, int trace = 0x0041,
			 const String& spkrParamFile = "", const String& spkrMeanVarStats = "",
			 const String& spkrListFile = "", const String& gCovFile = "",
			 bool onlyTopOne = false);


// ----- definition for class `RegClassTree::Node' -----
//
class RegClassTree::Node : public BaseTree::Node {
  friend class BaseRegClassKMeans;
  friend class HTKRegClassKMeans;
  friend class SATRegClassKMeans;
  friend class SATRegClassLhoodRatio;
 public:
  Node(RegClassTree& tree, UnShrt indx, CodebookSetFastSATPtr& cb, NodeType typ = Leaf);

  int nComps() const { return _list.size(); }
  float clusterScore() const { return _clusterScore; }

  void split(NodePtr& ch1, NodePtr& ch2, bool onlyTopOne);
  void print() const;
  void clusterChildren(NodePtr& ch1, NodePtr& ch2, bool onlyTopOne);
  void createChildNodes(NodePtr& ch1, NodePtr& ch2, bool onlyTopOne);
  void perturb(float pertDepth);

  GaussListFastSAT& list() { return _list; }

  virtual float calcNodeScore();
  virtual void calcClusterDistribution();
  virtual double calcDistance(NodePtr& ch1, NodePtr& ch2);
  virtual float distance(const GaussDensity& pdf);

 private:
  static       double				RCPerturbation;
  static const double				Threshold;
  static const unsigned				MaxClusterIterations;

  NaturalVector					_aveMean;  	// node cluster mean
  NaturalVector					_aveCovar;	// node cluster variance
  float						_clusterScore;	// node cluster score
  float						_clustAcc;	// accumulates in this cluster
  GaussListFastSAT				_list;		// list of the mixture components
};

class RegClassTree::LeafIterator : private BaseTree::Iterator {
 public:
  LeafIterator(RegClassTreePtr& tr)
    : BaseTree::Iterator(tr, /* onlyLeaves= */ true) { }

  BaseTree::Iterator::operator++;
  BaseTree::Iterator::more;
  BaseTree::Iterator::regClass;

  NodePtr& node() { return Cast<NodePtr>(BaseTree::Iterator::node());  }
};

typedef RegClassTree::Node    RCNode;
typedef RegClassTree::NodePtr RCNodePtr;


// ----- support classes for HTK clustering -----
//
class AccumSum {
 public:
  AccumSum(UnShrt len, UnShrt nsub = 1);
  AccumSum(const GaussDensity& mp);

  void zero();
  float occ() const { return _occ; }
  void operator+=(const AccumSum& acc);
  void calcMean(NaturalVector& mean);
  void calcVariance(NaturalVector& var);

 private:
  const UnShrt					_len;
  float						_occ;
  NaturalVector					_sum; 
  NaturalVector					_sqr;
};

class AccumSumList {
  typedef map<const ListPtr, AccumSum*>		_AccumSumList;
  typedef _AccumSumList::const_iterator		_AccumSumListIter;
 public:
  AccumSumList(CodebookSetFastSATPtr& cb);
  ~AccumSumList();

  const AccumSum& operator[](ListPtr pdf) const;
 private:
  CodebookSetFastSATPtr				_cbs;
  _AccumSumList					_list;
};


// ----- support classes for SAT-based clustering -----
//
class SpkrStat {
 public:
  SpkrStat(const GaussDensity& pdf, const TransformBasePtr& tr);

  const TransformBasePtr& trans()  const { return _trans; }
  float                   occ()    const { return _occ; }

  const NaturalVector&    sumO()   const { return _sumO; }
  const NaturalVector&    sumOsq() const { return _sumOsq; }

 private:
  const UnShrt					_featLen;
  const TransformBasePtr			_trans;
  NaturalVector					_sumO;
  NaturalVector					_sumOsq;
  float						_occ;
};

class SpkrStatList {
  typedef list<const SpkrStat*>			_SpkrStatList;
  typedef _SpkrStatList::const_iterator		_SpkrStatListIter;
  typedef map<ListPtr, _SpkrStatList>		_SpkrStatMap;
  typedef _SpkrStatMap::const_iterator		_SpkrStatMapIter;

 public:
  SpkrStatList(const SpeakerList& spkrList, const TransformerTreeList& transList,
	       CodebookSetFastSATPtr& hmm, const String& accumFileName);
  ~SpkrStatList();

  class Iterator;  friend class Iterator;
 private:
  class CodebookList;

  void _initSpkrStatList(const SpeakerList& spkrList,
			 const TransformerTreeList& transList,
			 CodebookSetFastSATPtr& cb, CodebookList& codebookList,
			 const String& accumDir, bool singleMix = false);

  int						_maxMix;
  CodebookSetFastSATPtr				_cbs;
  _SpkrStatMap					_statMap;
};

class SpkrStatList::CodebookList {
  typedef list<CodebookFastSATPtr>	_CodebookList;
  typedef _CodebookList::const_iterator _CodebookListIter;
 public:
  CodebookList(CodebookSetFastSATPtr& cb);

  class Iterator;  friend class Iterator;
 private:
   _CodebookList    _slist;
};

class SpkrStatList::CodebookList::Iterator {
 public:
  Iterator(const CodebookList& sl)
    : _slist(sl._slist), _itr(_slist.begin()) { }

  const CodebookFastSATPtr cbk() const { return (*_itr); }
  void operator++(int) { _itr++; }
  bool more() { return _itr != _slist.end(); }
  
 private:
  const _CodebookList&  _slist;
  _CodebookListIter  _itr;
};

class SpkrStatList::Iterator {
 public:
  Iterator(SpkrStatList& list, ListPtr pdf)
    : _list(_initList(list, pdf)), _itr(_list.begin()) { }

  const SpkrStat& operator()() { return *(*_itr); }
  void operator++(int) { _itr++; }
  bool more() { return _itr != _list.end(); }

 private:
  const _SpkrStatList&	_initList(SpkrStatList& list, ListPtr pdf);

  static const _SpkrStatList			_EmptyList;

  const _SpkrStatList&				_list;
  _SpkrStatListIter				_itr;
};

typedef enum { No, Yes, YesAndNo } ClusterState;


// ----- definition of class `BaseRegClassKMeans' -----
//
class BaseRegClassKMeans {
 public:
  virtual ~BaseRegClassKMeans() { }

  virtual float calcNodeScore(RCNode* rNode);
  virtual void calcClusterDistribution(RCNode* rNode) = 0;
  virtual double calcDistance(GaussListFastSAT& list, RCNodePtr& ch1, RCNodePtr& ch2) = 0;
  virtual float distance(const GaussDensity& pdf, RCNode* rNode) = 0;

 protected:
  BaseRegClassKMeans(CodebookSetFastSATPtr& cbs, const String& gCovFile, int trace);

  float _euclidean(const NaturalVector& v1, const NaturalVector& v2);
  void _loadGlobalCovariance(NaturalVector& invVar, const String& covFloorFile);

  const UnShrt					_featLen;
  NaturalVector					_grandVariance;
  NaturalVector					_invGrandVariance;
};


// ----- definition of class `HTKRegClassKMeans' -----
//
class HTKRegClassKMeans : public BaseRegClassKMeans {
 public:
  HTKRegClassKMeans(CodebookSetFastSATPtr hmm, const String& gCovFile = "", int trace = 0x0001);

  virtual void calcClusterDistribution(RCNode* rNode);
  virtual double calcDistance(GaussListFastSAT& list, RCNodePtr& ch1, RCNodePtr& ch2);
  virtual float distance(const GaussDensity& pdf, RCNode* rNode);

 private:
  AccumSumList					_accumList;
  AccumSum					_accumSum1;
  AccumSum					_accumSum2;
};


// ----- definition of class `SATRegClassKMeans' -----
//
class SATRegClassKMeans : public BaseRegClassKMeans {
 public:
  SATRegClassKMeans(CodebookSetFastSATPtr& cbs, const TransformerTreeList& transList,
		    SpeakerList& spkrList, const String& meanVarStats,
		    const String& gCovFile, int trace = 0x0001);

  virtual void calcClusterDistribution(RCNode* rNode);
  virtual double calcDistance(GaussListFastSAT& list, RCNodePtr& ch1, RCNodePtr& ch2);
  virtual float distance(const GaussDensity& pdf, RCNode* rNode);

 protected:
  static SpeakerList*				spkrList;
  static TransformerTreeList*			spkrTransList;

  SpkrStatList    				_statList;

 private:
  const UnShrt					_orgFeatLen;

  NaturalVector					_origMean1;
  NaturalVector					_origMean2;
  NaturalVector					_transMean;

  SATMean					_satMean1;
  SATMean					_satMean2;
  SATVariance					_satVariance;
};


// ----- definition of class `SATRegClassLhoodRatio' -----
//
class SATRegClassLhoodRatio : public SATRegClassKMeans {
public:
  SATRegClassLhoodRatio(CodebookSetFastSATPtr& cbs,
			const TransformerTreeList& transList,
			SpeakerList& spkrList, const String& meanVarStats,
			const String& gCovFile, int trace = 0x0001);

  virtual float calcNodeScore(RCNode* rNode);

private:
  static const float MinLogLhoodRatioCount;

  float _logLhoodRatio(const NaturalVector& variance);
  bool  _calcSpeakerVariance(GaussDensity& pdf);
  
  double					_ttlCounts;
  NaturalVector					_spkrVar;
};

#endif
