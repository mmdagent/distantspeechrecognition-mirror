//                              -*- C++ -*-
//
//                              Millennium
//                  Distant Speech Recognition System
//                                 (dsr)
//
//  Module:  asr.lattice
//  Purpose: Lattice pruning and rescoring.
//  Author:  John McDonough and Tobias Matzner
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


#ifndef _lattice_h_
#define _lattice_h_

#include "common/mlist.h"
#include "common/refcount.h"
#include "fsm/fsm.h"
#include "gaussian/distribBasic.h"


// ----- definition for class `Token' -----
//
template <class EdgePtr>
class _Token : public Countable {
 public:

  _Token(double acs, double lms, int frameX, const EdgePtr edge,
	 const refcountable_ptr<_Token>& prev = NULL, const refcountable_ptr<_Token>& worse = NULL)
    : _acScore(acs), _lmScore(lms),
      _frameX(frameX), _edgeA(edge), _prev(prev), _worseList(worse) { }

  static void memoryUsage() {
    memoryManager().report();
  }

  void*	operator new(size_t sz) { return memoryManager().newElem(); }
  void	operator delete(void* e) { memoryManager().deleteElem(e); }

  float	                        	acScore()      const { return _acScore;            }
  float	                        	lmScore()      const { return _lmScore;            }
  int					frameX()       const { return _frameX;             }
  float	                	        score()        const { return _acScore + _lmScore; }
  const EdgePtr&	                edge()         const { return _edgeA;              }
  const refcountable_ptr<_Token>&	prev()         const { return _prev;               }
  const refcountable_ptr<_Token>&	worse()        const { return _worseList;          }

  void write(FILE* fp = stdout) {
    printf("score = %10.4f : acScore = %10.4f : lmScore = %10.4f\n", score(), acScore(), lmScore());
  }

  void setWorse(refcountable_ptr<_Token>& worseToken) { _worseList = worseToken; }
  void setWorse(_Token* worseToken)                   { _worseList = worseToken; }

  static void setMemoryLimit(unsigned limit) { memoryManager().setLimit(limit); }

 protected:
  static MemoryManager<_Token>& memoryManager();

  const float					_acScore;
  const float					_lmScore;
  const int					_frameX;
  const EdgePtr					_edgeA;
  refcountable_ptr<_Token>			_prev;
  refcountable_ptr<_Token>			_worseList;
};

typedef _Token<WFSTransducer::EdgePtr>	Token;
typedef refcountable_ptr<Token>		TokenPtr;

template<>
MemoryManager<_Token<Inherit<WFSTransducer::Edge, refcountable_ptr<WFSAcceptor::Edge> > > >&
_Token<Inherit<WFSTransducer::Edge, refcountable_ptr<WFSAcceptor::Edge> > >::memoryManager();


// ----- definition for class `LatticeNodeData' -----
//
class LatticeNodeData {
 public:
  LatticeNodeData()
    : _forwardProb(LogZero), _backwardProb(LogZero), _tok(NULL)
  {
    // printf("Constructing LatticeNodeData\n");  fflush(stdout);
  }

  void read(FILE* fp) {
    // fscanf(fp, "%lf  %lf\n", &_forwardProb, &_backwardProb);
  }

  void write(FILE* fp = stdout) const {
    // fprintf(fp, "%f  %f\n", _forwardProb, _backwardProb);  fflush(fp);
  }

  const TokenPtr& tok() const { return _tok; }

  void  logAddForward(LogDouble score)  { _forwardProb  = ::logAdd(score, _forwardProb);  }
  void  logAddBackward(LogDouble score) { _backwardProb = ::logAdd(score, _backwardProb); }

  LatticeNodeData& logZeroForward()  { _forwardProb  = LogZero; return *this; }
  LatticeNodeData& logZeroBackward() { _backwardProb = LogZero; return *this; }
  LatticeNodeData& zeroForward()     { _forwardProb  = 0.0;     return *this; }
  LatticeNodeData& zeroBackward()    { _backwardProb = 0.0;     return *this; }

  void  clearToken() { _tok = NULL; }
  void  clearProbs() { _forwardProb = _backwardProb = LogZero; }
  void  setToken(const TokenPtr& t) { _tok = t; }

  LogDouble forwardProb()  const { return _forwardProb;  }
  LogDouble backwardProb() const { return _backwardProb; }

 private:
  LogDouble					_forwardProb;
  LogDouble					_backwardProb;

  TokenPtr					_tok;
};


// ----- definition for class `LatticeEdgeData' -----
//
class LatticeEdgeData {
 public:
  LatticeEdgeData()
    : _start(-1), _end(-1), _ac(0.0), _lm(0.0), _gamma(0.0) { }
  LatticeEdgeData(int s, int e, double ac, double lm)
    : _start(s), _end(e), _ac(ac), _lm(lm), _gamma(0.0) { }

  void read(FILE* fp) {
    int nmatch = fscanf(fp, "%d %d %lf %lf %lf\n", (int*) &_start, (int*) &_end, &_ac, &_lm, &_gamma);
    if (nmatch != 5)
      throw jio_error("Only matched %d elements.", nmatch);
    /*
    printf("%4d  %4d  %8.4f  %8.4f  %8.4f\n", _start, _end, _ac, _lm, _gamma);
    fflush(stdout);
    */
  }

  void write(FILE* fp = stdout) const {
    fprintf(fp, "%4d  %4d  %8.4f  %8.4f  %8.4f\n", _start, _end, _ac, _lm, _gamma);
    fflush(fp);
  }

  int    start() const { return _start; }
  int    end()   const { return _end;   }
  double ac()    const { return _ac;    }
  double lm()    const { return _lm;    }
  double gamma() const { return _gamma; }

  void setGamma(double g) { _gamma = g; }
  void setAc(double a)    { _ac = a; }

 private:
  const int					_start;
  const int					_end;
	double					_ac;
	double					_lm;
  	double					_gamma;
};

typedef WFST<LatticeNodeData, LatticeEdgeData >		LatticeData;
typedef Inherit<LatticeData, WFSTransducerPtr >		WFSTLatticeDataPtr;

template<>
MemoryManager<WFST<LatticeNodeData, LatticeEdgeData>::Edge>& WFST<LatticeNodeData, LatticeEdgeData>::Edge::memoryManager();

template<>
MemoryManager<WFST<LatticeNodeData, LatticeEdgeData>::Node>& WFST<LatticeNodeData, LatticeEdgeData>::Node::memoryManager();


// ----- definition for class `Lattice' -----
//
class Lattice;
typedef Inherit<Lattice,  WFSTLatticeDataPtr >	LatticePtr;

class Lattice : public LatticeData {
  template <class WFSType, class NodePtr, class EdgePtr, class WFSTypePtr> friend class _Decoder;
 public:
  static void reportMemoryUsage();

  class Node;
  class Edge;

  typedef Inherit<Node, LatticeData::NodePtr> NodePtr;
  typedef Inherit<Edge, LatticeData::EdgePtr> EdgePtr;

  class EdgeIterator;  friend class EdgeIterator;

  // private:
  typedef list<NodePtr>			_NodeList;
  typedef _NodeList::iterator		_Iterator;
  typedef _NodeList::reverse_iterator	_ReverseIterator;
  typedef _NodeList::const_iterator	_ConstIterator;

  typedef vector<NodePtr>		_NodeVector;
  typedef _NodeVector::iterator		_NodeVectorIterator;
  typedef _NodeVector::const_iterator	_ConstNodeVectorIterator;

  typedef map<unsigned, NodePtr >	_NodeMap;
  typedef _NodeMap::iterator		_NodeMapIterator;
  typedef _NodeMap::const_iterator	_ConstNodeMapIterator;
  typedef _NodeMap::value_type		_ValueType;

 public:
  // Lattice(LexiconPtr& statelex, LexiconPtr& inlex, LexiconPtr& outlex);
  Lattice(LexiconPtr statelex, LexiconPtr inlex, LexiconPtr outlex);
  Lattice(WFSTSortedInputPtr& wfst);
  ~Lattice();

  NodePtr& initial() { return Cast<NodePtr>(_initial); }

  NodePtr find(unsigned state, bool create = false) {
    return Cast<NodePtr>(_find(state, create));
  }

  // rescore the lattice with given LM scale factor and penalty
  float	rescore(double lmScale = 30.0, double lmPenalty = 0.0, double silPenalty = 0.0, const String& silSymbol = "SIL-m");

  // write best word hypo to disk in CTM format
  void	writeCTM(const String& conv, const String& channel, const String& spk, const String& utt,
		 double cfrom, double score, const String& fileName = "", double frameInterval = 0.01,
		 const String& endMarker = "</s>") /* const */;

  // write best phone hypo to disk in CTM format
  void	writePhoneCTM(const String& conv, const String& channel, const String& spk, const String& utt,
		      double cfrom, double score, const String& fileName = "", double frameInterval = 0.01,
		      const String& endMarker = "</s>") /* const */;

  // write best hypo to disk in HTK format
  void	writeHypoHTK(const String& conv, const String& channel, const String& spk, const String& utt,
		     double cfrom, double score, const String& fileName = "", int flag = 0, double frameInterval = 0.01,
		     const String& endMarker = "</s>");

  // get the 1-best hypo
  String bestHypo(bool useInputSymbols = false) /* const */;

  // prune the lattice
  void	prune(double threshold = 100.0);

  // prune the lattice
  void pruneEdges(unsigned edgesN = 0);

  // purge nodes from which an end node cannot be reached
  void	purge();

  // calculate posterior probabilities of the links
  double gammaProbs(double acScale = 1.0, double lmScale = 12.0, double lmPenalty = 0.0,
		    double silPenalty = 0.0, const String& silSymbol = "SIL-m");

  // calculate posterior probabilities of the links with given distribution set
  double gammaProbsDist(DistribSetBasicPtr& dss, double acScale = 1.0, double lmScale = 12.0, double lmPenalty = 0.0,
			double silPenalty = 0.0, const String& silSymbol = "SIL-m");

  // write the lattice in topo sorted order
  void write(const String& fileName = "", bool useSymbols = false, bool writeData = false);

  // write best word sequence annotated with confidence scores
  void writeWordConfs(const String& fileName, const String& uttId, const String& endMarker = "</s>");

  // create a phone lattice
  WFSTSortedInputPtr createPhoneLattice(LatticePtr& lattice, double acScale);

  // protected:
  virtual void _clear();

  virtual WFSAcceptor::Node* _newNode(unsigned state);
  virtual WFSAcceptor::Edge* _newEdge(NodePtr& from, NodePtr& to, unsigned input, unsigned output, Weight cost = ZeroWeight);

  _NodeMap&    _finis()    { return Cast<_NodeMap>(_final); }
  _NodeVector& _allNodes() { return Cast<_NodeVector>(_nodes); }
  _NodeList&   _snodes()   { return Cast<_NodeList>(_sortedNodes); }

  // helper methods for topo sorting
  void	_topoSort();
  void	_clearSorted();
  void	_visitNode(NodePtr& node);

  // helper methods for lattice purging
  void	_setSuccess(bool s);
  bool	_purgeNode(NodePtr& node);
  void	_removeUnsuccessful();

  // helper methods for rescoring
  void	_clearTokens();
  TokenPtr _bestToken() /* const */;

  // helper methods for forward-backward calculation
  void	_forwardProbs();
  void	_backwardProbs();
  void	_gammaProbs();

  void	_expandNode(const NodePtr& node) const;
  void	_forwardNode(const NodePtr& node) const;
  void	_backwardNode(NodePtr& node) const;
  void	_gammaNode(NodePtr& node) /* const */;

  // helper methods for updating edge acoustic likelihoods
  void	_updateAc(DistribSetBasicPtr& dss);
  void	_updateAcNode(DistribSetBasicPtr& dss, NodePtr& node);

  // helper method for phone lattice creation and manipulation
  void _addNode(WFSTSortedInputPtr& wfst, WFSTSortedInputNodePtr& node);
  unsigned _calculateIndex(const EdgePtr& edge);
  unsigned _calculateStart(const WFSTSortedInputEdgePtr& edge);
  unsigned _calculateEnd(const WFSTSortedInputEdgePtr& edge);
  unsigned _calculateOutput(const WFSTSortedInputEdgePtr& edge);


  double	_acScale;		// acoustic scale used for rescoring
  double	_lmScale;		// LM scale used for rescoring
  double	_lmPenalty;		// word insertion penalty
  double	_silPenalty;		// silence insertion penalty
  String	_silSymbol;		// silence input symbol
  unsigned	_silenceX;		// silence index

  double	_latticeForwardProb;	// forward probability

  _NodeList	_sortedNodes;		// topo sorted node list
};


// ----- definition for class `Lattice::Node' -----
//
class Lattice::Node : public LatticeData::Node {
  friend class Lattice;

  typedef list<EdgePtr>		 	_EdgeList;
  typedef _EdgeList::iterator		_EdgeListIterator;
  typedef _EdgeList::const_iterator	_ConstEdgeListIterator;
 public:
  Node(unsigned idx, const LatticeNodeData& d, Weight cost = ZeroWeight)
    : LatticeData::Node(idx, d, Weight(cost)), _success(false) { }

  virtual ~Node() { }

  void*	operator new(size_t sz) { return memoryManager().newElem(); }
  void	operator delete(void* e) { memoryManager().deleteElem(e); }

  void setSuccess(bool s) { _success = s; }
  bool success() const { return _success; }

  virtual void addEdge(EdgePtr& ed);

#if 0
  // NOTE: Why doesn't this work??!!??
  
  // ----- definition for class `Lattice::Node::Iterator' -----
  // 
  class Iterator : public LatticeData::Node::Iterator {
  public:
    Iterator(NodePtr& node)
      : LatticeData::Node::Iterator(node) { }

    ~Iterator() { }

    EdgePtr& edge() { return Cast<EdgePtr>(_edge()); }
    EdgePtr next() {
      if (!more())
	throw jiterator_error("end of edges!");

      EdgePtr ed(edge());
      operator++(1);
      return ed;
    }
  };
#endif

 protected:
  EdgePtr& _edges() { return Cast<EdgePtr>(_edgeList); }

 private:
  static MemoryManager<Node>& memoryManager();

  void _removeLinks(double threshold = 100.0);
  _EdgeList& _allEdges() { return Cast<_EdgeList>(_edges()); }

  bool			_success;
};



// ----- definition for class `Lattice::Edge' -----
//
class Lattice::Edge : public LatticeData::Edge {
  friend class Lattice;
 public:
  Edge(NodePtr& prev, NodePtr& next,
       unsigned input, unsigned output, const LatticeEdgeData& d, float cost = 0.0)
    : LatticeData::Edge(prev, next, input, output, d, Weight(cost)) { }

  virtual ~Edge() { }

  void* operator new(size_t sz) { return memoryManager().newElem(); }
  void	operator delete(void* e) { memoryManager().deleteElem(e); }

  	NodePtr& prev()       { return Cast<NodePtr>(_prev); }
  	NodePtr& next()       { return Cast<NodePtr>(_next); }
  const NodePtr& prev() const { return Cast<NodePtr>(_prev); }
  const NodePtr& next() const { return Cast<NodePtr>(_next); }

  EdgePtr& _edges() { return Cast<EdgePtr>(_edgeList); }

 private:
  static MemoryManager<Edge>& memoryManager();
};


// ----- definition for class `Lattice::EdgeIterator' -----
//
class Lattice::EdgeIterator {
  typedef list<EdgePtr>		 	_EdgeList;
  typedef _EdgeList::iterator		_EdgeListIterator;
  typedef _EdgeList::const_iterator	_ConstEdgeListIterator;
 public:
  EdgeIterator(LatticePtr& lat);
  EdgeIterator(Lattice& lat);
  ~EdgeIterator() { }

  void operator++(int) { if(more()) _itr++; }
  bool more()          { return _itr != _edgeList.end(); }
  EdgePtr& edge()      { return *_itr; }

 private:
  _EdgeList		_edgeList;
  _EdgeListIterator	_itr;
};

void Lattice_reportMemoryUsage();


// ----- definition for class `DepthFirstApplyConfidences' -----
// 
class DepthFirstApplyConfidences : private DepthFirstSearch {
  typedef Lattice::EdgePtr		EdgePtr;
  typedef Lattice::NodePtr		NodePtr;

 public:
  DepthFirstApplyConfidences(const ConfidenceListPtr& confidenceList)
    : _confidenceList(confidenceList) { }
  virtual ~DepthFirstApplyConfidences() { }

  void apply(LatticePtr& A) { _wordDepth = -1;  search(A); }

 protected:
  virtual void _expandNode(WFSTransducerPtr& A, WFSTransducer::NodePtr& node);

  const ConfidenceListPtr			_confidenceList;
  int						_wordDepth;
};

void applyConfidences(LatticePtr& A, const ConfidenceListPtr& confidenceList);


// ----- definition for class `ConsensusGraph' -----
// 
class ConsensusGraph : public Lattice {
  typedef Lattice::EdgePtr			EdgePtr;
  typedef Lattice::NodePtr			NodePtr;

  typedef list<EdgePtr>				_EdgeList;
  typedef _EdgeList::iterator			_EdgeListIterator;

  class _EquivalenceClass {
  public:
    _EquivalenceClass(EdgePtr& edge);

    void merge(_EquivalenceClass* equiv);

    unsigned symbol() const { return _symbolX; }
    int start() const { return _startX; }
    int end()   const { return _endX; }
    void add(EdgePtr& edge) { _edgeList.push_back(edge); }
    
    unsigned					_symbolX;
    int 					_startX;
    int						_endX;
    _EdgeList					_edgeList;
    double					_score;
  };

  class LessThan {
  public:
    bool operator()(_EquivalenceClass* e1, _EquivalenceClass* e2) {
      return (e1->start() < e2->start());
    }
  };

  typedef list<_EquivalenceClass>		_EquivalenceList;
  typedef _EquivalenceList::iterator		_EquivalenceListIterator;
  typedef _EquivalenceList::const_iterator	_EquivalenceListConstIterator;

  typedef map<unsigned, _EquivalenceList >	_EquivalenceMap;
  typedef _EquivalenceMap::iterator		_EquivalenceMapIterator;
  typedef _EquivalenceMap::const_iterator	_ConstEquivalenceMapIterator;
  typedef _EquivalenceMap::value_type		_ValueType;

  typedef map<_EquivalenceClass*, _EquivalenceClass>	_EquivalencePointerMap;
  typedef _EquivalencePointerMap::iterator		_EquivalencePointerMapIterator;
  typedef _EquivalencePointerMap::value_type		_EquivalencePointerMapValueType;

  typedef vector<_EquivalenceClass*>		_SortedClusters;
  typedef _SortedClusters::iterator		_SortedClustersIterator;
  typedef _SortedClusters::value_type		_SortedClustersValueType;

  class _MergeCandidate {
  public:
    _MergeCandidate(_EquivalenceClass* first, _EquivalenceClass* second, double score = 0.0)
      : _score(score), _first(first), _second(second) { }

    double					_score;
    _EquivalenceClass*				_first;
    _EquivalenceClass*				_second;
  };

  typedef list<_MergeCandidate>			_MergeList;
  typedef _MergeList::iterator			_MergeListIterator;
  typedef _MergeList::const_iterator		_MergeListConstIterator;

  typedef map<unsigned, double> 		PosteriorProbabilityMap;
  typedef PosteriorProbabilityMap::iterator	PosteriorProbabilityMapIterator;

public:
  ConsensusGraph(LatticePtr& lattice, WFSTLexiconPtr& lexicon);
  ~ConsensusGraph() { }

private:
  _MergeCandidate* _initializeIntraWordClusters(LatticePtr& lattice);
  void _intraWordClustering(LatticePtr& lattice);
  _MergeCandidate* _initializeInterWordClusters();
  void _interWordClustering(LatticePtr& lattice);
  void _constructGraph(LatticePtr& lattice);

  _MergeCandidate* _bestMerge(_EquivalenceClass* first, _EquivalenceClass* second, bool interWord);

  double _updateIntraWordScore(_MergeCandidate& merge);
  double _updateInterWordScore(_MergeCandidate& merge);

  _EquivalenceMap				_intraWordClusters;
  _EquivalencePointerMap			_interWordClusters;
  _SortedClusters				_sortedClusters;
  _MergeList					_mergeList;
  WFSTLexiconPtr				_lexicon;
};

typedef Inherit<ConsensusGraph, LatticePtr> ConsensusGraphPtr;

ConsensusGraphPtr createConsensusGraph(LatticePtr& lattice, WFSTLexiconPtr& lexicon);

#endif
