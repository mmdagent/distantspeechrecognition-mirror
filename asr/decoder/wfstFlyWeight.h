//                              -*- C++ -*-
//
//                               Millennium
//                    Distant Speech Recognition System
//                                 (dsr)
//
//  Module:  asr.decoder
//  Purpose: Token passing decoder with lattice generation.
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


#ifndef _wfstFlyWeight_h_
#define _wfstFlyWeight_h_

#include "fsm/fsm.h"

/**
* \addtogroup WFSTStructure
*/
/*@{*/

/**
* \defgroup WFSTFlyWeight Static WFST.
*/
/*@{*/

// ----- definition for class `WFSTFlyWeight' -----
//
class WFSTFlyWeight;
typedef refcountable_ptr<WFSTFlyWeight>		WFSTFlyWeightPtr;

typedef Countable WFSTFlyWeightCountable;
class WFSTFlyWeight : public WFSTFlyWeightCountable {
  static const int EndMarker;

 public:
  static void reportMemoryUsage();

  class Node;
  class Edge;
  class Iterator;

  template <class WFSType, class NodePtr, class EdgePtr, class WFSTypePtr> friend class _Decoder;

  WFSTFlyWeight(LexiconPtr& statelex, LexiconPtr& inlex, LexiconPtr& outlex,
		const String& name = "WFSTFlyWeightSortedInput");
  ~WFSTFlyWeight();

        Node* initial()       { return _initial; }
  const Node* initial() const { return _initial; }
  inline Node* initial(int idx);

  Node* find(unsigned state, bool create = false);

  const Edge* edges(WFSAcceptor::Node* node);
  void read(const String& fileName, bool binary = false);
  void reverseRead(const String& fileName);
  void write(const String& fileName, bool binary = true, bool useSymbols = false) const;

  void reverse(const WFSTFlyWeightPtr& wfst);

  void discount(const String& fileName);
  void calcAndApplyDiscount(const String& fileName);

  void indexNodes();

        LexiconPtr& stateLexicon()        { return _stateLexicon;  }
  const LexiconPtr& stateLexicon()  const { return _stateLexicon;  }
        LexiconPtr& inputLexicon()        { return _inputLexicon;  }
  const LexiconPtr& inputLexicon()  const { return _inputLexicon;  }
        LexiconPtr& outputLexicon()       { return _outputLexicon; }
  const LexiconPtr& outputLexicon() const { return _outputLexicon; }

  bool hasFinal(unsigned state);
  bool hasFinalState() const { return (_final.size() > 0); }

  // private:
  typedef map<unsigned, Node* >			_NodeMap;
  typedef _NodeMap::iterator			_NodeMapIterator;
  typedef _NodeMap::const_iterator		_ConstNodeMapIterator;
  typedef _NodeMap::value_type			_ValueType;

  typedef map<unsigned, float>			_DiscountMap;
  typedef map<unsigned, float>::const_iterator	_DiscountMapIterator;
  typedef map<unsigned, float>::value_type	_DiscountMapValueType;

  void _readText(FILE* fp);
  void _readBinary(FILE* fp);
  void _clear();
  _NodeMap& final()  { return _final; }
  void _addFinal(unsigned state, float cost);
  void _applyDiscount(const _DiscountMap& discountMap);
  void _calculateDiscount(_DiscountMap& discountMap);
  void _calcDiscount(_DiscountMap& discountMap, Edge edge);

  virtual Node* _newNode(unsigned state);
  virtual Edge* _newEdge(Node* from, Node* to, unsigned input, unsigned output, float cost = 0.0);

  String					_name;

  LexiconPtr					_stateLexicon;
  LexiconPtr					_inputLexicon;
  LexiconPtr					_outputLexicon;

  Node*						_initial;
  _NodeMap					_nodes;
  _NodeMap					_final;
};


// ----- definition for class `WFSTFlyWeight::Edge' -----
//
class WFSTFlyWeight::Edge {
  friend class WFSTFlyWeight;
  friend class WFSTFlyWeight::Node;
  friend void _addFinal(unsigned state, float cost);
 public:
  Edge(Node* prev, Node* next,
       unsigned input, unsigned output , float cost = 0.0)
    : _prev(prev), _next(next), _edgeList(NULL),
    _input(input), _output(output), _cost(cost) { }
    
  virtual ~Edge() { delete _edgeList; }

  void*	operator new(size_t sz) { return memoryManager().newElem(); }
  void	operator delete(void* e) { memoryManager().deleteElem(e); }

  	Node* prev()         { return _prev; }
  	Node* next()         { return _next; }
  const Node* prev()   const { return _prev; }
  const Node* next()   const { return _next; }

  unsigned    input()  const { return _input;  }
  unsigned    output() const { return _output; }
  float       cost()   const { return _cost;   }

  void setCost(float c = 0.0) { _cost = c; }

  void write(FILE* fp, const WFSTFlyWeight* wfst, bool binary) const;
  void write(const LexiconPtr& statelex, const LexiconPtr& inlex, const LexiconPtr& outlex, FILE* fp = stdout) const;

        Edge* _edges()       { return _edgeList; }
  const Edge* _edges() const { return _edgeList; }

  static void report() { memoryManager().report(); }

  // protected:
  Node*						_prev;
  Node*						_next;
  Edge*						_edgeList;

#if 1
  const unsigned				_input;
  const unsigned				_output;
#else
  const unsigned short				_input;
  const unsigned short				_output;
#endif
  /* const */ float				_cost;

  static MemoryManager<Edge>& memoryManager();
};


// ----- definition for class `WFSTFlyWeight::Node' -----
//
class WFSTFlyWeight::Node {

  typedef struct {
    unsigned final:1, color:2, index:29;
  } _NodeIndex;

  friend class WFSTFlyWeight;

 public:
  Node(unsigned idx, float cost = 0.0);
  virtual ~Node();

  unsigned index() const { return _index.index; }
  float cost()     const { return _cost; }
  bool isFinal() const { return (_index.final == 1); }

  void*	operator new(size_t sz) { return memoryManager().newElem(); }
  void	operator delete(void* e) { memoryManager().deleteElem(e); }

  class Iterator;  friend class Iterator;

        Edge* _edges()       { return _edgeList; }
  const Edge* _edges() const { return _edgeList; }


  void write(FILE* fp, bool binary);

  void writeArcs(const LexiconPtr& statelex, const LexiconPtr& inlex, const LexiconPtr& outlex, FILE* fp = stdout) const;

  static void report() { memoryManager().report(); }

 protected:
  static const unsigned _MaximumIndex;

  void _setCost(float c = 0.0) { _cost = c; _index.final = 1; }
  virtual void _addEdgeForce(Edge* newEdge);
  void _indexEdges();

  _NodeIndex					_index;
  float						_cost;
  Edge*						_edgeList;

  static MemoryManager<Node>& memoryManager();
};

WFSTFlyWeight::Node* WFSTFlyWeight::initial(int idx)
{
  if (_initial == NULL) _initial = ((idx >= 0) ? new Node(idx) : new Node(0));

  return Cast<Node*>(_initial);
}


// ----- definition for class `WFSTFlyWeight::Node::Iterator' -----
//
class WFSTFlyWeight::Node::Iterator {
 public:
  Iterator(const WFSTFlyWeight::Node* node)
    : _edgePtr(node->_edgeList) { }
  Iterator(WFSTFlyWeightPtr& wfst, const WFSTFlyWeight::Node* node)
    : _edgePtr(node->_edgeList) { }
  ~Iterator() { }

  bool more() const { return _edgePtr != NULL; }
  const Edge* edge() { return _edgePtr; }
  void operator++(int) {
    if (more()) _edgePtr = _edgePtr->_edges();
  }

 protected:
  const Edge*					_edgePtr;
};

void WFSTFlyWeight_reportMemoryUsage();

/*@}*/

/**
* \defgroup WFSTFlyWeightSortedInput Static WFST with sorted input symbols.
*/
/*@{*/

// ----- definition for class `WFSTFlyWeightSortedInput' -----
//
class WFSTFlyWeightSortedInput : public WFSTFlyWeight {
public:
  class Node;
  class Edge;
  class Iterator;

  WFSTFlyWeightSortedInput(LexiconPtr& statelex, LexiconPtr& inlex, LexiconPtr& outlex, const String& name = "WFSTFlyWeight")
    : WFSTFlyWeight(statelex, inlex, outlex, name) { }

        Node* initial()       { return Cast<Node*>(_initial); }
  const Node* initial() const { return Cast<const Node*>(_initial); }

  void hash();

protected:
  virtual WFSTFlyWeight::Node* _newNode(unsigned state);
  virtual WFSTFlyWeight::Edge* _newEdge(WFSTFlyWeight::Node* from, WFSTFlyWeight::Node* to,
					unsigned input, unsigned output, float cost = 0.0);
};

typedef Inherit<WFSTFlyWeightSortedInput, WFSTFlyWeightPtr>  WFSTFlyWeightSortedInputPtr;


// ----- definition for class `WFSTFlyWeightSortedInput::Edge' -----
//
class WFSTFlyWeightSortedInput::Edge : public WFSTFlyWeight::Edge {
  friend void _addFinal(unsigned state, float cost);
 public:
  Edge(Node* prev, Node* next,
       unsigned input, unsigned output, float cost = 0.0);
  virtual ~Edge() { }

  void*	operator new(size_t sz) { return memoryManager().newElem(); }
  void	operator delete(void* e) { memoryManager().deleteElem(e); }

  	Node* prev()         { return Cast<Node*>(_prev); }
  	Node* next()         { return Cast<Node*>(_next); }
  const Node* prev()   const { return Cast<Node*>(_prev); }
  const Node* next()   const { return Cast<Node*>(_next); }

        Edge* _edges()       { return Cast<Edge*>(_edgeList); }
  const Edge* _edges() const { return Cast<Edge*>(_edgeList); }

  Edge*						_chain;

 private:
  static MemoryManager<Edge>& memoryManager();
};


// ----- definition for class `WFSTFlyWeightSortedInput::Node' -----
//
class WFSTFlyWeightSortedInput::Node : public WFSTFlyWeight::Node {
private:
  static const float MaxHashDepth;
  static const unsigned PrimeN;
  static const unsigned Primes[];

  class _EdgeMap {
  public:
    _EdgeMap();
    ~_EdgeMap();

    inline Edge* edge(unsigned symbol) const;
    float hash(Edge* edge);

    float averageHashDepth() const;

  private:
    unsigned _hash(unsigned symbol) const { return symbol % _binsN; }

    unsigned					_binsN;
    Edge**					_bins;
  };

 public:
  Node(unsigned idx, float cost = 0.0)
    : WFSTFlyWeight::Node(idx, cost) { }
  virtual ~Node() { }

  void*	operator new(size_t sz) { return memoryManager().newElem(); }
  void	operator delete(void* e) { memoryManager().deleteElem(e); }

  void hash();
  Edge* edge(unsigned symbol) const { return _edgeMap.edge(symbol); }

  class Iterator;  friend class Iterator;

  // protected:
  static bool Verbose;
  virtual void _addEdgeForce(WFSTFlyWeight::Edge* ed);

        Edge* _edges()       { return Cast<Edge*>(_edgeList); }
  const Edge* _edges() const { return Cast<const Edge*>(_edgeList); }

 private:
  static MemoryManager<Node>& memoryManager();

  _EdgeMap					_edgeMap;
};

WFSTFlyWeightSortedInput::Edge* WFSTFlyWeightSortedInput::Node::_EdgeMap::edge(unsigned symbol) const
{
  if (_binsN == 0) return NULL;

  Edge* edge = _bins[_hash(symbol)];
  while (edge != NULL) {
    if (edge->input() == symbol) return edge;
    edge = edge->_chain;
  }
  return NULL;
}


// ----- definition for class `WFSTFlyWeightSortedInput::Node::Iterator' -----
//
class WFSTFlyWeightSortedInput::Node::Iterator : public WFSTFlyWeight::Node::Iterator {
 public:
  Iterator(const WFSTFlyWeightSortedInput::Node* node)
    : WFSTFlyWeight::Node::Iterator(node) { }
  Iterator(WFSTFlyWeightSortedInputPtr& wfst, const WFSTFlyWeightSortedInput::Node* node)
    : WFSTFlyWeight::Node::Iterator(wfst, node) { }

  ~Iterator() { }

  const WFSTFlyWeightSortedInput::Edge* edge() const { return Cast<WFSTFlyWeightSortedInput::Edge*>(_edgePtr); }
};


/*@}*/

/**
* \defgroup WFSTFlyWeightSortedOuput Static WFST with sorted output symbols.
*/
/*@{*/

// ----- definition for class `WFSTFlyWeightSortedOuput' -----
//
class WFSTFlyWeightSortedOutput : public WFSTFlyWeight {
public:
  class Node;
  class Edge;
  class Iterator;

  WFSTFlyWeightSortedOutput(LexiconPtr& statelex, LexiconPtr& inlex, LexiconPtr& outlex, const String& name = "WFSTFlyWeight")
    : WFSTFlyWeight(statelex, inlex, outlex, name) { }

        Node* initial()       { return Cast<Node*>(_initial); }
  const Node* initial() const { return Cast<const Node*>(_initial); }

  // populate edges with coming symbols for lookahead
  void setComingSymbols();

protected:
  virtual WFSTFlyWeight::Node* _newNode(unsigned state);
  virtual WFSTFlyWeight::Edge* _newEdge(WFSTFlyWeight::Node* from, WFSTFlyWeight::Node* to,
					unsigned input, unsigned output, float cost = 0.0);
};

typedef Inherit<WFSTFlyWeightSortedOutput, WFSTFlyWeightPtr>  WFSTFlyWeightSortedOutputPtr;


// ----- definition for class `WFSTFlyWeightSortedOutput::Edge' -----
//
class WFSTFlyWeightSortedOutput::Edge : public WFSTFlyWeight::Edge {
  friend void _addFinal(unsigned state, float cost);
 public:
  Edge(Node* prev, Node* next,
       unsigned input, unsigned output, float cost = 0.0);
  virtual ~Edge() { }

  void*	operator new(size_t sz) { return memoryManager().newElem(); }
  void	operator delete(void* e) { memoryManager().deleteElem(e); }

  	Node* prev()         { return Cast<Node*>(_prev); }
  	Node* next()         { return Cast<Node*>(_next); }
  const Node* prev()   const { return Cast<Node*>(_prev); }
  const Node* next()   const { return Cast<Node*>(_next); }

        Edge* _edges()       { return Cast<Edge*>(_edgeList); }
  const Edge* _edges() const { return Cast<const Edge*>(_edgeList); }

  list<unsigned> findComing() const;

 private:
  static MemoryManager<Edge>& memoryManager();
};


// ----- definition for class `WFSTFlyWeightSortedOutput::Node' -----
//
class WFSTFlyWeightSortedOutput::Node : public WFSTFlyWeight::Node {
 public:
  Node(unsigned idx, float cost = 0.0);
  virtual ~Node();

  void*	operator new(size_t sz) { return memoryManager().newElem(); }
  void	operator delete(void* e) { memoryManager().deleteElem(e); }

  const unsigned* comingSymbols() const { return _comingSymbols; }
  void findComing();

  class Iterator;  friend class Iterator;

  // protected:
  virtual void _addEdgeForce(WFSTFlyWeight::Edge* ed);

        Edge* _edges()       { return Cast<Edge*>(_edgeList); }
  const Edge* _edges() const { return Cast<const Edge*>(_edgeList); }

  unsigned*					_comingSymbols;

 private:
  static MemoryManager<Node>& memoryManager();
};


// ----- definition for class `WFSTFlyWeightSortedOutput::Node::Iterator' -----
//
class WFSTFlyWeightSortedOutput::Node::Iterator : public WFSTFlyWeight::Node::Iterator {
 public:
  Iterator(const WFSTFlyWeightSortedOutput::Node* node)
    : WFSTFlyWeight::Node::Iterator(node) { }
  Iterator(WFSTFlyWeightSortedOutputPtr& wfst, const WFSTFlyWeightSortedOutput::Node* node)
    : WFSTFlyWeight::Node::Iterator(wfst, node) { }

  ~Iterator() { }

  WFSTFlyWeightSortedOutput::Edge* edge() const { return Cast<WFSTFlyWeightSortedOutput::Edge*>(_edgePtr); }
};

/*@}*/

/*@}*/

/**
* \defgroup Decoder
*/
/*@{*/

// ----- definition for class `Fence' -----
//
typedef Countable FenceCountable;
class Fence : public FenceCountable {
public:
  class Iterator;
  class FenceIndex {
    friend class Fence;
    friend class Iterator;
  public:
    FenceIndex(unsigned indexA, unsigned indexB, FenceIndex* next, FenceIndex* chain)
      : _indexA(indexA), _indexB(indexB), _next(next), _chain(chain) { }
    
    void* operator new(size_t sz) { return memoryManager().newElem(); }
    void  operator delete(void* e) { memoryManager().deleteElem(e); }

    static MemoryManager<FenceIndex>& memoryManager();

    unsigned indexA() const { return _indexA; }
    unsigned indexB() const { return _indexB; }

  private:
    const unsigned				_indexA;
    const unsigned				_indexB;

    FenceIndex*					_next;
    FenceIndex*					_chain;
  };

  Fence(unsigned bucketN = 4000001, const String& fileName = "", float rehashFactor = 2.0);
  ~Fence();

  bool insert(unsigned indexA, unsigned indexB);
  bool present(unsigned indexA, unsigned indexB) const;

  void clear();

  unsigned size() const { return _indexN; }

  void read(const String& fileName);

  void write(const String& fileName) const;

  class Iterator {
  public:
    Iterator(const Fence& fence)
      : _list(fence._list), _next(_list) { }

    bool more() const {
      if (_next == _list && _next != NULL)
	return true;
      return (_next != NULL && _next->_next != NULL);
    }
    void operator++(int) { if (_next != NULL) _next = _next->_next; }
    const FenceIndex& index() const { return *_next; }

  private:
    FenceIndex*					_list;
    FenceIndex*					_next;
  };
  friend class Iterator;

private:
  void _rehash();
  unsigned _hash(unsigned stateA, unsigned stateB) const { return int(stateA + 337 * stateB) % int(_bucketN); }

  float						_rehashFactor;
  unsigned					_bucketN;
  unsigned					_bucketN2;
  unsigned					_indexN;

  FenceIndex**					_fence;
  FenceIndex*					_list;
};

typedef refcountable_ptr<Fence>			FencePtr;


// ----- definition for class `FenceFinder' -----
//
class FenceFinder {

  class _State {
  public:
    _State()
      : _nodeA(NULL), _nodeB(NULL), _next(NULL), _chain(NULL) { }
    _State(const WFSTFlyWeight::Node* nodeA, const WFSTFlyWeight::Node* nodeB, _State* next = NULL, _State* chain = NULL)
      : _nodeA(nodeA), _nodeB(nodeB), _next(next), _chain(chain) { }
    _State(const _State& state, _State* next = NULL, _State* chain = NULL)
      : _nodeA(state._nodeA), _nodeB(state._nodeB), _next(next), _chain(chain) { }

    bool operator<(const _State& rhs) const {
      return (_nodeA->index() < rhs._nodeA->index() ||
	      (_nodeA->index() == rhs._nodeA->index() && _nodeB->index() < rhs._nodeB->index()));
    }
    bool operator==(const _State& rhs) const { return (_nodeA->index() == rhs._nodeA->index() && _nodeB->index() == rhs._nodeB->index()); }

    const WFSTFlyWeight::Node* nodeA() const { return _nodeA; }
    const WFSTFlyWeight::Node* nodeB() const { return _nodeB; }

    void* operator new(size_t sz, void* mem = NULL) { return memoryManager().newElem(); }
    void  operator delete(void* e) { memoryManager().deleteElem(e); }

    static MemoryManager<_State>& memoryManager();

    class _Iterator {
    public:
      _Iterator(const _State& state)
	: _nodeA(NULL), _edgeA(state.nodeA()->_edges()),
	  _nodeB(state.nodeB()), _edgeB(NULL) { operator++(0); }
      void operator++(int);

      const WFSTFlyWeight::Node* nodeA() const { return _nodeA; }
      const WFSTFlyWeight::Node* nodeB() const { return _nodeB; }

      bool more() const { return (_nodeA != NULL); }

    private:
      const WFSTFlyWeight::Node*		_nodeA;
      const WFSTFlyWeight::Node*		_nodeB;

      const WFSTFlyWeight::Edge*		_edgeA;
      const WFSTFlyWeight::Edge*		_edgeB;
    };

    class _NonDeterministicIterator {
    public:
      _NonDeterministicIterator(const _State& state)
	: _nodeA(NULL), _edgeA(state.nodeA()->_edges()),
	  _nodeB(state.nodeB()), _edgeB(NULL), _firstMatchEdgeB(state.nodeB()->_edges()) { operator++(0); }
      void operator++(int);

      const WFSTFlyWeight::Node* nodeA() const { return _nodeA; }
      const WFSTFlyWeight::Node* nodeB() const { return _nodeB; }

      bool more() const { return (_nodeA != NULL); }

    private:
      const WFSTFlyWeight::Node*		_nodeA;
      const WFSTFlyWeight::Node*		_nodeB;

      const WFSTFlyWeight::Edge*		_edgeA;
      const WFSTFlyWeight::Edge*		_edgeB;
      const WFSTFlyWeight::Edge*		_firstMatchEdgeB;
    };

    // private:
    const WFSTFlyWeight::Node*			_nodeA;
    const WFSTFlyWeight::Node*			_nodeB;

    _State*					_next;
    _State*					_chain;
  };

  class _StateSet {

  public:
    _StateSet(unsigned bucketN = 4000001);
    ~_StateSet();

    bool insert(const _State& state);
    bool present(const _State& state) const;

    void clear();

    unsigned size() const { return _indexN; }

    class _Iterator {
    public:
      _Iterator(const _StateSet& stateSet)
	: _next(stateSet._list) { }

      bool more() const { _next->_next != NULL; }
      void operator++(int) { _next = _next->_next; }
      const _State& state() const { return *_next; }

    private:
      _State*					_next;
    };
    friend class _Iterator;

  private:
    void _rehash();
    unsigned _hash(const _State& state) const { return int(state.nodeA()->index() + 337 * state.nodeB()->index()) % int(_bucketN); }

    unsigned					_bucketN;
    unsigned					_bucketN2;
    unsigned					_indexN;

    _State**					_fence;
    _State*					_list;
  };

  typedef FirstInFirstOut<_State>		_StateQueue;

public:
  FenceFinder(const WFSTFlyWeightSortedOutputPtr& wfstA, const WFSTFlyWeightSortedInputPtr& wfstB, unsigned bucketN = 4000001);

  FencePtr fence();

private:
  void _findAccessibleNodes();
  void _findCoaccessibleNodes();
  FencePtr _findFence();

  const WFSTFlyWeightSortedOutputPtr		_wfstA;
        WFSTFlyWeightSortedOutputPtr		_wfstAReverse;

  const WFSTFlyWeightSortedInputPtr		_wfstB;
        WFSTFlyWeightSortedInputPtr		_wfstBReverse;

  _StateQueue					_stateQueue;
  _StateSet					_discoveredNodes;

  Fence						_endNodes;
  Fence						_accessibleNodes;
  Fence						_coaccessibleNodes;
};

FencePtr findFence(const WFSTFlyWeightSortedOutputPtr& wfstA, const WFSTFlyWeightSortedInputPtr& wfstB, unsigned bucketN = 4000001);

/*@}*/

#endif
