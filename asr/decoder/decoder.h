//                              -*- C++ -*-
//
//                              Millennium
//                   Distant Speech Recognition System
//                                (dsr)
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


#ifndef _decoder_h_
#define _decoder_h_

#include <typeinfo>
#include <map>
#include <algorithm>

#include "common/jpython_error.h"
#include "natural/natural.h"
#include "fsm/fsm.h"
#include "gaussian/distribBasic.h"
#include "path/distribPath.h"
#include "lattice/lattice.h"
#include "decoder/wfstFlyWeight.h"

/**
* \defgroup Decoder Viterbi Decoding
*/
/*@{*/

// ----- definition for class template `_TokenList' -----
//
template <class TokPtr>
class _TokenList : public Countable {
 public:
  class Iterator;	 friend class Iterator;
  class SortedIterator;	 friend class SortedIterator;

 private:
  class TokenHolder {
    friend class _TokenList;
    friend class Iterator;
  public:
    TokenHolder(unsigned stateX, const TokPtr& tok)
      : _stateA(stateX), _stateB(0), _tok(tok), _next(NULL), _chain(NULL) { }
    TokenHolder(unsigned stateA, unsigned stateB, const TokPtr& tok)
      : _stateA(stateA), _stateB(stateB), _tok(tok), _next(NULL), _chain(NULL) { }

    void* operator new(size_t sz) { return memoryManager().newElem(); }
    void  operator delete(void* e) { memoryManager().deleteElem(e); }

    static MemoryManager<TokenHolder>& memoryManager();

    unsigned				_stateA;
    unsigned				_stateB;
    const TokPtr			_tok;

    TokenHolder*			_next;
    TokenHolder*			_chain;
  };

 public:
  _TokenList(unsigned bucketN = 5000);
  ~_TokenList();

  void* operator new(size_t sz) { return memoryManager().newElem(); }
  void  operator delete(void* e) { memoryManager().deleteElem(e); }

  void clear();
  void insert(unsigned stateA, const TokPtr& tok, unsigned stateB = 0);
  void replace(unsigned stateA, const TokPtr& tok, unsigned stateB = 0);
  bool isPresent(unsigned stateA, unsigned stateB = 0) const { return _find(stateA, stateB) != NULL; }

  inline const TokPtr& token(unsigned stateA, unsigned stateB = 0) const;

  unsigned activeTokens() const { return _activeTokens; }

  static MemoryManager<_TokenList>& memoryManager();

 private:
  static list<TokenHolder**>& bucketList();

  TokenHolder** _getBuckets();

  unsigned _hash(unsigned stateA, unsigned stateB = 0) const { return int(stateA + 337 * stateB) % int(_bucketN); }
  inline TokenHolder* _find(unsigned stateA, unsigned stateB = 0) const;

  const unsigned			_bucketN;
  unsigned				_activeTokens;
  TokenHolder**				_token;
  TokenHolder*				_list;
  float					_minScore;
};


// ----- declare partial specializations of 'MemoryManager' -----
//
template<>
MemoryManager<_TokenList<refcountable_ptr<_Token<Inherit<WFSTransducer::Edge, refcountable_ptr<WFSAcceptor::Edge> > > > >::TokenHolder>& _TokenList<refcountable_ptr<_Token<Inherit<WFSTransducer::Edge, refcountable_ptr<WFSAcceptor::Edge> > > > >::TokenHolder::memoryManager();

template<>
MemoryManager<_TokenList<refcountable_ptr<_Token<WFSTFlyWeight::Edge*> > >::TokenHolder>&
_TokenList<refcountable_ptr<_Token<WFSTFlyWeight::Edge*> > >::TokenHolder::memoryManager();

template<>
MemoryManager<_TokenList<refcountable_ptr<_Token<WFSTFlyWeightSortedOutput::Edge*> > >::TokenHolder>&
_TokenList<refcountable_ptr<_Token<WFSTFlyWeightSortedOutput::Edge*> > >::TokenHolder::memoryManager();

template<>
MemoryManager<_TokenList<refcountable_ptr<_Token<Inherit<WFSTransducer::Edge, refcountable_ptr<WFSAcceptor::Edge> > > > > >&
_TokenList<refcountable_ptr<_Token<Inherit<WFSTransducer::Edge, refcountable_ptr<WFSAcceptor::Edge> > > > >::memoryManager();

template<>
MemoryManager<_TokenList<refcountable_ptr<_Token<WFSTFlyWeight::Edge*> > > >&
_TokenList<refcountable_ptr<_Token<WFSTFlyWeight::Edge*> > >::memoryManager();

template<>
MemoryManager<_TokenList<refcountable_ptr<_Token<WFSTFlyWeightSortedOutput::Edge*> > > >&
_TokenList<refcountable_ptr<_Token<WFSTFlyWeightSortedOutput::Edge*> > >::memoryManager();

template<>
MemoryManager<_TokenList<refcountable_ptr<_Token<WFSTFlyWeightSortedOutput::Edge*> > > >&
_TokenList<refcountable_ptr<_Token<WFSTFlyWeightSortedOutput::Edge*> > >::memoryManager();

template<>
list<_TokenList<refcountable_ptr<_Token<Inherit<WFSTransducer::Edge, refcountable_ptr<WFSAcceptor::Edge> > > > >::TokenHolder**>&
_TokenList<refcountable_ptr<_Token<Inherit<WFSTransducer::Edge, refcountable_ptr<WFSAcceptor::Edge> > > > >::bucketList();

template<>
list<_TokenList<refcountable_ptr<_Token<WFSTFlyWeight::Edge*> > >::TokenHolder**>&
_TokenList<refcountable_ptr<_Token<WFSTFlyWeight::Edge*> > >::bucketList();

template<>
list<_TokenList<refcountable_ptr<_Token<WFSTFlyWeightSortedOutput::Edge*> > >::TokenHolder**>&
_TokenList<refcountable_ptr<_Token<WFSTFlyWeightSortedOutput::Edge*> > >::bucketList();


// ----- methods for class template `_TokenList' -----
//
template <class TokPtr>
typename _TokenList<TokPtr>::TokenHolder* _TokenList<TokPtr>::_find(unsigned stateA, unsigned stateB) const
{
  unsigned hv = _hash(stateA, stateB);
  TokenHolder* holder = _token[hv];
  while(holder != NULL) {
    if (holder->_stateA == stateA && holder->_stateB == stateB) return holder;
    holder = holder->_chain;
  }
  return NULL;
}

template <class TokPtr>
const TokPtr& _TokenList<TokPtr>::token(unsigned stateA, unsigned stateB) const
{
  TokenHolder* holder = _find(stateA, stateB);

  if (holder == NULL)
    throw jindex_error("No token for state (%d x %d) is present.", stateA, stateB);

  return holder->_tok;
}

template <class TokPtr>
_TokenList<TokPtr>::_TokenList(unsigned bucketN)
  : _bucketN(bucketN), _activeTokens(0),
    _token(_getBuckets()), _list(NULL)
{
  clear();
}

template <class TokPtr>
typename _TokenList<TokPtr>::TokenHolder** _TokenList<TokPtr>::_getBuckets()
{
  if (_bucketN > 500) return new TokenHolder*[_bucketN];

  if (bucketList().size() > 0) {
    TokenHolder** buckets = *(bucketList().begin());
    bucketList().pop_front();

    return buckets;
  }

  return new TokenHolder*[_bucketN];
}

template <class TokPtr>
_TokenList<TokPtr>::~_TokenList()
{
  clear();

  if (_bucketN > 500) { delete[] _token;  return; }

  bucketList().push_front(_token);
}

template <class TokPtr>
void _TokenList<TokPtr>::clear()
{
  TokenHolder* holder;
  while ((holder = _list) != NULL) {
    _list = _list->_next;
    delete holder;
    _activeTokens--;
  }

  for (unsigned bucketX = 0; bucketX < _bucketN; bucketX++)
    _token[bucketX] = NULL;
}

template <class TokPtr>
void _TokenList<TokPtr>::replace(unsigned stateA, const TokPtr& tok, unsigned stateB)
{
  TokenHolder* holder = _find(stateA, stateB);

  if (holder == NULL)
    throw jkey_error("Could not find token for state (%d x %d)", stateA, stateB);

  Cast<TokPtr>(holder->_tok) = tok;
}

template <class TokPtr>
void _TokenList<TokPtr>::insert(unsigned stateA, const TokPtr& tok, unsigned stateB)
{
  unsigned     bucketX   = _hash(stateA, stateB);
  TokenHolder* holder    = _token[bucketX];
  TokenHolder* newHolder = new TokenHolder(stateA, stateB, tok);

  newHolder->_chain = holder;
  _token[bucketX]   = newHolder;

  newHolder->_next  = _list;
  _list             = newHolder;

  _activeTokens++;
}


// ----- definition for class `_TokenList::Iterator' -----
//
template <class TokPtr>
class _TokenList<TokPtr>::Iterator {
  friend class _TokenList<TokPtr>::SortedIterator;
 public:
  Iterator(_TokenList* tlist)
    : _itr(tlist->_list) { }

  const TokPtr& tok()  { return _itr->_tok; }
  const TokPtr& operator*()   { return _itr->_tok; }
  void operator++(int) { if (more()) _itr = _itr->_next; }
  bool more()          { return _itr != NULL; }

 private:
  TokenHolder*				_itr;
};


// ----- definition for class `_TokenList::SortedIterator' -----
//
template <class TokPtr>
class _TokenList<TokPtr>::SortedIterator {
  typedef vector<TokenHolder*>		_TokList;
  typedef typename _TokList::iterator	_Iterator;
  class LessThan {
  public:
    bool operator()(TokenHolder* h1, TokenHolder* h2) {
      return (h1->_tok->score() < h2->_tok->score());
    }
  };
 public:
  SortedIterator(_TokenList* tlist);

  const TokPtr& tok()       { return (*_itr)->_tok; }
  const TokPtr& operator*() { return (*_itr)->_tok; }
  void operator++(int)        { if (more()) _itr++; }
  bool more()                 { return _itr != _tokenList.end(); }

 private:
  _TokList				_tokenList;
  _Iterator				_itr;
};


// ----- methods for class `TokenList::SortedIterator' -----
//
template <class TokPtr>
_TokenList<TokPtr>::SortedIterator::SortedIterator(_TokenList* tlist)
  : _tokenList(tlist->activeTokens())
{
  unsigned i = 0;
  for (Iterator itr(tlist); itr.more(); itr++)
    _tokenList[i++] = itr._itr;

  if (i != tlist->activeTokens())
    throw jconsistency_error("No. of tokens (%d) should be %d", i, tlist->activeTokens());

  sort(_tokenList.begin(), _tokenList.begin() + i, LessThan());
  _itr = _tokenList.begin();

  /*
  printf("Token list:\n");
  for (_Iterator itr = _tokenList.begin(); itr != _tokenList.end(); itr++)
    (*itr)->_tok->write();
  fflush(stdout);
  */
}


// ----- definition for class `_Decoder' -----
//
template <class WFSType, class NodePtr, class EdgePtr, class WFSTypePtr>
class _Decoder : public Countable {
  // protected:
public:
  typedef _Token<EdgePtr>			Token;
  typedef refcountable_ptr<Token>		TokPtr;
  typedef _TokenList<TokPtr>			TokenList;
  typedef typename TokenList::Iterator		Iterator;
  typedef typename TokenList::SortedIterator	SortedIterator;

protected:
  typedef pair<unsigned, int>                   _LNIndex; 	// node, frame
  typedef map<_LNIndex, Lattice::NodePtr>       _LNMap;
  typedef _LNMap::value_type                    _LNType;
  typedef _LNMap::iterator                      _LNIterator;

  class _State {
  public:
    _State(StateId l, StateId r) : _l(l), _r(r) {}
    StateId l() const { return _l; }
    StateId r() const { return _r; }
    bool operator< (const _State &s) const {
      if (_l > s._l) return false;
      if (_l < s._l) return true;
      return (_r < s._r);
    }
    bool operator== (const _State &s) const { return (_l == s._l) && (_r == s._r); }

  private:
    StateId _l;
    StateId _r;
  };

  typedef set<_State>					_LatticeSet;

 public:
  _Decoder(DistribSetPtr& dist,
	   double beam = 100.0, double lmScale = 12.0, double lmPenalty = 0.0,
	   double silPenalty = 0.0, const String& silSymbol = "SIL-m", const String& eosSymbol = "</s>",
	   unsigned heapSize = 5000, unsigned topN = 0, double epsilon = 0.0, unsigned validEndN = 30,
	   bool generateLattice = true);
  virtual ~_Decoder();

  // decode current utterance
  virtual double decode(bool verbose = false);

  // specify beam width
  void setBeam(double beam) { _beam = beam; }

  // specify decoding network
  void _set(WFSTypePtr& wfst);

  // specify auxiliary decoding network
  void _setAB(refcountable_ptr<WFSType>& A, refcountable_ptr<WFSType>& B);

  // specify fence
  void _setFence(FencePtr& fence) { _fence = fence; }

  // back trace to get 1-best hypo
  String bestHypo(bool useInputSymbols = false);

  // back trace to get best distribPath
  DistribPathPtr bestPath();

  // number of final states for current utterance
  unsigned finalStatesN() const;

  // generate a lattice
  LatticePtr lattice();

  // set upper limit on memory allocated for tokens
  void setTokenMemoryLimit(unsigned limit) { Token::setMemoryLimit(limit); }

  // write 1-best hypo in CTM format
  virtual void writeCTM(const String& conv, const String& channel, const String& spk, const String& utt,
			double cfrom, double score, const String& fileName = "", double frameInterval = 0.01)
  { j_error("'writeCTM' is not supported in _Decoder base class template.\n"); }

  // write GMM labels for 1-best hypo in CTM format
  void writeGMM(const String& conv, const String& channel, const String& spk, const String& utt,
		double cfrom, double score, const String& fileName = "", double frameInterval = 0.01);

  // token reached the end state
  bool traceBackSucceeded() const;

  // return output lexicon
  virtual LexiconPtr& outputLexicon() const { return _wfst->outputLexicon(); }

  unsigned activeHypos() const { return _activeHypos; }

 protected:
  // helper methods for decoding
          void _swap(TokenList** curr, TokenList** next);
  virtual void _processFirstFrame();
  virtual void _processFrame();
  virtual void _newUtterance();

          void _placeOnList(const EdgePtr edge, double acScore, double lmScore, const TokPtr& thisToken);
          void _expandNode(const NodePtr node, const TokPtr& thisToken = NULL);
          void _expandNodeToEnd(const NodePtr node, const TokPtr& thisToken);
  virtual void _expandToEnd();

  // helper methods for traceback and lattice generation
  TokPtr _bestToken() const;

          void _majorTrace(LatticePtr& lattice, const TokPtr& start, _LNIndex prev);
  virtual void _minorTrace(LatticePtr& lattice, const TokPtr& start, _LNIndex prev);
  virtual void _clear();

  Lattice::NodePtr _findLNode(LatticePtr& lattice, _LNIndex lnode, bool* create = NULL);

  WFSTypePtr				_wfst;
  FencePtr				_fence;
  DistribSetPtr				_dist;

  	double				_beam;
  const unsigned			_topN;
  const double				_lmScale;
  const double				_lmPenalty;
  const double				_silPenalty;
  const String				_silSymbol;
  const String				_eosSymbol;
  unsigned				_silenceX;
  unsigned				_eosX;
  const bool				_generateLattice;

  double				_topScore;
  double				_topEndScore;
  double				_epsilon;
  unsigned				_validEndX;
  unsigned				_validEndN;
  int					_frameX;
  unsigned				_activeHypos;

  TokenList*				_current;
  TokenList*				_next;

  _LNMap				_latticeNodes;
  unsigned				_lnStateIndices;
  _LatticeSet				_latticeSet;
};


// ----- methods for class `_Decoder' -----
//
template <class WFSType, class NodePtr, class EdgePtr, class WFSTypePtr>
_Decoder<WFSType, NodePtr, EdgePtr, WFSTypePtr>::
_Decoder(DistribSetPtr& dist,
	 double beam, double lmScale, double lmPenalty, double silPenalty, const String& silSymbol, const String& eosSymbol,
	 unsigned heapSize, unsigned topN, double epsilon, unsigned validEndN, bool generateLattice)
  : _wfst(NULL), _fence(NULL), _dist(dist),
    _beam(beam), _topN(topN), _epsilon(epsilon), _validEndN(validEndN),
    _lmScale(lmScale), _lmPenalty(lmPenalty),
    _silPenalty(silPenalty), _silSymbol(silSymbol), _eosSymbol(eosSymbol),
    _generateLattice(generateLattice), _current(new TokenList(heapSize)), _next(new TokenList(heapSize)) { }

template <class WFSType, class NodePtr, class EdgePtr, class WFSTypePtr>
_Decoder<WFSType, NodePtr, EdgePtr, WFSTypePtr>::~_Decoder()
{
  delete _current;  delete _next;
}

template <class WFSType, class NodePtr, class EdgePtr, class WFSTypePtr>
void _Decoder<WFSType, NodePtr, EdgePtr, WFSTypePtr>::_newUtterance()
{
  _current->clear();  _next->clear();  _frameX = 0;  _activeHypos = 0;
  _dist->resetCache();  _dist->resetFeature();
}

template <class WFSType, class NodePtr, class EdgePtr, class WFSTypePtr>
void _Decoder<WFSType, NodePtr, EdgePtr, WFSTypePtr>::_swap(TokenList** curr, TokenList** next) {
  TokenList* temp = *curr;  *curr = *next;  *next = temp;
}

template <class WFSType, class NodePtr, class EdgePtr, class WFSTypePtr>
void _Decoder<WFSType, NodePtr, EdgePtr, WFSTypePtr>::_expandToEnd()
{
  _next->clear();
  for (typename TokenList::Iterator itr(_current); itr.more(); itr++) {
    const TokPtr& tok(itr.tok());

    if (tok->edge()->next()->isFinal()) {
      float lmScore = tok->lmScore() + _lmScale * float(tok->edge()->next()->cost());
      _placeOnList(tok->edge(), tok->acScore(), lmScore, tok->prev());
    }

    _expandNodeToEnd(tok->edge()->next(), tok);
  }
}

template <class WFSType, class NodePtr, class EdgePtr, class WFSTypePtr>
void _Decoder<WFSType, NodePtr, EdgePtr, WFSTypePtr>::
_placeOnList(const EdgePtr edge, double acScore, double lmScore, const TokPtr& thisToken)
{
  double ttlScore = acScore + lmScore;

  if ( ttlScore < _topScore && edge->input() != 0 ) _topScore = ttlScore;

  unsigned stateX = edge->next()->index();

  if (_next->isPresent(stateX)) {
    const TokPtr& nextToken(_next->token(stateX));

    if (ttlScore < nextToken->score()) {

      if (_generateLattice)
	_next->replace(stateX, TokPtr(new Token(acScore, lmScore, _frameX, edge, thisToken, nextToken)));
      else
	_next->replace(stateX, TokPtr(new Token(acScore, lmScore, _frameX, edge, thisToken)));	

    } else if (_generateLattice) {

      TokPtr wt(new Token(acScore, lmScore, _frameX, edge, thisToken, nextToken->worse()));
      nextToken->setWorse(wt);

    }

  } else {
    _next->insert(stateX, TokPtr(new Token(acScore, lmScore, _frameX, edge, thisToken)));
  }
}

template <class WFSType, class NodePtr, class EdgePtr, class WFSTypePtr>
void _Decoder<WFSType, NodePtr, EdgePtr, WFSTypePtr>::_processFirstFrame()
{
  printf(".");  fflush(stdout);

  _topScore = _topEndScore = HUGE;
  _frameX = 0;

  _expandNode(_wfst->initial());
  _activeHypos = _next->activeTokens();
}

template <class WFSType, class NodePtr, class EdgePtr, class WFSTypePtr>
void _Decoder<WFSType, NodePtr, EdgePtr, WFSTypePtr>::_processFrame()
{
  if (_frameX % 100 == 0) { printf(".");  fflush(stdout); }

  _swap(&_current, &_next);  _next->clear();

  double thresh = _topScore + _beam;
  _topScore = _topEndScore = HUGE;

  // printf("FrameX = %d : Active Tokens = %d\n", _frameX, _current->activeTokens());

  if (_topN > 0) {
    unsigned hypoCnt = 0;
    for (SortedIterator itr(_current); itr.more(); itr++) {
      if (hypoCnt == _topN) break;
      const TokPtr& tok(itr.tok());

      printf("Hypo %d : Score = %g\n", hypoCnt, tok->score());

      _expandNode(tok->edge()->next(), tok);
      hypoCnt++;
    }
  } else {
    for (Iterator itr(_current); itr.more(); itr++) {
      const TokPtr& tok(itr.tok());

      double score = tok->score();

      if (score > thresh) continue;

      _expandNode(tok->edge()->next(), tok);
    }
  }

  _activeHypos += _next->activeTokens();
}

// number of final states for current utterance
template <class WFSType, class NodePtr, class EdgePtr, class WFSTypePtr>
unsigned _Decoder<WFSType, NodePtr, EdgePtr, WFSTypePtr>::finalStatesN() const
{
  unsigned n = 0;
  for (typename TokenList::Iterator itr(_next); itr.more(); itr++) {
    const TokPtr& tok(*itr);
    const NodePtr node(tok->edge()->next());
    if (node->isFinal()) n++;
  }
  return n;
}

template <class WFSType, class NodePtr, class EdgePtr, class WFSTypePtr>
bool _Decoder<WFSType, NodePtr, EdgePtr, WFSTypePtr>::traceBackSucceeded() const
{
  // find the best token in an end state
  TokPtr   bestTok;
  double   bestScore = HUGE;
  unsigned cnt       = 0;
  for (typename TokenList::Iterator itr(_next); itr.more(); itr++) {
    const TokPtr& tok(itr.tok());
    const NodePtr node(tok->edge()->next());

    if (node->isFinal() == false)
      throw jconsistency_error("Node %d is not a final node.", node->index());

    cnt++;
    double score = tok->score();

    if (score < bestScore) {
      bestScore = score;
      bestTok = tok;
    }
  }

  bool traceBack = (bestTok.isNull() ? false : true);

  return traceBack;
}

template <class WFSType, class NodePtr, class EdgePtr, class WFSTypePtr>
typename _Decoder<WFSType, NodePtr, EdgePtr, WFSTypePtr>::TokPtr _Decoder<WFSType, NodePtr, EdgePtr, WFSTypePtr>::_bestToken() const
{
  // find the best token in an end state
  TokPtr   bestTok;
  double   bestScore = HUGE;
  unsigned cnt = 0;
  for (typename TokenList::Iterator itr(_next); itr.more(); itr++) {
    const TokPtr& tok(itr.tok());
    const NodePtr node(tok->edge()->next());

    if (node->isFinal() == false)
      throw jconsistency_error("Node %d is not a final node.", node->index());

    cnt++;
    double score = tok->score();

    if (score < bestScore) {
      bestScore = score;
      bestTok   = tok;
    }
  }

  // if no tokens have reached the end state, take the best hypothesis from '_current'
  if (bestTok.isNull()) {
    printf("Warning: No tokens reached the end state.\n");  fflush(stdout);
    cnt       = 0;
    bestScore = HUGE;

    for (typename TokenList::Iterator itr(_current); itr.more(); itr++) {
      const TokPtr& tok(itr.tok());

      /*
      cout << "Token from " << tok->edge()->prev()->index() << " to " << tok->edge()->next()->index() << " score " << tok->score() << endl;
      */

      cnt++;
      double score = tok->score();

      if (score < bestScore) {
	bestScore = score;
	bestTok = tok;
      }
    }
  }

  return bestTok;
}

template <class WFSType, class NodePtr, class EdgePtr, class WFSTypePtr>
double _Decoder<WFSType, NodePtr, EdgePtr, WFSTypePtr>::decode(bool verbose)
{
  _newUtterance();
  //try {
  _processFirstFrame();
  //} catch (...) {
  //throw j_error("No frames of speech ... returning from decode()\n");
  //  }

  try {
    while(true) {
      _frameX++;
      _processFrame();
    }
  } catch (j_error& e) {
    // this is a hack !!! A this point we only catch
    // j_error exceptions even if its subclasses are thrown
    if (e.getCode() == JPYTHON) {
      jpython_error *pe = static_cast<jpython_error*>(&e);
      throw jpython_error();
    }
  } catch (...) {

    Token::memoryUsage();
    throw;
  }
  printf("\n");

  _frameX--;
  _expandToEnd();
  TokPtr tok(_bestToken());

  double acScore  = tok->acScore();
  double lmScore  = tok->lmScore();
  double ttlScore = acScore + lmScore;

  if (verbose) {
    double avgHypos = double(_activeHypos) / _frameX;
    printf("\n");
    printf("Total Frames         = %d\n",     _frameX);
    printf("Acoustic score       = %10.4f\n", acScore);
    printf("Language model score = %10.4f\n", lmScore);
    printf("Total score          = %10.4f\n", ttlScore);
    printf("Average active hypos = %10.4f\n", avgHypos);
    fflush(stdout);
    Token::memoryUsage();
  }

  return ttlScore;
}

template <class WFSType, class NodePtr, class EdgePtr, class WFSTypePtr>
void _Decoder<WFSType, NodePtr, EdgePtr, WFSTypePtr>::_set(WFSTypePtr& wfst)
{
  _wfst     = wfst;
  _silenceX = wfst->inputLexicon()->index(_silSymbol);
  _eosX     = wfst->outputLexicon()->index(_eosSymbol);
}

template <class WFSType, class NodePtr, class EdgePtr, class WFSTypePtr>
String _Decoder<WFSType, NodePtr, EdgePtr, WFSTypePtr>::bestHypo(bool useInputSymbols)
{
  TokPtr bestTok = _bestToken();

  unsigned lastX = 0;
  String hypo = "";
  do {
    if (useInputSymbols) {
      unsigned inX = bestTok->edge()->input();

      if (inX != 0 && inX != lastX) {
	hypo  = _wfst->inputLexicon()->symbol(inX) + " " + hypo;
	lastX = inX;
      }
    } else {
      unsigned outX = bestTok->edge()->output();

      if (outX != 0)
	hypo = _wfst->outputLexicon()->symbol(outX) + " " + hypo;
    }
    bestTok = bestTok->prev();
    
  } while(bestTok.isNull() == false);

  return hypo;
}

template <class WFSType, class NodePtr, class EdgePtr, class WFSTypePtr>
DistribPathPtr _Decoder<WFSType, NodePtr, EdgePtr, WFSTypePtr>::bestPath()
{
  list<String> distNames;

  TokPtr bestTok = _bestToken();

  String hypo = "";
  do {
    unsigned inX = bestTok->edge()->input();

    if (inX != 0)
      distNames.push_front(_wfst->inputLexicon()->symbol(inX));

    bestTok = bestTok->prev();
    
  } while(bestTok.isNull() == false);

  DistribPathPtr ptr(new DistribPath(_dist, distNames));

  return ptr;
}

template <class WFSType, class NodePtr, class EdgePtr, class WFSTypePtr>
void _Decoder<WFSType, NodePtr, EdgePtr, WFSTypePtr>::_clear()
{
  _latticeNodes.clear();  _latticeSet.clear();
}

template <class WFSType, class NodePtr, class EdgePtr, class WFSTypePtr>
LatticePtr _Decoder<WFSType, NodePtr, EdgePtr, WFSTypePtr>::lattice()
{
  if (_generateLattice == false)
    throw jconsistency_error("Must enable lattice generation during decoding.");

  printf("Generating lattice with %d frames.\n", _frameX+1);  fflush(stdout);
  
  _clear();

  LexiconPtr stateLexicon(NULL);
  LexiconPtr outputLex(outputLexicon());
  LatticePtr lattice(new Lattice(stateLexicon, _wfst->inputLexicon(), outputLex));

  _lnStateIndices = 0;

  bool create = true;
  if (finalStatesN() > 0) {		// expand all final states

    for (typename TokenList::Iterator itr(_next); itr.more(); itr++) {
      const TokPtr& tok(*itr);
      const NodePtr node(tok->edge()->next());

      if (node->isFinal() == false) continue;

      _LNIndex initindex = make_pair(node->index(), _frameX+1);
      lattice->_addFinal(++_lnStateIndices);
      Lattice::NodePtr latNode(lattice->find(_lnStateIndices));

      _latticeNodes.insert(_LNType(initindex, latNode));

      _majorTrace(lattice, tok, initindex);
    }

  } else {				// no tokens reached a final state;
					// expand only best path overall

    TokPtr best(_bestToken());

    _LNIndex initIndex = make_pair(best->edge()->next()->index(), _frameX+1);
    Lattice::NodePtr latNode(lattice->find(++_lnStateIndices, create));
    _latticeNodes.insert(_LNType(initIndex, latNode));

    Lattice::NodePtr endNode(lattice->find(++_lnStateIndices, create));
    lattice->_addFinal(_lnStateIndices);

    LatticeEdgeData d(_frameX+1, _frameX+1, /* acScore= */ 0.0, /* lmScore= */ 0.0);
    Lattice::EdgePtr edgePtr(new Lattice::Edge(latNode, endNode, /* input= */ 0, _eosX, d));
    latNode->addEdgeForce(edgePtr);

    _majorTrace(lattice, best, initIndex);
  }

  printf("Finished lattice generation.\n");  fflush(stdout);

  return lattice;
}

// trace along all tokens for this frame
template <class WFSType, class NodePtr, class EdgePtr, class WFSTypePtr>
void _Decoder<WFSType, NodePtr, EdgePtr, WFSTypePtr>::_majorTrace(LatticePtr& lattice, const TokPtr& start, _LNIndex prev)
{
  TokPtr tok = start;
  while ( tok.isNull() == false ) {
    _minorTrace(lattice, tok, prev);
    tok = tok->worse();
  }
}

// trace along prev pointers of tokens until input symbol changes
template <class WFSType, class NodePtr, class EdgePtr, class WFSTypePtr>
void _Decoder<WFSType, NodePtr, EdgePtr, WFSTypePtr>::_minorTrace(LatticePtr& lattice, const TokPtr& endTok, _LNIndex prev)
{
  TokPtr tok             = endTok;
  int    endFrame        = endTok->frameX();

  unsigned currentInput  = tok->edge()->input();
  unsigned currentOutput = tok->edge()->output();

  // trace backwards until input or output symbol changes
  while ( tok->prev().isNull() == false ) {

    if (tok->prev()->edge()->output() != 0 && currentOutput != 0) break;

    if (tok->prev()->edge()->input() != 0) {
      if ( currentInput != 0 && tok->prev()->edge()->input() != currentInput ) break;
      currentInput = tok->prev()->edge()->input();
    }

    if (tok->prev()->edge()->output() != 0) {
      if (currentOutput != 0) break;
      currentOutput = tok->prev()->edge()->output();
    }

    tok = tok->prev();
  }
  int begFrame = tok->frameX();
  const TokPtr& prevTok(tok->prev());

  // create or find the correct lattice node
  Lattice::NodePtr endNode(_findLNode(lattice, prev));

  // create link between nodes
  _LNIndex lnode;
  Lattice::NodePtr begNode;
  bool   create  = true;
  double acScore = endTok->acScore();
  double lmScore = endTok->lmScore();
  if (prevTok.isNull()) {
    begNode = lattice->initial();
  } else {
    acScore -= prevTok->acScore();
    lmScore -= prevTok->lmScore();

    lnode    = make_pair(tok->edge()->prev()->index(), begFrame);
    begNode  = _findLNode(lattice, lnode, &create);
  }
  if (currentInput  == _silenceX) lmScore -= (_lmScale * _silPenalty);
  if (currentOutput != 0) lmScore -= (_lmScale * _lmPenalty);
  lmScore /= _lmScale;

  LatticeEdgeData d(begFrame, endFrame, acScore, lmScore);
  Lattice::EdgePtr edgePtr(new Lattice::Edge(begNode, endNode, currentInput, currentOutput, d));
  begNode->addEdgeForce(edgePtr);

  // recursive call to continue traceback
  if (prevTok.isNull() == false && create)
    _majorTrace(lattice, prevTok, lnode);
}

// check if node is present in lattice, if not create it 
template <class WFSType, class NodePtr, class EdgePtr, class WFSTypePtr>
Lattice::NodePtr _Decoder<WFSType, NodePtr, EdgePtr, WFSTypePtr>::_findLNode(LatticePtr& lattice, _LNIndex lnode, bool* create)
{
  _LNIterator lniter = _latticeNodes.find(lnode);

  if (lniter != _latticeNodes.end()) {
    if (create) *create = false;
    return (*lniter).second;
  }

  if (*create == false)
    throw jconsistency_error("Could not find node %d at frame %d.\n",
			     lnode.first, lnode.second);

  Lattice::NodePtr newNode = lattice->find(++_lnStateIndices, *create);
  _latticeNodes.insert(_LNType(lnode, newNode));

  return newNode;
}

template <class WFSType, class NodePtr, class EdgePtr, class WFSTypePtr>
void _Decoder<WFSType, NodePtr, EdgePtr, WFSTypePtr>::_expandNode(const NodePtr node, const TokPtr& thisToken)
{
  double acScoreNode = 0.0;
  double lmScoreNode = 0.0;

  if (thisToken.isNull() == false) {
    acScoreNode = thisToken->acScore();
    lmScoreNode = thisToken->lmScore();
  }

  for (typename WFSType::Node::Iterator itr(_wfst, Cast<NodePtr>(node)); itr.more(); itr++) {
    const EdgePtr edge(Cast<EdgePtr>(itr.edge()));

    unsigned distX   = edge->input();
    double   lmScore = lmScoreNode + _lmScale * float(edge->cost());

    // apply word insertion penalty
    if (edge->output() != 0) lmScore += (_lmScale * _lmPenalty);

    // apply silence insertion penalty
    if (edge->input() == _silenceX && (thisToken.isNull() || thisToken->edge()->input() != _silenceX))
      lmScore += (_lmScale * _silPenalty);

    if (distX == 0) {
      // cout << "Expanding node " << edge->next()->index() << endl;
      _expandNode(edge->next(), TokPtr(new Token(acScoreNode, lmScore, _frameX, edge, thisToken)));
      continue;
    }

    double acScore = acScoreNode + _dist->find(distX-1)->score(_frameX);

    _placeOnList(edge, acScore, lmScore, thisToken);
  }
}

template <class WFSType, class NodePtr, class EdgePtr, class WFSTypePtr>
void _Decoder<WFSType, NodePtr, EdgePtr, WFSTypePtr>::_expandNodeToEnd(const NodePtr node, const TokPtr& thisToken)
{
  double acScoreNode = thisToken->acScore();
  double lmScoreNode = thisToken->lmScore();

  for (typename WFSType::Node::Iterator itr(_wfst, Cast<NodePtr>(node)); itr.more(); itr++) {
    const EdgePtr edge(Cast<EdgePtr>(itr.edge()));

    if (edge->input() != 0) continue;

    double lmScore = lmScoreNode + _lmScale * float(edge->cost());

    // apply word insertion penalty
    if (edge->output() != 0) lmScore += (_lmScale * _lmPenalty);

    // apply silence insertion penalty
    if (edge->input() == _silenceX && (thisToken.isNull() || thisToken->edge()->input() != _silenceX))
      lmScore += (_lmScale * _silPenalty);

    if (edge->next()->isFinal())
      _placeOnList(edge, acScoreNode, lmScore + _lmScale * float(node->cost()), thisToken);
    _expandNodeToEnd(edge->next(), TokPtr(new Token(acScoreNode, lmScore, _frameX, edge, thisToken)));
  }
}

// write 1-best hypo in GMM format
template <class WFSType, class NodePtr, class EdgePtr, class WFSTypePtr>
void _Decoder<WFSType, NodePtr, EdgePtr, WFSTypePtr>::
writeGMM(const String& conv, const String& channel, const String& spk, const String& utt,
	 double cfrom, double score, const String& fileName, double frameInterval)
{
  FILE* fp = (fileName == "") ? stdout : fileOpen(fileName, "a");

  fprintf(fp, "# %s %10.4f %10.4f\n", utt.c_str(), cfrom, score);

  TokPtr   tok    = _bestToken();
  EdgePtr& edge(Cast<EdgePtr>(tok->edge()));
  // double   wscore = tok->score();
  int endX;
  double wscore;

  vector<String> gmms;
  vector<double> starts;
  vector<double> durations;
  vector<double> scores;

  // move past null arcs
  TokPtr nextTok = tok;
  unsigned thisX = tok->edge()->input();
  while (tok.isNull() == false && thisX == 0) {
    nextTok = tok;
    TokPtr tmp = tok->prev(); tok = tmp;
    if (tok.isNull() == false)
      thisX = tok->edge()->input();
  }
  wscore = nextTok->acScore();

  if (tok.isNull() == false)
    endX = tok->frameX();

  while(tok.isNull() == false) {

    // find all arcs with the same GMM label
    nextTok = tok;
    unsigned inX = tok->edge()->input();
    while (tok.isNull() == false && inX == thisX) {
      nextTok = tok;
      TokPtr tmp = tok->prev(); tok = tmp;
      if (tok.isNull() == false)
	inX = tok->edge()->input();
    }

    int           startX = nextTok->frameX();
    double        beg    = cfrom + startX * frameInterval;
    double        len    = (endX - startX + 1) * frameInterval;
    const String& label  = _wfst->inputLexicon()->symbol(thisX);

    gmms.push_back(label);
    starts.push_back(beg);
    durations.push_back(len);
    // double oscore = (tok->prev().isNull()) ? 0.0 : tok->prev()->score();
    double oscore = (tok.isNull() || tok->prev().isNull()) ? 0.0 : tok->prev()->acScore();
    scores.push_back(wscore - oscore);

    // wscore = tok->score();
    wscore = nextTok->score();

    // move past null arcs
    thisX = inX;
    while (tok.isNull() == false && thisX == 0) {
      TokPtr tmp = tok->prev(); tok = tmp;
      if (tok.isNull() == false)
	thisX = tok->edge()->input();
    }
    if (tok.isNull() == false)
      endX = tok->frameX();
  }

  /*
  fprintf(fp, "%s %s %7.2f %7.2f %-20s %7.2f\n",
	  conv.c_str(), channel.c_str(), cfrom, starts[0] - cfrom, "<s>", wscore);
  */
  for (int i = gmms.size() - 1; i >= 0; i--) {
    const String& label(gmms[i]);
    if (label == _eosSymbol) continue;
    fprintf(fp, "%s %s %7.2f %7.2f %-20s %7.2f\n",
	    conv.c_str(), channel.c_str(), starts[i], durations[i], label.c_str(), scores[i]);
  }

  if (fp != stdout) fileClose(fileName, fp);
}


// ----- definition for class `Decoder' -----
//
class Decoder : public _Decoder<WFSTransducer, WFSTransducer::NodePtr, WFSTransducer::EdgePtr, WFSTransducerPtr> {
 public:
  Decoder(DistribSetPtr& dist,
	  double beam = 100.0, double lmScale = 12.0, double lmPenalty = 0.0,
	  double silPenalty = 0.0, const String& silSymbol = "SIL-m", const String& eosSymbol = "</s>",
	  unsigned heapSize = 5000, unsigned topN = 0, bool generateLattice = true)
    : _Decoder<WFSTransducer, WFSTransducer::NodePtr, WFSTransducer::EdgePtr, WFSTransducerPtr>(dist, beam, lmScale, lmPenalty, silPenalty,
												silSymbol, eosSymbol, heapSize, topN, generateLattice) { }

  // specify decoding network
  void set(WFSTransducerPtr& wfst) { _set(wfst); }
};

typedef refcountable_ptr<Decoder> DecoderPtr;


typedef WFSTFlyWeight::Node* WFSTFlyWeightNodePtr;

// ----- definition for class `DecoderFlyWeight' -----
//
class DecoderFlyWeight : public _Decoder<WFSTFlyWeight, WFSTFlyWeight::Node*, WFSTFlyWeight::Edge*, WFSTFlyWeightPtr> {
 public:
  DecoderFlyWeight(DistribSetPtr& dist,
		   double beam = 100.0, double lmScale = 12.0, double lmPenalty = 0.0,
		   double silPenalty = 0.0, const String& silSymbol = "SIL-m", const String& eosSymbol = "</s>",
		   unsigned heapSize = 5000, unsigned topN = 0, bool generateLattice = true)
    : _Decoder<WFSTFlyWeight, WFSTFlyWeightNodePtr,
	       WFSTFlyWeight::Edge*, WFSTFlyWeightPtr>(dist, beam, lmScale, lmPenalty, silPenalty,
						       silSymbol, eosSymbol, heapSize, topN, generateLattice) { }

  // specify decoding network
  void set(WFSTFlyWeightPtr& wfst) { _set(Cast<refcountable_ptr<WFSTFlyWeight> >(wfst)); }
};

typedef refcountable_ptr<DecoderFlyWeight> DecoderFlyWeightPtr;


// ----- definition for class `DecoderWordTrace' -----
//
class DecoderWordTrace : public _Decoder<WFSTFlyWeightSortedOutput, WFSTFlyWeightSortedOutput::Node*,
					 WFSTFlyWeightSortedOutput::Edge*, WFSTFlyWeightSortedOutputPtr> {
protected:
  typedef _Decoder<WFSTFlyWeightSortedOutput, WFSTFlyWeightSortedOutput::Node*,
		   WFSTFlyWeightSortedOutput::Edge*, WFSTFlyWeightSortedOutputPtr> BaseDecoder;

  class WordTrace;
  typedef refcountable_ptr<WordTrace> WordTracePtr;

  class Token;
  typedef Inherit<Token, BaseDecoder::TokPtr > TokPtr;

  class WordTrace : public Countable {
  public:
    WordTrace(unsigned wordX, unsigned wordSequenceX, int endX, const TokPtr& tokenList = NULL)
      : _wordX(wordX), _wordSequenceX(wordSequenceX), _endX(endX), _tokenList(tokenList) { }

    unsigned wordX()         const { return _wordX;         }
    unsigned wordSequenceX() const { return _wordSequenceX; }
    int      endX()          const { return _endX;          }

    const TokPtr& tokenList() const { return _tokenList; }

    static void memoryUsage() { 
      memoryManager().report();
    }

    void* operator new(size_t sz) { return memoryManager().newElem(); }
    void  operator delete(void* e) { memoryManager().deleteElem(e); }

  private:
    static MemoryManager<WordTrace>& memoryManager();

    const unsigned			_wordX;
    const unsigned			_wordSequenceX;
    const int				_endX;
    TokPtr				_tokenList;
  };

  class Token : public _Token<WFSTFlyWeightSortedOutput::Edge*> {
  public:
    Token(double acs, double lms, int frameX, const WFSTFlyWeightSortedOutput::Edge* edgeA)
      : _Token<WFSTFlyWeightSortedOutput::Edge*>(acs, lms, frameX, Cast<WFSTFlyWeightSortedOutput::Edge*>(edgeA), NULL, NULL),
	_wordTrace(NULL) { }

    Token(double acs, double lms, int frameX, const WFSTFlyWeightSortedOutput::Edge* edgeA, const WordTracePtr& wordTrace,
	  const TokPtr& worse)
      : _Token<WFSTFlyWeightSortedOutput::Edge*>(acs, lms, frameX, Cast<WFSTFlyWeightSortedOutput::Edge*>(edgeA), NULL, worse),
	_wordTrace(wordTrace) { }

    Token(double acs, double lms, int frameX, const WFSTFlyWeightSortedOutput::Edge* edgeA, const WordTracePtr& wordTrace)
      : _Token<WFSTFlyWeightSortedOutput::Edge*>(acs, lms, frameX, Cast<WFSTFlyWeightSortedOutput::Edge*>(edgeA)),
	_wordTrace(wordTrace) { }

    void* operator new(size_t sz) { return memoryManager().newElem(); }
    void  operator delete(void* e) { memoryManager().deleteElem(e); }

    const WordTracePtr&                wordTrace()    const { return _wordTrace; }
    const TokPtr&		       worse()        const { return Cast<TokPtr>(_worseList); }

  private:
    static MemoryManager<Token>& memoryManager();

    WordTracePtr				_wordTrace;
  };

  typedef _TokenList<TokPtr>			TokenList;
  typedef TokenList::Iterator			Iterator;
  typedef TokenList::SortedIterator		SortedIterator;

  typedef pair<unsigned, int>                   _LNIndex; 	// node, frame
  typedef map<_LNIndex, Lattice::Node*>		_LNMap;
  typedef _LNMap::value_type                    _LNType;
  typedef _LNMap::iterator                      _LNIterator;

  typedef pair<unsigned, unsigned>		_StateIndex; 	// nodeA, nodeB
  typedef map<_StateIndex, unsigned>		_StateIndexMap;
  typedef _StateIndexMap::value_type		_StateIndexMapType;
  typedef _StateIndexMap::iterator		_StateIndexMapIterator;

public:
  DecoderWordTrace(DistribSetPtr& dist,
		   double beam = 100.0, double lmScale = 12.0, double lmPenalty = 0.0,
		   double silPenalty = 0.0, const String& silSymbol = "SIL-m", const String& eosSymbol = "</s>",
		   unsigned heapSize = 5000, unsigned topN = 0, double epsilon = 0.0, unsigned validEndN = 30,
		   bool generateLattice = true, unsigned propagateN = 5,
		   bool fastHash = false, bool insertSilence = false);
  virtual ~DecoderWordTrace() { }

  void set(WFSTFlyWeightSortedOutputPtr& wfst) { _set(wfst); }

 protected:
  // these declarations introduced to make gcc 4.0.2 happy
  BaseDecoder::_findLNode;

  BaseDecoder::_wfst;
  BaseDecoder::_dist;

  BaseDecoder::_beam;
  BaseDecoder::_topN;
  BaseDecoder::_lmScale;
  BaseDecoder::_lmPenalty;
  BaseDecoder::_silPenalty;
  BaseDecoder::_silSymbol;
  BaseDecoder::_eosSymbol;
  BaseDecoder::_silenceX;
  BaseDecoder::_eosX;
  BaseDecoder::_generateLattice;

  BaseDecoder::_topScore;
  BaseDecoder::_topEndScore;
  BaseDecoder::_frameX;
  BaseDecoder::_activeHypos;

  BaseDecoder::_current;
  BaseDecoder::_next;

  BaseDecoder::_latticeNodes;
  BaseDecoder::_lnStateIndices;
  BaseDecoder::_latticeSet;

  // new declarations
  typedef List<String>			_WordSequenceHash;
  typedef vector<unsigned>		_UniqueIndices;

  virtual void _newUtterance();
  virtual void _processFirstFrame();
  virtual void _processFrame();

  inline  bool _notPresent(unsigned wordSeqX);
          void _placeOnList(const WFSTFlyWeightSortedOutput::Edge* edge, TokPtr tok);
          void _expandNode(const WFSTFlyWeightSortedOutput::Node* node, const TokPtr& thisToken = NULL);
          void _expandNodeToEnd(const WFSTFlyWeightSortedOutput::Node* node,
				const TokPtr& thisToken);
  virtual void _expandToEnd();

  virtual void _minorTrace(LatticePtr& lattice, const BaseDecoder::TokPtr& start, _LNIndex prev);
  virtual void _clear();

  inline unsigned _hashWordSequence(const TokPtr& thisToken, unsigned wordX);
  TokPtr _advanceTokens(const WFSTFlyWeightSortedOutput::Edge* edgeA, double acScoreEdge, double lmScoreEdge, const TokPtr& topToken);

  const unsigned			_propagateN;
  const bool				_fastHash;
  const bool				_insertSilence;

  _WordSequenceHash			_wordSequenceHash;
  _UniqueIndices			_uniqueIndices;
  unsigned				_tokenX;
  _StateIndexMap			_stateIndices;
};

template<>
MemoryManager<_Token<WFSTFlyWeight::Edge*> >&
_Token<WFSTFlyWeight::Edge*>::memoryManager();

typedef refcountable_ptr<DecoderWordTrace> DecoderWordTracePtr;


// ----- definition for class `DecoderFastComposition' -----
//
class DecoderFastComposition : public DecoderWordTrace {
protected:
  class Token : public DecoderWordTrace::Token {
  public:
    Token(double acs, double lms, int frameX, const WFSTFlyWeightSortedOutput::Edge* edgeA,
	  const WordTracePtr& wordTrace, const TokPtr& worse)
      : DecoderWordTrace::Token(acs, lms, frameX, edgeA, wordTrace, worse), _pushedWeight(0.0), _edgeB(NULL) { }

    Token(double acs, double lms, int frameX, const WFSTFlyWeightSortedOutput::Edge* edgeA,
	  const WFSTFlyWeightSortedInput::Edge* edgeB)
      : DecoderWordTrace::Token(acs, lms, frameX, edgeA, NULL), _pushedWeight(0.0), _edgeB(edgeB) { }

    Token(double acs, double lms, int frameX, const WFSTFlyWeightSortedOutput::Edge* edgeA,
	  const WordTracePtr& wordTrace, const WFSTFlyWeightSortedInput::Edge* edgeB)
      : DecoderWordTrace::Token(acs, lms, frameX, edgeA, wordTrace),
	_pushedWeight(0.0), _edgeB(edgeB) { }

    void* operator new(size_t sz) { return memoryManager().newElem(); }
    void  operator delete(void* e) { memoryManager().deleteElem(e); }

    float pushedWeight() const { return _pushedWeight; }
    void setPushedWeight(float wgt) { _pushedWeight = wgt; }
    const WFSTFlyWeightSortedInput::Edge* edgeB() { return _edgeB; }

    const TokPtr&		       worse()        const { return Cast<TokPtr>(_worseList); }

  private:
    static MemoryManager<Token>& memoryManager();

    const WFSTFlyWeightSortedInput::Edge*	_edgeB;
    float					_pushedWeight;
  };

  typedef Inherit<Token, DecoderWordTrace::TokPtr> TokPtr;

  class _WeightPusher : public Countable {
  public:
    class _WeightIndex {
      friend class _WeightPusher;
    public:
      _WeightIndex(unsigned indexA, unsigned indexB, float weight, int frameX, _WeightIndex* chain)
	: _indexA(indexA), _indexB(indexB), _weight(weight), _frameX(frameX), _chain(chain) { }
    
      void* operator new(size_t sz) { return memoryManager().newElem(); }
      void  operator delete(void* e) { memoryManager().deleteElem(e); }

      static MemoryManager<_WeightIndex>& memoryManager();

      unsigned indexA() const { return _indexA; }
      unsigned indexB() const { return _indexB; }

    private:
      const unsigned			_indexA;
      const unsigned			_indexB;
      const float			_weight;
            int				_frameX;

      _WeightIndex*			_chain;
    };

    _WeightPusher(unsigned bucketN = 4000001, int pastFramesN = 40);
    ~_WeightPusher();

    float push(const WFSTFlyWeightSortedOutput::Edge* edgeA, const WFSTFlyWeightSortedInput::Edge* edgeB);
    void increment() { _deleteX++; _frameX++; }
    void clear();

  private:
    unsigned _hash(unsigned stateA, unsigned stateB) const { return int(stateA + 337 * stateB) % int(_binN); }

    unsigned				_binN;
    _WeightIndex**			_index;
    int					_frameX;
    int					_deleteX;
    int					_pastFramesN;
  };

  typedef refcountable_ptr<_WeightPusher> _WeightPusherPtr;

public:
  DecoderFastComposition(DistribSetPtr& dist,
			 double beam = 100.0, double lmScale = 12.0, double lmPenalty = 0.0,
			 double silPenalty = 0.0, const String& silSymbol = "SIL-m", const String& eosSymbol = "</s>",
			 unsigned heapSize = 5000, unsigned topN = 0, bool generateLattice = true,
			 unsigned propagateN = 5, bool fastHash = false, bool insertSilence = false,
			 bool pushWeights = false);

  // specify decoding network
  void setAB(WFSTFlyWeightSortedOutputPtr& A, WFSTFlyWeightSortedInputPtr& B);

  // specify fence
  void setFence(FencePtr& fence) { _setFence(fence); }

  // decode current utterance
  virtual double decode(bool verbose = false);

  // return output lexicon
  virtual LexiconPtr& outputLexicon() const { return _wfstB->outputLexicon(); }

  void writeCTM(const String& conv, const String& channel, const String& spk, const String& utt,
		double cfrom, double score, const String& fileName = "", double frameInterval = 0.01);

private:
  virtual void _newUtterance();
  virtual void _processFirstFrame();
  virtual void _processFrame();
          void _placeOnList(const WFSTFlyWeightSortedOutput::Edge* edgeA, const WFSTFlyWeightSortedInput::Edge* edgeB, TokPtr tok);
  virtual void _expandToEnd();
          void _expandNode(const WFSTFlyWeightSortedOutput::Node* node, const TokPtr& topToken = NULL);
          void _expandNodeToEnd(const WFSTFlyWeightSortedOutput::Node* node, const TokPtr& topToken);

  virtual void _minorTrace(LatticePtr& lattice, const BaseDecoder::TokPtr& start, _LNIndex prev);
  unsigned _stateIndex(const TokPtr& token);
  TokPtr _advanceTokens(const WFSTFlyWeightSortedOutput::Edge* edgeA, double acScoreEdge, double lmScoreEdge, const TokPtr& topToken,
			const WFSTFlyWeightSortedInput::Edge* edgeB = NULL);

  bool						_pushWeights;
  WFSTFlyWeightSortedInputPtr			_wfstB;
  _WeightPusherPtr				_weightPusher;
};

typedef Inherit<DecoderFastComposition, DecoderWordTracePtr> DecoderFastCompositionPtr;

/*@}*/

#endif
