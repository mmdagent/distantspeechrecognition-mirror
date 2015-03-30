//
//                               Millennium
//                   Distant Speech Recognition System
//                                 (dsr)
//
//  Module:  dsr.decoder
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


#include <math.h>
#include "decoder/decoder.h"

#include <typeinfo>
#include <iostream>

using namespace std;


// ----- partial specializations for `MemoryManager' class template -----
//
template<>
MemoryManager<_Token<WFSTFlyWeight::Edge*> >&
_Token<WFSTFlyWeight::Edge*>::memoryManager()
{
  static MemoryManager<_Token<WFSTFlyWeight::Edge*> > _MemoryManager("_Token<WFSTFlyWeight::Edge*>");
  return _MemoryManager;
}

template<>
MemoryManager<_Token<WFSTFlyWeightSortedOutput::Edge*> >&
_Token<WFSTFlyWeightSortedOutput::Edge*>::memoryManager()
{
  static MemoryManager<_Token<WFSTFlyWeightSortedOutput::Edge*> > _MemoryManager("_Token<WFSTFlyWeightSortedOutput::Edge*>");
  return _MemoryManager;
}

template<>
MemoryManager<_TokenList<refcountable_ptr<_Token<Inherit<WFSTransducer::Edge, refcountable_ptr<WFSAcceptor::Edge> > > > >::TokenHolder>& _TokenList<refcountable_ptr<_Token<Inherit<WFSTransducer::Edge, refcountable_ptr<WFSAcceptor::Edge> > > > >::TokenHolder::memoryManager()
{
  static MemoryManager<TokenHolder> _MemoryManager("_TokenList<WFSTransducer::EdgePtr>::TokenHolder");
  return _MemoryManager;
}

template<>
MemoryManager<_TokenList<refcountable_ptr<_Token<WFSTFlyWeight::Edge*> > >::TokenHolder>&
_TokenList<refcountable_ptr<_Token<WFSTFlyWeight::Edge*> > >::TokenHolder::memoryManager()
{
  static MemoryManager<TokenHolder> _MemoryManager("_TokenList<WFSTFlyWeight::EdgePtr>::TokenHolder");
  return _MemoryManager;
}

template<>
MemoryManager<_TokenList<refcountable_ptr<_Token<WFSTFlyWeightSortedOutput::Edge*> > >::TokenHolder>&
_TokenList<refcountable_ptr<_Token<WFSTFlyWeightSortedOutput::Edge*> > >::TokenHolder::memoryManager()
{
  static MemoryManager<TokenHolder> _MemoryManager("_TokenList<WFSTFlyWeightSortedOutput::EdgePtr>::TokenHolder");
  return _MemoryManager;
}

template<>
MemoryManager<_TokenList<refcountable_ptr<_Token<Inherit<WFSTransducer::Edge, refcountable_ptr<WFSAcceptor::Edge> > > > > >&
_TokenList<refcountable_ptr<_Token<Inherit<WFSTransducer::Edge, refcountable_ptr<WFSAcceptor::Edge> > > > >::memoryManager()
{
  static MemoryManager<_TokenList> _MemoryManager("_TokenList<refcountable_ptr<_Token<Inherit<WFSTransducer::Edge, refcountable_ptr<WFSAcceptor::Edge> > > >");
  return _MemoryManager;
}

template<>
MemoryManager<_TokenList<refcountable_ptr<_Token<WFSTFlyWeight::Edge*> > > >&
_TokenList<refcountable_ptr<_Token<WFSTFlyWeight::Edge*> > >::memoryManager()
{
  static MemoryManager<_TokenList> _MemoryManager("_TokenList<WFSTFlyWeight::Edge*>");
  return _MemoryManager;
}

template<>
MemoryManager<_TokenList<refcountable_ptr<_Token<WFSTFlyWeightSortedOutput::Edge*> > > >&
_TokenList<refcountable_ptr<_Token<WFSTFlyWeightSortedOutput::Edge*> > >::memoryManager()
{
  static MemoryManager<_TokenList> _MemoryManager("_TokenList<WFSTFlyWeightSortedOutput::Edge*>");
  return _MemoryManager;
}

template<>
list<_TokenList<refcountable_ptr<_Token<Inherit<WFSTransducer::Edge, refcountable_ptr<WFSAcceptor::Edge> > > > >::TokenHolder**>&
_TokenList<refcountable_ptr<_Token<Inherit<WFSTransducer::Edge, refcountable_ptr<WFSAcceptor::Edge> > > > >::bucketList()
{
  static list<TokenHolder**> _bucketList;
  return _bucketList;
}

template<>
list<_TokenList<refcountable_ptr<_Token<WFSTFlyWeight::Edge*> > >::TokenHolder**>&
_TokenList<refcountable_ptr<_Token<WFSTFlyWeight::Edge*> > >::bucketList()
{
  static list<TokenHolder**> _bucketList;
  return _bucketList;
}

template<>
list<_TokenList<refcountable_ptr<_Token<WFSTFlyWeightSortedOutput::Edge*> > >::TokenHolder**>&
_TokenList<refcountable_ptr<_Token<WFSTFlyWeightSortedOutput::Edge*> > >::bucketList()
{
  static list<TokenHolder**> _bucketList;
  return _bucketList;
}


// ----- methods for class `DecoderWordTrace' -----
//
DecoderWordTrace::
DecoderWordTrace(DistribSetPtr& dist,
		 double beam, double lmScale, double lmPenalty, double silPenalty, const String& silSymbol, const String& eosSymbol,
		 unsigned heapSize, unsigned topN, double epsilon, unsigned validEndN, bool generateLattice, unsigned propagateN, bool fastHash, bool insertSilence)
  : _Decoder<WFSTFlyWeightSortedOutput, WFSTFlyWeightSortedOutput::Node*,
	     WFSTFlyWeightSortedOutput::Edge*, WFSTFlyWeightSortedOutputPtr>(dist, beam, lmScale, lmPenalty, silPenalty, silSymbol,
									     eosSymbol, heapSize, topN, epsilon, validEndN, generateLattice),
    _propagateN(propagateN), _fastHash(fastHash), _insertSilence(insertSilence),
    _wordSequenceHash("Word Sequence Hash"), _uniqueIndices(_propagateN) { }

void DecoderWordTrace::_processFirstFrame()
{
  printf(".");  fflush(stdout);

  _topScore = HUGE;
  _frameX = _validEndX = 0;

  _expandNode(_wfst->initial());
  _activeHypos = _next->activeTokens();
}

void DecoderWordTrace::_processFrame()
{
  if (_frameX % 100 == 0) { printf(".");  fflush(stdout); }

  _swap(&_current, &_next);  _next->clear();

  double thresh = _topScore + _beam;
  _topScore = _topEndScore = HUGE;

  // printf("FrameX = %d : Active Tokens = %d\n", _frameX, _current->activeHypos());

  for (Iterator itr(Cast<TokenList*>(_current)); itr.more(); itr++) {
    const TokPtr& tok(Cast<TokPtr>(itr.tok()));

    double score = tok->score();

    if (score > thresh) continue;

    _expandNode(tok->edge()->next(), tok);
  }

  _activeHypos += _next->activeTokens();

  if (_epsilon > 0.0 &&_topEndScore > 0.0 && _topEndScore < _topScore + _epsilon) {
    _validEndX++;
    // cout << _validEndX << endl;
    if (_validEndX == _validEndN) {
      printf("Stopping forward decoding : _topScore = %g : _topEndScore = %g\n", _topScore, _topEndScore);
      throw jiterator_error("end of samples!");
    }
  } else
    _validEndX = 0;
}

void DecoderWordTrace::_expandToEnd()
{
  _next->clear();
  for (TokenList::Iterator itr(Cast<TokenList*>(_current)); itr.more(); itr++) {
    const TokPtr& tok(itr.tok());

    if (tok->edge()->next()->isFinal()) {
      float lmScoreEdge = _lmScale * float(tok->edge()->next()->cost());
      TokPtr wordToken(_advanceTokens(tok->edge(), /*acScoreEdge=*/ 0.0, lmScoreEdge, tok));

      _placeOnList(tok->edge(), wordToken);
    }

    _expandNodeToEnd(tok->edge()->next(), tok);
  }
}

bool DecoderWordTrace::_notPresent(unsigned wordSeqX)
{
  for (unsigned i = 0; i < _tokenX; i++)
    if (_uniqueIndices[0] == wordSeqX) return false;

  _uniqueIndices[_tokenX++] = wordSeqX;

  return true;
}

void DecoderWordTrace::_placeOnList(const WFSTFlyWeightSortedOutput::Edge* edge, TokPtr tok)
{
  double ttlScore = tok->score();

  if ( ttlScore < _topScore && edge->input() != 0 ) _topScore = ttlScore;

  unsigned stateX = edge->next()->index();
  if (_next->isPresent(stateX)) {
    TokPtr nextToken(Cast<TokPtr>(_next->token(stateX)));

    if (_generateLattice) {
      // perform merge-sort-unique
      _tokenX = 0;
      TokPtr mergeToken, sortToken;

      // find next token to insert
      while (_tokenX < _propagateN && (tok.isNull() == false || nextToken.isNull() == false)) {
	TokPtr bestToken;
	if (tok.isNull()) {

	  bestToken = nextToken;  nextToken = nextToken->worse();

	} else if (nextToken.isNull()) {

	  bestToken = tok;  tok = tok->worse();

	} else {

	  if (tok->score() < nextToken->score()) {
	    bestToken = tok;  tok = tok->worse();
	  } else {
	    bestToken = nextToken;  nextToken = nextToken->worse();
	  }

	}

	// check for uniqueness and perform insertion
	if (_notPresent(bestToken->wordTrace()->wordSequenceX())) {
	  if (mergeToken.isNull()) {
	    mergeToken = bestToken;  sortToken = mergeToken;  sortToken->setWorse(NULL);
	  } else {
	    sortToken->setWorse(bestToken);  sortToken = bestToken;  sortToken->setWorse(NULL);
	  }
	}
      }
      _next->replace(stateX, mergeToken);

    } else if (ttlScore < nextToken->score()) {
      _next->replace(stateX, tok);
    }

  } else {
    _next->insert(stateX, tok);
  }
}

void DecoderWordTrace::_clear()
{
  _Decoder<WFSTFlyWeightSortedOutput, WFSTFlyWeightSortedOutput::Node*,
    WFSTFlyWeightSortedOutput::Edge*, WFSTFlyWeightSortedOutputPtr>::_clear();
  _stateIndices.clear();
}

// create a single link in lattice
void DecoderWordTrace::_minorTrace(LatticePtr& lattice, const BaseDecoder::TokPtr& eTok, _LNIndex prev)
{
  const TokPtr& endTok(Cast<TokPtr>(eTok));
  const WordTracePtr& wordTrace(endTok->wordTrace());

  int      endFrame      = endTok->frameX();

  unsigned currentInput  = endTok->edge()->input();
  unsigned currentOutput = wordTrace.isNull() ? 0    : wordTrace->wordX();

  TokPtr   beginTok      = wordTrace.isNull() ? NULL : wordTrace->tokenList();
  int      begFrame      = beginTok.isNull()  ? 0    : beginTok->frameX();

  // create or find the correct lattice node
  Lattice::NodePtr endNode(_findLNode(lattice, prev));

  // create link between nodes
  _LNIndex lnode;
  Lattice::NodePtr begNode;
  bool   create  = true;
  double acScore = endTok->acScore();
  double lmScore = endTok->lmScore();
  if (beginTok.isNull()) {
    begNode = lattice->initial();
  } else {
    acScore -= beginTok->acScore();
    lmScore -= beginTok->lmScore();

    lnode    = make_pair(beginTok->edge()->prev()->index(), begFrame);
    begNode  = _findLNode(lattice, lnode, &create);
  }
  if (currentInput  == _silenceX) lmScore -= (_lmScale * _silPenalty);
  if (currentOutput != 0) lmScore -= (_lmScale * _lmPenalty);
  lmScore /= _lmScale;

  LatticeEdgeData d(begFrame, endFrame, acScore, lmScore);
  Lattice::EdgePtr edgePtr(new Lattice::Edge(begNode, endNode, currentInput, currentOutput, d));
  begNode->addEdgeForce(edgePtr);

  // recursive call to continue traceback
  if (beginTok.isNull() == false && create)
    _majorTrace(lattice, beginTok, lnode);
}

unsigned DecoderWordTrace::_hashWordSequence(const TokPtr& thisToken, unsigned wordX)
{
  if (_fastHash) {
    unsigned wordSequenceX = 0;
    if (thisToken.isNull() == false && thisToken->wordTrace().isNull() == false)
      wordSequenceX = thisToken->wordTrace()->wordSequenceX();
    return 337 * wordSequenceX + wordX + 1;
  }

  String newSequence;
  if (thisToken.isNull() || thisToken->wordTrace().isNull()) {
    newSequence = _wfst->outputLexicon()->symbol(wordX);
  } else {
    const String& word(_wfst->outputLexicon()->symbol(wordX));
    const String& sequence(_wordSequenceHash[thisToken->wordTrace()->wordSequenceX()]);
    newSequence = sequence + " " + word;
  }

  if (_wordSequenceHash.isPresent(newSequence) == false)
    _wordSequenceHash.add(newSequence, newSequence);

  return _wordSequenceHash.index(newSequence);
}

void DecoderWordTrace::_newUtterance()
{
  BaseDecoder::_newUtterance();
  _wordSequenceHash.clear();
}

DecoderWordTrace::TokPtr DecoderWordTrace::
_advanceTokens(const WFSTFlyWeightSortedOutput::Edge* edgeA, double acScoreEdge, double lmScoreEdge, const TokPtr& topToken)
{
  if (topToken.isNull()) {
    if (edgeA->input() == _silenceX)
      lmScoreEdge += (_lmScale * _silPenalty);
    
    return TokPtr(new Token(acScoreEdge, lmScoreEdge, _frameX, edgeA));
  }

  TokPtr newTopToken, newToken;

  TokPtr thisToken(topToken);
  while (thisToken.isNull() == false) {
    double acScore = acScoreEdge + thisToken->acScore();
    double lmScore = lmScoreEdge + thisToken->lmScore();

    // apply silence insertion penalty
    if (edgeA->input() == _silenceX && (thisToken.isNull() || thisToken->edge()->input() != _silenceX))
      lmScore += (_lmScale * _silPenalty);

    WordTracePtr wordTrace(thisToken->wordTrace());
    WordTrace* wtrace = wordTrace.operator->();
    TokPtr tok(new Token(acScore, lmScore, _frameX, edgeA, thisToken->wordTrace()));
    if (newTopToken.isNull())
      newTopToken = tok;
    else
      newToken->setWorse(tok);

    newToken = tok;

    const TokPtr tempToken(thisToken->worse());  thisToken = tempToken;
  }

  return newTopToken;
}

void DecoderWordTrace::_expandNode(const WFSTFlyWeightSortedOutput::Node* node, const TokPtr& topToken)
{
  double acScoreNode = 0.0;
  double lmScoreNode = 0.0;

  if (topToken.isNull() == false) {
    acScoreNode = topToken->acScore();
    lmScoreNode = topToken->lmScore();
  }

  for (WFSTFlyWeightSortedOutput::Node::Iterator itr(node); itr.more(); itr++) {
    const WFSTFlyWeightSortedOutput::Edge* edge(itr.edge());

    unsigned distX       = edge->input();
    unsigned wordX       = edge->output();
    double   acScoreEdge = (distX == 0) ? 0.0 : _dist->find(distX-1)->score(_frameX);
    double   lmScoreEdge = _lmScale * float(edge->cost());
    bool     newWord     = false;

    // apply word insertion penalty
    if (wordX != 0) {
      lmScoreEdge += (_lmScale * _lmPenalty);
      newWord = true;
    }

    if (edge->next()->isFinal()) {
      double ttlScoreNode = acScoreNode + lmScoreNode + acScoreEdge + lmScoreEdge;
      if (ttlScoreNode < _topEndScore) _topEndScore = ttlScoreNode;
    }

    TokPtr wordToken(_advanceTokens(edge, acScoreEdge, lmScoreEdge, topToken));

    if (_insertSilence && edge->input() == _silenceX && (topToken.isNull() || topToken->edge()->input() != _silenceX))
      newWord = true;

    // create new word trace at word boundary
    if (newWord) {
      unsigned wordSequenceX = _hashWordSequence(topToken, wordX);
      WordTracePtr wordTrace(new WordTrace(wordX, wordSequenceX, _frameX, wordToken));

      // propagate only best token past word boundary
      wordToken = new Token(wordToken->acScore(), wordToken->lmScore(), _frameX, edge, wordTrace);
    }

    if (distX == 0)
      _expandNode(edge->next(), wordToken);
    else
      _placeOnList(edge, wordToken);
  }
}

void DecoderWordTrace::_expandNodeToEnd(const WFSTFlyWeightSortedOutput::Node* node, const TokPtr& topToken)
{
  for (WFSTFlyWeightSortedOutput::Node::Iterator itr(node); itr.more(); itr++) {
    const WFSTFlyWeightSortedOutput::Edge* edge(itr.edge());

    if (edge->input() != 0) continue;

    unsigned wordX       = edge->output();
    double   lmScoreEdge = _lmScale * float(edge->cost());

    // apply word insertion penalty
    if (wordX != 0) lmScoreEdge += (_lmScale * _lmPenalty);

    TokPtr wordToken(_advanceTokens(edge, /*acScoreEdge=*/ 0.0, lmScoreEdge, topToken));

    // apply word insertion penalty
    if (wordX != 0) {
      unsigned wordSequenceX = _hashWordSequence(topToken, wordX);
      WordTracePtr wordTrace(new WordTrace(wordX, wordSequenceX, _frameX, wordToken));

      // propagate only best token past word boundary
      wordToken = new Token(wordToken->acScore(), wordToken->lmScore(), _frameX, edge, wordTrace);
    }

    if (edge->next()->isFinal()) {
      float lmScoreFinal = _lmScale * float(edge->next()->cost());
      TokPtr endToken(_advanceTokens(edge, /*acScoreEdge=*/ 0.0, lmScoreFinal, wordToken));

      _placeOnList(edge, endToken);

      if (endToken->score() < _topEndScore) { _topEndScore = endToken->score(); }
    }

    _expandNodeToEnd(edge->next(), wordToken);
  }
}

MemoryManager<DecoderWordTrace::WordTrace>& DecoderWordTrace::WordTrace::memoryManager()
{
  static MemoryManager<DecoderWordTrace::WordTrace > _MemoryManager("DecoderWordTrace::WordTrace");
  return _MemoryManager;
}

MemoryManager<DecoderWordTrace::Token>& DecoderWordTrace::Token::memoryManager()
{
  static MemoryManager<Token> _MemoryManager("DecoderWordTrace::Token");
  return _MemoryManager;
}


// ----- methods for class `DecoderFastComposition' -----
//
DecoderFastComposition::
DecoderFastComposition(DistribSetPtr& dist,
		       double beam, double lmScale, double lmPenalty,
		       double silPenalty, const String& silSymbol, const String& eosSymbol,
		       unsigned heapSize, unsigned topN, bool generateLattice,
		       unsigned propagateN, bool fastHash, bool insertSilence,
		       bool pushWeights)
  : DecoderWordTrace(dist, beam, lmScale, lmPenalty, silPenalty,
		     silSymbol, eosSymbol, heapSize, topN, 0.0, 30, generateLattice,
		     propagateN, fastHash, insertSilence),
    _pushWeights(pushWeights)
{
  if (_pushWeights)
    _weightPusher = new _WeightPusher;
  cout << "TopN = " << _topN << endl;
}

void DecoderFastComposition::setAB(WFSTFlyWeightSortedOutputPtr& A, WFSTFlyWeightSortedInputPtr& B)
{
  _wfst     = A;
  _wfstB    = B;
  _silenceX = A->inputLexicon()->index(_silSymbol);
  _eosX     = B->outputLexicon()->index(_eosSymbol);

  if (_pushWeights) A->setComingSymbols();
  B->hash();
}

// write 1-best hypo in CTM format
void DecoderFastComposition::
writeCTM(const String& conv, const String& channel, const String& spk, const String& utt,
	 double cfrom, double score, const String& fileName, double frameInterval)
{
  FILE* fp = (fileName == "") ? stdout : fileOpen(fileName, "a");

  fprintf(fp, "# %s %10.4f %10.4f\n", utt.c_str(), cfrom, score);

  TokPtr   tok(Cast<TokPtr>(_bestToken()));
  WFSTFlyWeightSortedInput::Edge* edge(Cast<WFSTFlyWeightSortedInput::Edge*>(tok->edge()));
  int      endX   = tok->frameX();
  // double   wscore = tok->score();
  double   wscore = tok->acScore();

  vector<String> words;
  vector<double> starts, durations, scores;
  do {
    unsigned outX = tok->edgeB()->output();

    if (outX != 0) {
      int    startX = tok->frameX();
      double beg    = cfrom + startX * frameInterval;
      double len    = (endX - startX) * frameInterval;
      String entry  = _wfst->outputLexicon()->symbol(outX);

      String::size_type first = 0, colon;
      do {
	colon = entry.substr(first).find(":");
	String word = entry;
	if (colon != String::npos) { word = entry.substr(colon + 1); /* printf("Splitting %s --> %s\n", entry.c_str(), word.c_str()); */ len /= 2; beg += len; }
	words.push_back(word);
	starts.push_back(beg);
	durations.push_back(len);
	// double oscore = (tok->prev().isNull()) ? 0.0 : tok->prev()->score();
	double oscore = (tok->prev().isNull()) ? 0.0 : tok->prev()->acScore();
	scores.push_back(wscore - oscore);
	endX   = startX;
	// wscore = tok->score();
	wscore = oscore;

	if (colon != String::npos) { entry = entry.substr(first, colon); beg = cfrom + startX * frameInterval; }
      } while(colon != String::npos);
    }
    TokPtr tmp(Cast<TokPtr>(tok->prev())); tok = tmp;
  } while(tok.isNull() == false);

  for (int i = words.size() - 1; i >= 0; i--) {
    const String& word(words[i]);
    if (word == _eosSymbol) continue;
    fprintf(fp, "%s %s %7.2f %7.2f %-20s %7.2f\n",
	    conv.c_str(), channel.c_str(), starts[i], durations[i], word.c_str(), scores[i]);
  }

  if (fp != stdout) fileClose(fileName, fp);
}

void DecoderFastComposition::_processFirstFrame()
{
  printf(".");  fflush(stdout);

  _topScore = HUGE;
  _frameX = 0;

  _expandNode(_wfst->initial());
  _activeHypos = _next->activeTokens();
}

void DecoderFastComposition::_processFrame()
{
  if (_frameX % 100 == 0) { printf(".");  fflush(stdout); }
  if (_pushWeights) _weightPusher->increment();

  _swap(&_current, &_next);  _next->clear();

  double thresh = _topScore + _beam;
  _topScore = HUGE;

  // printf("FrameX = %d : Active Tokens = %d\n", _frameX, _current->activeHypos());

  if (_topN > 0) {
    unsigned hypoCnt = 0;
    for (SortedIterator itr(Cast<TokenList*>(_current)); itr.more(); itr++) {
      if (hypoCnt == _topN) break;
      const TokPtr& tok(Cast<TokPtr>(itr.tok()));

      if (tok->score() > thresh || hypoCnt++ == _topN) break;

      // printf("Hypo %d : Score = %g\n", hypoCnt, tok->score());

      _expandNode(tok->edge()->next(), tok);

      hypoCnt++;
    }
  } else {
    for (Iterator itr(Cast<TokenList*>(_current)); itr.more(); itr++) {
      const TokPtr& tok(Cast<TokPtr>(itr.tok()));

      if (tok->score() > thresh) continue;

      _expandNode(tok->edge()->next(), tok);
    }
  }

  _activeHypos += _next->activeTokens();
}

void DecoderFastComposition::_expandToEnd()
{
  _next->clear();
  for (TokenList::Iterator itr(Cast<TokenList*>(_current)); itr.more(); itr++) {
    const TokPtr& tok(Cast<TokPtr>(itr.tok()));

    if (tok->edge()->next()->isFinal()) {
      float lmScoreEdge = _lmScale * float(tok->edge()->next()->cost());

      if (tok->edgeB() != NULL) {
	if (tok->edgeB()->next()->isFinal()) {
	  lmScoreEdge += _lmScale * float(tok->edgeB()->next()->cost());
	} else continue;
      }

      TokPtr wordToken(_advanceTokens(tok->edge(), /*acScoreEdge=*/ 0.0, lmScoreEdge, tok));

      _placeOnList(tok->edge(), tok->edgeB(), wordToken);
    }

    _expandNodeToEnd(tok->edge()->next(), tok);
  }
}

void DecoderFastComposition::
_expandNode(const WFSTFlyWeightSortedOutput::Node* node, const TokPtr& topToken)
{
  const WFSTFlyWeightSortedInput::Node* nodeB    = NULL;
  const WFSTFlyWeightSortedInput::Edge* nextEdge = NULL;
  if (topToken.isNull() || topToken->edgeB() == NULL) {
    nodeB    = _wfstB->initial();    
  } else {
    nextEdge = topToken->edgeB();
    nodeB    = nextEdge->next();
  }
  for (WFSTFlyWeightSortedOutput::Node::Iterator itr(node); itr.more(); itr++) {
    const WFSTFlyWeightSortedOutput::Edge* edgeA(itr.edge());

    unsigned distX       = edgeA->input();
    unsigned wordX       = edgeA->output();
    double   lmScoreEdge = _lmScale * float(edgeA->cost());

    float pushedWeight = topToken.isNull() ? 0.0 : topToken->pushedWeight();
    if (wordX != 0) {
      const WFSTFlyWeightSortedInput::Edge* testB = nodeB->edge(wordX);
      if (testB == NULL) continue;

      wordX        = testB->output();
      lmScoreEdge += _lmScale * (float(testB->cost()) - pushedWeight);
      nextEdge     = testB;
      pushedWeight = 0.0;
    }

    // check fence and push weights
    if (nextEdge != NULL && edgeA->prev() != edgeA->next()) {
      if (_fence->present(edgeA->next()->index(), nextEdge->next()->index())) continue;

      if (_pushWeights) {
	float push = _weightPusher->push(edgeA, nextEdge) - pushedWeight;  pushedWeight += push;  lmScoreEdge += _lmScale * push;
      }
    }
    double acScoreEdge = (distX == 0) ? 0.0 : _dist->find(distX-1)->score(_frameX);

    // apply word insertion penalty
    bool newWord = false;
    if (wordX != 0) {
      lmScoreEdge += (_lmScale * _lmPenalty);
      newWord = true;
    }

    if (_insertSilence && edgeA->input() == _silenceX && (topToken.isNull() || topToken->edge()->input() != _silenceX))
      newWord = true;

    // create new word trace at word boundary
    TokPtr wordToken;
    if (newWord) {
      // create new word trace at word boundary
      unsigned wordSequenceX = _hashWordSequence(topToken, wordX);
      WordTracePtr wordTrace(new WordTrace(wordX, wordSequenceX, _frameX, topToken));

      // propagate only best token past word boundary
      if (topToken.isNull() == false) { acScoreEdge += topToken->acScore(); lmScoreEdge += topToken->lmScore(); }
      wordToken = new Token(acScoreEdge, lmScoreEdge, _frameX, edgeA, wordTrace, nextEdge);
    } else {
      wordToken = _advanceTokens(edgeA, acScoreEdge, lmScoreEdge, topToken, nextEdge);
    }
    wordToken->setPushedWeight(pushedWeight);

    if (distX == 0) {
      /*
      if (topToken.isNull() == false && topToken->edgeB() != NULL)
	printf("Expanding (%d x %d) with epsilon transition : Frame %d.\n",
	       edgeA->next()->index(), topToken->edgeB()->next()->index(), _frameX);
      else
	printf("Expanding (%d) with epsilon transition : Frame %d.\n",
	       edgeA->next()->index(), _frameX);
      fflush(stdout);
      */
      _expandNode(edgeA->next(), wordToken);
    } else {
      _placeOnList(edgeA, nextEdge, wordToken);
    }
  }
}

void DecoderFastComposition::
_expandNodeToEnd(const WFSTFlyWeightSortedOutput::Node* node, const TokPtr& topToken)
{
  const WFSTFlyWeightSortedInput::Node* nodeB    = NULL;
  const WFSTFlyWeightSortedInput::Edge* nextEdge = NULL;
  if (topToken.isNull() || topToken->edgeB() == NULL) {
    nodeB    = _wfstB->initial();    
  } else {
    nextEdge = topToken->edgeB();
    nodeB    = nextEdge->next();
  }
  for (WFSTFlyWeightSortedOutput::Node::Iterator itr(node); itr.more(); itr++) {
    const WFSTFlyWeightSortedOutput::Edge* edgeA(itr.edge());

    if (edgeA->input() != 0) continue;

    unsigned wordX       = edgeA->output();
    double   acScoreEdge = 0.0;
    double   lmScoreEdge = _lmScale * float(edgeA->cost());

    float pushedWeight = topToken->pushedWeight();
    if (wordX != 0) {
      const WFSTFlyWeightSortedInput::Edge* testB = nodeB->edge(wordX);
      if (testB == NULL) continue;

      wordX        = testB->output();
      lmScoreEdge += _lmScale * (float(testB->cost()) - pushedWeight);
      nextEdge     = testB;
      pushedWeight = 0.0;
    }

    // check fence and push weights
    if (nextEdge != NULL && edgeA->prev() != edgeA->next()) {
      if (_fence->present(edgeA->next()->index(), nextEdge->next()->index())) continue;

      if (_pushWeights) {
	float push = _weightPusher->push(edgeA, nextEdge) - pushedWeight;  pushedWeight += push;  lmScoreEdge += _lmScale * push;
      }
    }

    // apply word insertion penalty
    TokPtr wordToken;
    if (wordX == 0) {
      wordToken = _advanceTokens(edgeA, acScoreEdge, lmScoreEdge, topToken, nextEdge);
    } else {
      lmScoreEdge += (_lmScale * _lmPenalty);
      unsigned wordSequenceX = _hashWordSequence(topToken, wordX);
      WordTracePtr wordTrace(new WordTrace(wordX, wordSequenceX, _frameX, topToken));

      // propagate only best token past word boundary
      if (topToken.isNull() == false) { acScoreEdge += topToken->acScore(); lmScoreEdge += topToken->lmScore(); }
      wordToken = new Token(acScoreEdge, lmScoreEdge, _frameX, edgeA, wordTrace, nextEdge);
    }
    wordToken->setPushedWeight(pushedWeight);

    bool isFinal = false;
    if (edgeA->next()->isFinal())
      if (nextEdge == NULL)
	isFinal = true;
      else if (nextEdge->next()->isFinal())
	isFinal = true;
    if (isFinal) {
      float lmScoreFinal = _lmScale * float(edgeA->next()->cost());
      if (nextEdge != NULL) lmScoreFinal += _lmScale * float(nextEdge->next()->cost());
      TokPtr endToken(_advanceTokens(edgeA, /*acScoreEdge=*/ 0.0, lmScoreFinal, wordToken, nextEdge));

      _placeOnList(edgeA, nextEdge, endToken);
    }

    _expandNodeToEnd(edgeA->next(), wordToken);
  }
}

DecoderFastComposition::TokPtr DecoderFastComposition::
_advanceTokens(const WFSTFlyWeightSortedOutput::Edge* edgeA, double acScoreEdge, double lmScoreEdge, const TokPtr& topToken,
	       const WFSTFlyWeightSortedInput::Edge* edgeB)
{
  if (topToken.isNull()) {
    if (edgeA->input() == _silenceX)
      lmScoreEdge += (_lmScale * _silPenalty);

    return TokPtr(new Token(acScoreEdge, lmScoreEdge, _frameX, edgeA, edgeB));
  }

  TokPtr newTopToken, newToken;

  TokPtr thisToken(topToken);
  while (thisToken.isNull() == false) {
    double acScore = acScoreEdge + thisToken->acScore();
    double lmScore = lmScoreEdge + thisToken->lmScore();

    // apply silence insertion penalty
    if (edgeA->input() == _silenceX && (thisToken.isNull() || thisToken->edge()->input() != _silenceX))
      lmScore += (_lmScale * _silPenalty);

    TokPtr tok(new Token(acScore, lmScore, _frameX, edgeA, thisToken->wordTrace(), edgeB));
    if (newTopToken.isNull())
      newTopToken = tok;
    else
      newToken->setWorse(tok);

    newToken = tok;

    const TokPtr tempToken(Cast<TokPtr>(thisToken->worse()));  thisToken = tempToken;
  }

  return newTopToken;
}

void DecoderFastComposition::
_placeOnList(const WFSTFlyWeightSortedOutput::Edge* edgeA, const WFSTFlyWeightSortedInput::Edge* edgeB, TokPtr tok)
{
  double ttlScore = tok->score();

  if ( ttlScore < _topScore && edgeA->input() != 0 ) _topScore = ttlScore;

  unsigned stateA = edgeA->next()->index();
  unsigned stateB = (edgeB == NULL) ? 0 : edgeB->next()->index();
  if (_next->isPresent(stateA, stateB)) {
    TokPtr nextToken(Cast<TokPtr>(_next->token(stateA, stateB)));

    if (_generateLattice) {
      // perform merge-sort-unique
      _tokenX = 0;
      TokPtr mergeToken, sortToken;

      // find next token to insert
      while (_tokenX < _propagateN && (tok.isNull() == false || nextToken.isNull() == false)) {
	TokPtr bestToken;
	if (tok.isNull()) {

	  bestToken = nextToken;  nextToken = Cast<TokPtr>(nextToken->worse());

	} else if (nextToken.isNull()) {

	  bestToken = tok;  tok = Cast<TokPtr>(tok->worse());

	} else {

	  if (tok->score() < nextToken->score()) {
	    bestToken = tok;  tok = Cast<TokPtr>(tok->worse());
	  } else {
	    bestToken = nextToken;  nextToken = Cast<TokPtr>(nextToken->worse());
	  }

	}

	// check for uniqueness and perform insertion
	if (_notPresent(bestToken->wordTrace()->wordSequenceX())) {
	  if (mergeToken.isNull()) {
	    mergeToken = bestToken;  sortToken = mergeToken;  sortToken->setWorse(NULL);
	  } else {
	    sortToken->setWorse(bestToken);  sortToken = bestToken;  sortToken->setWorse(NULL);
	  }
	}
      }
      _next->replace(stateA, mergeToken, stateB);

    } else if (ttlScore < nextToken->score()) {
      _next->replace(stateA, tok, stateB);
    }

  } else {
    _next->insert(stateA, tok, stateB);
  }
}

unsigned DecoderFastComposition::_stateIndex(const TokPtr& token)
{
  unsigned indexA = token->edge()->prev()->index();
  unsigned indexB = (token->edgeB() == NULL) ? _wfstB->initial()->index() : token->edgeB()->prev()->index();
  _StateIndex stateIndex(indexA, indexB);
  _StateIndexMapIterator iter = _stateIndices.find(stateIndex);

  if (iter != _stateIndices.end())
    return (*iter).second;

  unsigned index = _stateIndices.size();
  _stateIndices.insert(_StateIndexMapType(stateIndex, index));

  return index;
}

// create a single link in lattice
void DecoderFastComposition::_minorTrace(LatticePtr& lattice, const BaseDecoder::TokPtr& eTok, _LNIndex prev)
{
  const TokPtr& endTok(Cast<TokPtr>(eTok));
  const WordTracePtr& wordTrace(endTok->wordTrace());

  int      endFrame      = endTok->frameX();

  unsigned currentInput  = endTok->edge()->input();
  unsigned currentOutput = wordTrace.isNull() ? 0    : wordTrace->wordX();

  TokPtr   beginTok      = wordTrace.isNull() ? NULL : Cast<TokPtr>(wordTrace->tokenList());
  int      begFrame      = beginTok.isNull()  ? 0    : beginTok->frameX();

  // create or find the correct lattice node
  Lattice::NodePtr endNode(_findLNode(lattice, prev));

  // create link between nodes
  _LNIndex lnode;
  Lattice::NodePtr begNode;
  bool   create  = true;
  double acScore = endTok->acScore();
  double lmScore = endTok->lmScore();
  if (beginTok.isNull()) {
    begNode = lattice->initial();
  } else {
    acScore -= beginTok->acScore();
    lmScore -= beginTok->lmScore();

    lnode    = make_pair(_stateIndex(beginTok), begFrame);
    begNode  = _findLNode(lattice, lnode, &create);
  }
  if (currentInput  == _silenceX) lmScore -= (_lmScale * _silPenalty);
  if (currentOutput != 0) lmScore -= (_lmScale * _lmPenalty);
  lmScore /= _lmScale;

  LatticeEdgeData d(begFrame, endFrame, acScore, lmScore);
  Lattice::EdgePtr edgePtr(new Lattice::Edge(begNode, endNode, currentInput, currentOutput, d));
  begNode->addEdgeForce(edgePtr);

  // recursive call to continue back trace
  if (beginTok.isNull() == false && create)
    _majorTrace(lattice, beginTok, lnode);
}

// decode current utterance
double DecoderFastComposition::decode(bool verbose)
{
  if (_fence.isNull())
    throw jconsistency_error("Must execute 'setFence' before 'decode'.");

  _Decoder<WFSTFlyWeightSortedOutput, WFSTFlyWeightSortedOutput::Node*,
    WFSTFlyWeightSortedOutput::Edge*, WFSTFlyWeightSortedOutputPtr>::decode(verbose);
}

void DecoderFastComposition::_newUtterance()
{
  DecoderWordTrace::_newUtterance();
  if (_pushWeights) _weightPusher->clear();
}

MemoryManager<DecoderFastComposition::Token>& DecoderFastComposition::Token::memoryManager()
{
  static MemoryManager<Token> _MemoryManager("DecoderFastComposition::Token");
  return _MemoryManager;
}


// ----- methods for class `DecoderFastComposition::_WeightPusher' -----
//
DecoderFastComposition::_WeightPusher::_WeightPusher(unsigned bucketN, int pastFramesN)
  : _binN(bucketN), _index(new _WeightIndex*[_binN]), _frameX(0),
    _deleteX(-pastFramesN), _pastFramesN(pastFramesN)
{
  for (unsigned binX = 0; binX < _binN; binX++)
    _index[binX] = NULL;
}

DecoderFastComposition::_WeightPusher::~_WeightPusher()
{
  delete[] _index;
}

void DecoderFastComposition::_WeightPusher::clear()
{
  printf("Clearing '_WeightPusher'.\n");  fflush(stdout);

  _frameX = 0;  _deleteX = -_pastFramesN;

  unsigned allocatedN = 0;
  for (unsigned binX = 0; binX < _binN; binX++) {
    _WeightIndex* index = _index[binX];
    while (index != NULL) {
      _WeightIndex* mustDelete = index;
      index = index->_chain;
      delete mustDelete;
      allocatedN++;
    }
    _index[binX] = NULL;
  }
  printf("Deallocated %d '_WeightIndex' objects.\n", allocatedN);  fflush(stdout);
}

// look ahead for weight pushing
float DecoderFastComposition::_WeightPusher::
push(const WFSTFlyWeightSortedOutput::Edge* edgeA, const WFSTFlyWeightSortedInput::Edge* edgeB)
{
  float pushedWeight  = HUGE;

  unsigned stateA     = edgeA->next()->index();
  unsigned stateB     = edgeB->next()->index();
  unsigned hashKey    = _hash(stateA, stateB);

  // look for desired weight in hash table, delete old entries
  _WeightIndex* index = _index[hashKey];
  _WeightIndex* prev  = NULL;
  while (index != NULL) {
    if (index->_indexA == stateA && index->_indexB == stateB) {
      index->_frameX = _frameX;
      return index->_weight;
    }
    if (index->_frameX < _deleteX) {
      if (prev != NULL)
	prev->_chain    = index->_chain;
      else
	_index[hashKey] = index->_chain;
      _WeightIndex* mustDelete = index;
      index = index->_chain;
      delete mustDelete;
    } else {
      prev  = index;
      index = index->_chain;
    }
  }

  // weight not found, recalculate it
  const unsigned* symbolsA = edgeA->next()->comingSymbols();
  while (*symbolsA != UINT_MAX) {
    const WFSTFlyWeightSortedInput::Edge* nextEdge = edgeB->next()->edge(*symbolsA);
    if (nextEdge != NULL && nextEdge->input() == *symbolsA && float(nextEdge->cost()) < pushedWeight)
      pushedWeight = float(nextEdge->cost());
    symbolsA++;
  }

  // store recalculated weight in hash table
  if (pushedWeight < HUGE) {
    index = new _WeightIndex(stateA, stateB, pushedWeight, _frameX, _index[hashKey]);
    _index[hashKey] = index;
    return pushedWeight;
  }

  return 0.0;
}

MemoryManager<DecoderFastComposition::_WeightPusher::_WeightIndex>&
DecoderFastComposition::_WeightPusher::_WeightIndex::memoryManager()
{
  static MemoryManager<_WeightIndex> _MemoryManager("DecoderFastComposition::_WeightPusher::_WeightIndex");
  return _MemoryManager;
}
