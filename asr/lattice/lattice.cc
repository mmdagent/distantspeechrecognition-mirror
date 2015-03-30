//
//                              Millennium
//                    Distant Speech Recognition System
//                                 (dsr)
//
//  Module:  dsr.lattice
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


#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <algorithm>
#include "lattice/lattice.h"
#include <typeinfo>

template<>
MemoryManager<_Token<Inherit<WFSTransducer::Edge, refcountable_ptr<WFSAcceptor::Edge> > > >&
_Token<Inherit<WFSTransducer::Edge, refcountable_ptr<WFSAcceptor::Edge> > >::memoryManager() {
  static MemoryManager<_Token> _MemoryManager("_Token<Inherit<WFSTransducer::Edge, refcountable_ptr<WFSAcceptor::Edge> > > ");
  return _MemoryManager;
}

template<>
MemoryManager<WFST<LatticeNodeData, LatticeEdgeData>::Edge>& WFST<LatticeNodeData, LatticeEdgeData>::Edge::memoryManager() {
  static MemoryManager<Edge> _MemoryManager("WFST<LatticeNodeData, LatticeEdgeData>::Edge");
  return _MemoryManager;
}

template<>
MemoryManager<WFST<LatticeNodeData, LatticeEdgeData>::Node>& WFST<LatticeNodeData, LatticeEdgeData>::Node::memoryManager() {
  static MemoryManager<Node> _MemoryManager("WFST<LatticeNodeData, LatticeEdgeData>::Node");
  return _MemoryManager;
}


// ----- methods for class `Lattice' -----
//
#if 0
Lattice::Lattice(LexiconPtr& statelex, LexiconPtr& inlex, LexiconPtr& outlex)
  : LatticeData(statelex, inlex, outlex),
    _acScale(1.0), _lmScale(1.0), _lmPenalty(0.0), _silPenalty(0.0)
{
  _initial = _newNode(0);
}
#endif

Lattice::Lattice(LexiconPtr statelex, LexiconPtr inlex, LexiconPtr outlex)
  : LatticeData(statelex, inlex, outlex),
    _acScale(1.0), _lmScale(1.0), _lmPenalty(0.0), _silPenalty(0.0)
{
  _initial = _newNode(0);
}

Lattice::Lattice(WFSTSortedInputPtr& wfst)
  : LatticeData(wfst->stateLexicon(), wfst->inputLexicon(), wfst->outputLexicon()),
    _acScale(1.0), _lmScale(1.0), _lmPenalty(0.0), _silPenalty(0.0)
{
  _addNode(wfst, wfst->initial());

  for (WFSAcceptor::_NodeVectorIterator itr = wfst->_nodes.begin(); itr != wfst->_nodes.end(); itr++)
    _addNode(wfst, Cast<WFSTSortedInput::NodePtr>(*itr));
}

void Lattice::_addNode(WFSTSortedInputPtr& wfst, WFSTSortedInputNodePtr& node)
{
  NodePtr fromNode(find(node->index()));
  for (WFSTSortedInput::Node::Iterator itr(wfst, node); itr.more(); itr++) {
    WFSTSortedInput::EdgePtr edge(itr.edge());
    NodePtr toNode(find(edge->next()->index()));

    unsigned startT = _calculateStart(edge);
    unsigned endT   = _calculateEnd(edge);
    unsigned output = _calculateOutput(edge);

    double acScore(0.0);
    double lmScore(edge->cost());

    LatticeEdgeData d(startT, endT, acScore, lmScore);

    EdgePtr newEdge(new Edge(fromNode, toNode, edge->input(), edge->output(), d));
    fromNode->_addEdgeForce(newEdge);
  }
}

unsigned Lattice::_calculateStart(const WFSTSortedInputEdgePtr& edge)
{
  return 0;
}

unsigned Lattice::_calculateEnd(const WFSTSortedInputEdgePtr& edge)
{
  return 0;
}

unsigned Lattice::_calculateOutput(const WFSTSortedInputEdgePtr& edge)
{
  return 0;
}

Lattice::~Lattice()
{
  _clear();
}

float Lattice::rescore(double lmScale, double lmPenalty, double silPenalty, const String& silSymbol)
{
  _lmScale = lmScale;  _lmPenalty = lmPenalty;  _silPenalty = silPenalty;  _silenceX = inputLexicon()->index(silSymbol);

  _topoSort();
  _clearTokens();

  for (_Iterator itr = _snodes().begin(); itr != _snodes().end(); itr++)
    _expandNode(*itr);

  TokenPtr bestTok = _bestToken();
  return bestTok->score();
}

void Lattice::_expandNode(const NodePtr& node) const
{
  double acScoreNode = 0.0;
  double lmScoreNode = 0.0;

  if (node->data().tok().isNull() == false) {
    acScoreNode = node->data().tok()->acScore();
    lmScoreNode = node->data().tok()->lmScore();
  }

  for (Node::Iterator itr(node); itr.more(); itr++) {
    EdgePtr& edge(Cast<EdgePtr>(itr.edge()));

    double acScore = acScoreNode + _acScale * edge->data().ac();
    double lmScore = lmScoreNode + _lmScale * edge->data().lm();

    // apply word insertion penalty
    if (edge->output() != 0) lmScore += _lmScale * _lmPenalty;

    // apply silence insertion penalty
    if (edge->input() == _silenceX && (node->data().tok().isNull() || node->data().tok()->edge()->input() != _silenceX))
      lmScore += _lmScale * _silPenalty;

    double ttlScore = acScore + lmScore;

    if (edge->next()->data().tok().isNull() || ttlScore < edge->next()->data().tok()->score()) {
      /*
      printf("Adding token from %d to %d : acScore = %g : lmScore = %g : ttlScore = %g\n",
	     node->index(), edge->next()->index(), acScore, lmScore, ttlScore);
      */
	     
      edge->next()->data().setToken(TokenPtr(new Token(acScore, lmScore, /*frameX=*/ -1, edge, node->data().tok())));
    }
  }

  fflush(stdout);
}

// propagate forward probabilities from a given node
void Lattice::_forwardNode(const NodePtr& node) const
{
  LogDouble scoreNode = node->data().forwardProb();

  for (Node::Iterator itr(node); itr.more(); itr++) {
    EdgePtr& edge(Cast<EdgePtr>(itr.edge()));

    LogDouble acScore = _acScale * edge->data().ac();
    double    lmScore = _lmScale * edge->data().lm();

    // apply word insertion penalty
    if (edge->output() != 0) lmScore += _lmScale * _lmPenalty;

    // apply silence insertion penalty
    if (edge->input() == _silenceX && (node->data().tok().isNull() || node->data().tok()->edge()->input() != _silenceX))
      lmScore += _lmScale * _silPenalty;

    edge->next()->data().logAddForward(scoreNode + acScore + lmScore);
  }
}

// calculate backward probability for a given node
void Lattice::_backwardNode(NodePtr& node) const
{
  for (Node::Iterator itr(node); itr.more(); itr++) {
    EdgePtr& edge(Cast<EdgePtr>(itr.edge()));
    NodePtr  src(edge->next());

    LogDouble acScore = _acScale * edge->data().ac();
    LogDouble lmScore = _lmScale * edge->data().lm();

    // apply word insertion penalty
    if (edge->output() != 0) lmScore += _lmScale * _lmPenalty;

    // apply silence insertion penalty
    if (edge->input() == _silenceX && (node->data().tok().isNull() || node->data().tok()->edge()->input() != _silenceX))
      lmScore += _lmScale * _silPenalty;

    LogDouble ttlScore = src->data().backwardProb() + acScore + lmScore;

    if (ttlScore >= LogZero) continue;

    node->data().logAddBackward(ttlScore);
  }
}

// set gamma probabilities for each adjoining edge
void Lattice::_gammaNode(NodePtr& node) /* const */
{
  for (Node::Iterator itr(node); itr.more(); itr++) {
    EdgePtr& edge(Cast<EdgePtr>(itr.edge()));
    NodePtr next(edge->next());

    LogDouble acScore = _acScale * edge->data().ac();
    double    lmScore = _lmScale * edge->data().lm();

    // apply word insertion penalty
    if (edge->output() != 0) lmScore += _lmScale * _lmPenalty;

    // apply silence insertion penalty
    if (edge->input() == _silenceX && (node->data().tok().isNull() || node->data().tok()->edge()->input() != _silenceX))
      lmScore += _lmScale * _silPenalty;

    double gamma =
      node->data().forwardProb() + acScore + lmScore + next->data().backwardProb() - _latticeForwardProb;

    if (gamma < 0.0) {
      if (gamma < -0.0001)
	throw jconsistency_error("Warning: Neg. Log-Probability (%g) of edge %d --> %d is negative",
				 gamma, edge->prev()->index(), edge->next()->index());
      gamma = 0.0;
    }

    edge->data().setGamma(gamma);
  }
}

void Lattice::_clearTokens()
{
  for (_Iterator itr = _snodes().begin(); itr != _snodes().end(); itr++) {
    NodePtr nd(*itr);
    nd->data().clearToken();
  }
}

// find the best token in a final state
TokenPtr Lattice::_bestToken() /* const */
{
  TokenPtr bestTok;
  double   bestScore = HUGE;

  for (_ConstNodeMapIterator itr = _finis().begin(); itr != _finis().end(); itr++) {
    const NodePtr nd((*itr).second);
    if (nd->data().tok().isNull() == false && nd->data().tok()->score() < bestScore) {
      bestTok   = nd->data().tok();
      bestScore = nd->data().tok()->score();
    }
  }

  /*
  printf("Best token %d : score %g\n", bestTok->edge()->prev()->index(), bestScore);
  */

  return bestTok;
}

String Lattice::bestHypo(bool useInputSymbols) /* const */
{
  TokenPtr bestTok = _bestToken();

  unsigned lastX = 0;
  String hypo = "";
  do {
    if (useInputSymbols) {
      unsigned inX = bestTok->edge()->input();

      if (inX != 0 && inX != lastX) {
	hypo  = inputLexicon()->symbol(inX) + " " + hypo;
	lastX = inX;
      }
    } else {
      unsigned outX = bestTok->edge()->output();

      if (outX != 0)
	hypo = outputLexicon()->symbol(outX) + " " + hypo;
    }
    bestTok = bestTok->prev();

  } while(bestTok.isNull() == false);

  return hypo;
}

// calculate posterior (gamma) probabilities for lattice links
double Lattice::gammaProbs(double acScale, double lmScale, double lmPenalty, double silPenalty, const String& silSymbol)
{
  _acScale = acScale;  _lmScale = lmScale;  _lmPenalty = lmPenalty;  _silPenalty = silPenalty;  _silenceX = inputLexicon()->index(silSymbol);

  _topoSort();

  // initialize nodes for forward and backward passes
  initial()->data().zeroForward().logZeroBackward();
  for (_NodeVectorIterator itr = _allNodes().begin(); itr != _allNodes().end(); itr++) {
    NodePtr node(*itr);
    if (node.isNull()) continue;
    node->data().logZeroForward().logZeroBackward();
  }
  for (_NodeMapIterator itr = _finis().begin(); itr != _finis().end(); itr++)
    (*itr).second->data().logZeroForward().zeroBackward();

  // perform forward, backward and gamma passes
  _forwardProbs();  _backwardProbs();  _gammaProbs();

  return _latticeForwardProb;
}

double Lattice::gammaProbsDist(DistribSetBasicPtr& dss, double acScale, double lmScale, double lmPenalty,
			       double silPenalty, const String& silSymbol)
{
  dss->resetCache();

  _clearSorted();
  _updateAc(dss);

  return gammaProbs(acScale, lmScale, lmPenalty, silPenalty, silSymbol);
}

// calculate forward probabilities for lattice nodes
void Lattice::_forwardProbs()
{
  for (_Iterator itr = _snodes().begin(); itr != _snodes().end(); itr++)
    _forwardNode(*itr);

  // probability of entire lattice is sum of forward probabilities of end nodes
  _latticeForwardProb = LogZero;
  for (_NodeMapIterator itr = _finis().begin(); itr != _finis().end(); itr++) {
    double nodeCost = (*itr).second->data().forwardProb();
    _latticeForwardProb = logAdd(_latticeForwardProb, nodeCost);
  }

  /*
  printf("Lattice log-probability = %8.2f\n", _latticeForwardProb);
  */
}

// calculate backward probabilities for lattice nodes
void Lattice::_backwardProbs()
{
  for (_ReverseIterator itr = _snodes().rbegin(); itr != _snodes().rend(); itr++)
    _backwardNode(*itr);

  // probability of entire lattice is backward probability of initial node
  double latticeBackwardProb = initial()->data().backwardProb();
  if ((fabs(latticeBackwardProb - _latticeForwardProb) / _latticeForwardProb) > 0.0001)
    throw jconsistency_error("Lattice forward (%g) and backward probabilities (%g) are not equal.",
			     _latticeForwardProb, latticeBackwardProb);
}

// calculate forward probabilities for lattice links
void Lattice::_gammaProbs()
{
  for (_Iterator itr = _snodes().begin(); itr != _snodes().end(); itr++)
    _gammaNode(*itr);
}

// update acoustic likelihoods for all lattice edges
void Lattice::_updateAc(DistribSetBasicPtr& dss)
{
  _updateAcNode(dss, initial());
  for (_NodeVectorIterator itr = _allNodes().begin(); itr != _allNodes().end(); itr++) {
    NodePtr node(*itr);
    if (node.isNull()) continue;
    _updateAcNode(dss, node);
  }
}

// update acoustic likelihoods for edges from a particular node
void Lattice::_updateAcNode(DistribSetBasicPtr& dss, NodePtr& node)
{
  for (Node::Iterator itr(node); itr.more(); itr++) {
    EdgePtr& edge(Cast<EdgePtr>(itr.edge()));

    unsigned distX = edge->input();
    if (distX == 0) continue;

    double sum = 0.0;
    for (int frameX = edge->data().start(); frameX <= edge->data().end(); frameX++)
      sum += dss->find(distX-1)->score(frameX);

    if (sum > LogZero)
      throw jconsistency_error("Log-prob (%g) > LogZero (%g)\n", sum, LogZero);

    edge->data().setAc(sum);
  }
}

void Lattice::reportMemoryUsage()
{
  Lattice::Edge::memoryManager().report();
  Lattice::Node::memoryManager().report();  
}

void Lattice_reportMemoryUsage() { Lattice::reportMemoryUsage(); }

// write 1-best hypo in CTM format
void Lattice::
writeCTM(const String& conv, const String& channel, const String& spk, const String& utt,
	 double cfrom, double score, const String& fileName, double frameInterval,
	 const String& endMarker)
{
  FILE* fp = (fileName == "") ? stdout : fileOpen(fileName, "a");

  fprintf(fp, ";; %s %10.4f %10.4f\n", utt.c_str(), cfrom, score);

  TokenPtr tok    = _bestToken();
  EdgePtr& edge(Cast<EdgePtr>(tok->edge()));
  int      endX   = edge->data().end();
  // double   wscore = tok->score();
  double   wscore = tok->acScore();

  vector<String> words;
  vector<double> starts, durations, scores;
  do {
    unsigned outX = tok->edge()->output();

    if (outX != 0) {
      EdgePtr& edge(Cast<EdgePtr>(tok->edge()));
      int    startX = edge->data().start();
      double beg    = cfrom + startX * frameInterval;
      double len    = (endX - startX) * frameInterval;
      String entry  = outputLexicon()->symbol(outX);

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
    TokenPtr tmp = tok->prev(); tok = tmp;
  } while(tok.isNull() == false);

  for (int i = words.size() - 1; i >= 0; i--) {
    const String& word(words[i]);
    if (word == endMarker) continue;
    fprintf(fp, "%s %s %7.2f %7.2f %-20s %7.2f\n",
	    conv.c_str(), channel.c_str(), starts[i], durations[i], word.c_str(), scores[i]);
  }

  if (fp != stdout) fileClose(fileName, fp);
}

// write 1-best phone hypo in CTM format
void Lattice::
writePhoneCTM(const String& conv, const String& channel, const String& spk, const String& utt,
	      double cfrom, double score, const String& fileName, double frameInterval,
	      const String& endMarker)
{
  FILE* fp = (fileName == "") ? stdout : fileOpen(fileName, "a");

  fprintf(fp, ";; %s %10.4f %10.4f\n", utt.c_str(), cfrom, score);

  TokenPtr tok    = _bestToken();
  EdgePtr& edge(Cast<EdgePtr>(tok->edge()));
  int      endX   = edge->data().end();
  // double   wscore = tok->score();
  double   wscore = tok->acScore();

  vector<String> phones;
  vector<double> starts;
  vector<double> durations;
  vector<double> scores;
  unsigned lastPhoneX = 0;
  do {
    unsigned phoneX = tok->edge()->input();

    if (phoneX != 0 || phoneX != lastPhoneX) {
      EdgePtr& edge(Cast<EdgePtr>(tok->edge()));
      int           startX = edge->data().start();
      double        beg    = cfrom + startX * frameInterval;
      double        len    = (endX - startX) * frameInterval;
      const String& phone  = inputLexicon()->symbol(phoneX);

      phones.push_back(phone);
      starts.push_back(beg);
      durations.push_back(len);
      // double oscore = (tok->prev().isNull()) ? 0.0 : tok->prev()->score();
      double oscore = (tok->prev().isNull()) ? 0.0 : tok->prev()->acScore();
      scores.push_back(wscore - oscore);

      endX	 = startX;
      lastPhoneX = phoneX;
      // wscore = tok->score();
      wscore = oscore;
    }
    TokenPtr tmp = tok->prev(); tok = tmp;
  } while(tok.isNull() == false);

  /*
  fprintf(fp, "%s %s %7.2f %7.2f %-20s %7.2f\n",
	  conv.c_str(), channel.c_str(), cfrom, starts[0] - cfrom, "<s>", wscore);
  */
  for (int i = phones.size() - 1; i >= 0; i--) {
    const String& phone(phones[i]);
    if (phone == endMarker) continue;
    fprintf(fp, "%s %s %7.2f %7.2f %-20s %7.2f\n",
	    conv.c_str(), channel.c_str(), starts[i], durations[i], phone.c_str(), scores[i]);
  }

  if (fp != stdout) fileClose(fileName, fp);
}

// write 1-best hypo in HTK format
void Lattice::
writeHypoHTK(const String& conv, const String& channel, const String& spk, const String& utt,
	     double cfrom, double score, const String& fileName, int flag, double frameInterval,
	     const String& endMarker)
{
  FILE* fp = (fileName == "") ? stdout : fileOpen(fileName, "a");

  fprintf(fp, "\"%s.rec\"\n", utt.c_str());

  TokenPtr tok    = _bestToken();
  EdgePtr& edge(Cast<EdgePtr>(tok->edge()));
  int      endX   = edge->data().end();
  // double   wscore = tok->score();
  double   wscore = tok->acScore();

  vector<String> words;
  vector<double> starts;
  vector<double> durations;
  vector<double> scores;
  do {
    unsigned outX = tok->edge()->output();

    if (outX != 0) {
      EdgePtr& edge(Cast<EdgePtr>(tok->edge()));
      int           startX = edge->data().start();
      double        beg    = cfrom + startX * frameInterval;
      double        len    = (endX - startX + 1) * frameInterval; //(endX - startX) * frameInterval;
      const String& word   = outputLexicon()->symbol(outX);

      words.push_back(word);
      starts.push_back(beg);
      durations.push_back(len);
      // double oscore = (tok->prev().isNull()) ? 0.0 : tok->prev()->score();
      double oscore = (tok->prev().isNull()) ? 0.0 : tok->prev()->acScore();
      scores.push_back(wscore - oscore);

      endX   = startX - 1;
      // wscore = tok->score();
      wscore = oscore;
    }
    TokenPtr tmp = tok->prev(); tok = tmp;
  } while(tok.isNull() == false);

  
  /*
  fprintf(fp, "%s %s %7.2f %7.2f %-20s %7.2f\n",
	  conv.c_str(), channel.c_str(), cfrom, starts[0] - cfrom, "<s>", wscore);
  */
  for (int i = words.size() - 1; i >= 0; i--) {
    const String& word(words[i]);
    if (word == endMarker) continue;
    if ( flag & 0x01 )
      fprintf(fp, "%lld %lld ", (long long)( starts[i] * 10e7 ), (long long)( ( starts[i] + durations[i] ) * 10e7 ) );
    fprintf(fp, "%s", word.c_str());
    if ( flag & 0x02 )
      fprintf(fp, " %f", scores[i] );
    fprintf(fp, "\n");
  }
  fprintf(fp, ".\n");

  if (fp != stdout) fileClose(fileName, fp);
}

// write 1-best hypo in Janus conf format
void Lattice::
writeWordConfs(const String& fileName, const String& uttId, const String& endMarker)
{
  FILE* fp = (fileName == "") ? stdout : fileOpen(fileName, "a");

  TokenPtr tok = _bestToken();

  vector<String> words;
  vector<double> gammas;
  do {
    unsigned outX = tok->edge()->output();

    if (outX != 0) {
      EdgePtr& edge(Cast<EdgePtr>(tok->edge()));
      const String& word  = outputLexicon()->symbol(outX);
            double  gamma = edge->data().gamma();

      words.push_back(word);
      gammas.push_back(gamma);
    }
    TokenPtr tmp = tok->prev(); tok = tmp;
  } while(tok.isNull() == false);

  String output(uttId);
  static char buffer[100];
  for (int i = words.size() - 1; i >= 0; i--) {
    const String& word(words[i]);
          double  gamma = exp(-gammas[i]);

    if (gamma < 1.0E-04)
      gamma = 0.0;
    else if (gamma > 1.0)
      gamma = 1.0;

    if (word == endMarker) continue;

    sprintf(buffer, " { {%s} %8.6f}", word.c_str(), gamma);
    output += String(buffer);
  }
  fprintf(fp, "%s\n", output.c_str());

  if (fp != stdout) fileClose(fileName, fp);
}

// prune lattice
void Lattice::prune(double threshold)
{
  if (threshold < 0.0)
    throw jconsistency_error("Lattice pruning threshold (%g) < 0.0.\n");

  printf("Pruning lattice with threshold %g.\n", threshold);

  // remove all links with posterior probability below threshold
  initial()->_removeLinks(threshold);
  for (_NodeVectorIterator itr = _allNodes().begin(); itr != _allNodes().end(); itr++) {
    NodePtr nd(*itr);
    if (nd.isNull()) continue;
    nd->_removeLinks(threshold);
  }
  
  // topo sort to find which nodes are still reachable
  _clearSorted();  _topoSort();

  // remove unreachable nodes
  list<unsigned> nodes2BRemoved;
  for (_NodeVectorIterator itr = _allNodes().begin(); itr != _allNodes().end(); itr++) {
    NodePtr nd(*itr);
    if (nd.isNull()) continue;
    if (nd->color() == WFSAcceptor::White)
      nodes2BRemoved.push_back(nd->index());
  }
  for (list<unsigned>::iterator itr = nodes2BRemoved.begin(); itr != nodes2BRemoved.end(); itr++)
    _allNodes()[*itr] = NULL;

  // remove unreachable final nodes
  list<_NodeMapIterator> finalNodes2BRemoved;
  for (_NodeMapIterator itr = _finis().begin(); itr != _finis().end(); itr++) {
    NodePtr nd((*itr).second);
    if (nd->color() == WFSAcceptor::White)
      finalNodes2BRemoved.push_back(itr);
  }
  for (list<_NodeMapIterator>::iterator itr = finalNodes2BRemoved.begin(); itr != finalNodes2BRemoved.end(); itr++)
    _finis().erase(*itr);

  // renumber remaining nodes
  unsigned idx = 0;
  for (_Iterator itr = _snodes().begin(); itr != _snodes().end(); itr++)
    (*itr)->_setIndex(idx++);

  printf("%d nodes remain after pruning\n", idx);
}

void Lattice::pruneEdges(unsigned edgesN)
{
  unsigned edgeX = 0;
  for (EdgeIterator itr(*this); itr.more(); itr++)
    edgeX++;

  if (edgesN >= edgeX) return;

  vector<double> scores(edgeX);
  edgeX = 0;
  for (EdgeIterator itr(*this); itr.more(); itr++)
    scores[edgeX++] = itr.edge()->data().gamma();
  sort(scores.begin(), scores.end());

  printf("scores[edgesN] = %f : scores[edgesN+1] = %f\n", scores[edgesN], scores[edgesN+1]);

  double thresh = scores[edgesN];
  prune(thresh);
}

void Lattice::write(const String& fileName, bool useSymbols, bool writeData)
{
  FILE* fp = stdout;
  if (fileName != "")
    fp = fileOpen(fileName, "w");

  _topoSort();

  // write edges leaving from initial and intermediate states
  for (_Iterator itr = _snodes().begin(); itr != _snodes().end(); itr++) {
    const NodePtr nd(*itr);

    if (nd->isFinal()) continue;

    for (Node::Iterator itr(nd); itr.more(); itr++)
      if (useSymbols)
	itr.edge()->write(stateLexicon(), inputLexicon(), outputLexicon(), fp, writeData);
      else
	itr.edge()->write(fp, writeData);
  }

  // write final states
  for (_NodeMapIterator itr=_finis().begin(); itr != _finis().end(); itr++) {
    NodePtr nd((*itr).second);

    // write edges
    for (Node::Iterator itr(nd); itr.more(); itr++)
      if (useSymbols)
	itr.edge()->write(stateLexicon(), inputLexicon(), outputLexicon(), fp, writeData);
      else
	itr.edge()->write(fp, writeData);

    // write nodes
    if (useSymbols)
      nd->write(stateLexicon(), fp, writeData);
    else
      nd->write(fp, writeData);
  }

  if (fp != stdout)
    fileClose( fileName, fp);
}

void Lattice::_clear()
{
  _clearTokens();
  _clearSorted();
  WFSAcceptor::_clear();
}

WFSAcceptor::Node* Lattice::_newNode(unsigned state)
{
  Lattice::Node* result = new Lattice::Node(state, LatticeNodeData());
  return result;
}

WFSAcceptor::Edge* Lattice::_newEdge(NodePtr& from, NodePtr& to, unsigned input, unsigned output, Weight cost)
{
  return new Edge(from, to, input, output, LatticeEdgeData(), cost);
}

bool Lattice::_purgeNode(NodePtr& node)
{
  if (node->color() == WFSAcceptor::Black) return node->success();

  if (node->color() == WFSAcceptor::Gray)
    throw jconsistency_error("Node %d is gray; graph is not acyclic.",
			     node->index());

  // paint node gray, then finish it
  bool success = node->isFinal();
  node->setColor(WFSAcceptor::Gray);
  for (Node::Iterator itr(node); itr.more(); itr++) {
    EdgePtr& edge(Cast<EdgePtr>(itr.edge()));
    success = (_purgeNode(edge->next()) || success);
  }

  // paint node black, and place at front of linked list
  node->setSuccess(success);
  node->setColor(WFSAcceptor::Black);
  if (success)
    _sortedNodes.push_front(node);

  return success;
}

void Lattice::_removeUnsuccessful()
{
  // remove unreachable nodes
  list<unsigned> nodes2BRemoved;
  for (_NodeVectorIterator itr = _allNodes().begin(); itr != _allNodes().end(); itr++) {
    NodePtr nd(*itr);
    if (nd.isNull()) continue;
    if (nd->success() == false)
      nodes2BRemoved.push_back(nd->index());
  }
  for (list<unsigned>::iterator itr = nodes2BRemoved.begin(); itr != nodes2BRemoved.end(); itr++)
    _allNodes()[*itr] = NULL;

  // remove unreachable final nodes
  list<_NodeMapIterator> finalNodes2BRemoved;
  for (_NodeMapIterator itr = _finis().begin(); itr != _finis().end(); itr++) {
    NodePtr nd((*itr).second);
    if (nd->success() == false)
      finalNodes2BRemoved.push_back(itr);
  }
  for (list<_NodeMapIterator>::iterator itr = finalNodes2BRemoved.begin(); itr != finalNodes2BRemoved.end(); itr++)
    _finis().erase(*itr);

  // renumber remaining nodes
  unsigned idx = 0;
  for (_Iterator itr = _sortedNodes.begin(); itr != _sortedNodes.end(); itr++)
    (*itr)->_setIndex(idx++);
}

// purge nodes from which an end node cannot be reached
void Lattice::purge()
{
  unsigned ttlNodes = 1 + _allNodes().size() + _finis().size();
  printf("%d nodes before purging.\n", ttlNodes);

  setColor(White);  _setSuccess(false);

  _clearSorted();  _purgeNode(initial());  _removeUnsuccessful();

  printf("%d nodes remain after purging.\n", _sortedNodes.size());
}

// these routines sort lattice nodes topologically
void Lattice::_setSuccess(bool s)
{
  initial()->setSuccess(s);
  for (_NodeVectorIterator itr = _allNodes().begin(); itr != _allNodes().end(); itr++) {
    NodePtr nd(*itr);
    if (nd.isNull()) continue;
    nd->setSuccess(s);
  }
  for (_NodeMapIterator itr = _finis().begin(); itr != _finis().end(); itr++) {
    NodePtr nd((*itr).second);
    nd->setSuccess(s);
  }
}

void Lattice::_visitNode(NodePtr& node)
{
  if (node->color() == Black) return;

  if (node->color() == Gray)
    throw jconsistency_error("Node %d is gray; graph is not acyclic.",
			     node->index());

  // paint node gray, then finish it
  node->setColor(Gray);
  for (Node::Iterator itr(node); itr.more(); itr++) {
    EdgePtr& edge(Cast<EdgePtr>(itr.edge()));
    _visitNode(edge->next());
  }

  // paint node black, and place at front of linked list
  node->setColor(Black);
  _sortedNodes.push_front(node);
}

void Lattice::_topoSort()
{
  if (_sortedNodes.size() > 0) return;

  setColor(White);

  _visitNode(initial());

  printf("Topo sorted %d nodes.\n", _sortedNodes.size());
}

void Lattice::_clearSorted()
{
  _sortedNodes.erase(_sortedNodes.begin(), _sortedNodes.end());
}

WFSTSortedInputPtr Lattice::createPhoneLattice(LatticePtr& lattice, double acScale)
{
  /*
  LexiconPtr stateLexicon(new Lexicon("State Lattice"));
  LexiconPtr inputLexicon(new Lexicon("Input Lattice"));
  LexiconPtr outputLexicon(new Lexicon("Output Lattice"));

  WFSTSortedInputPtr phoneLattice(new WFSTSortedInput(stateLexicon, inputLexicon, outputLexicon));
  */

  WFSTSortedInputPtr phoneLattice(new WFSTSortedInput(stateLexicon(), inputLexicon(), outputLexicon()));

  _topoSort();
  for (_Iterator itr = _snodes().begin(); itr != _snodes().end(); itr++) {
    WFSTSortedInput::NodePtr node(phoneLattice->find((*itr)));
    for (Node::Iterator nitr(*itr); nitr.more(); nitr++) {
      EdgePtr& edge(Cast<EdgePtr>(nitr.edge()));
      unsigned index = _calculateIndex(edge);
      WFSTSortedInput::NodePtr nextNode(phoneLattice->find(edge->next()));
      Weight cost(acScale * edge->data().ac() + edge->data().lm());

      WFSTSortedInput::EdgePtr newEdge((WFSTSortedInput::Edge*) phoneLattice->_newEdge(node, nextNode, index, 0, cost));
      node->_addEdgeForce(newEdge);
    }
    if ((*itr)->isFinal())
      _addFinal((*itr)->index());
  }

  return phoneLattice;
}

unsigned Lattice::_calculateIndex(const EdgePtr& edge)
{
  unsigned start  = edge->data().start();
  unsigned end    = edge->data().end();
  unsigned output = edge->output();

  return output + start >> 10 + end >> 20;
}


// ----- methods for class `Lattice::Node' -----
//
void Lattice::Node::addEdge(EdgePtr& ed)
{
  EdgePtr ptr(_edges());
  while (ptr.isNull() == false) {
    if (ptr->input() == ed->input() && ptr->output() == ed->output() &&
	ptr->prev() == ed->prev() && ptr->next() == ed->next()) return;
    EdgePtr tmp = ptr->_edges();  ptr = tmp;
  }

  ed->_edges() = _edges();
  _edges() = ed;
}

void Lattice::Node::_removeLinks(double threshold)
{
  EdgePtr prev(NULL);
  EdgePtr curr(_edges());
  while (curr.isNull() == false) {
    if (curr->data().gamma() > threshold) {
      if (prev.isNull())
	_edges() = curr->_edges();
      else
	prev->_edges() = curr->_edges();
    } else {
      prev = curr;
    }

    EdgePtr tmp = curr->_edges();  curr = tmp;
  }
}

MemoryManager<Lattice::Node>& Lattice::Node::memoryManager() {
  static MemoryManager<Node> _MemoryManager("Lattice::Node");
  return _MemoryManager;
}


// ----- methods for class `Lattice::Edge' -----
//
MemoryManager<Lattice::Edge>& Lattice::Edge::memoryManager() {
  static MemoryManager<Edge> _MemoryManager("Lattice::Edge");
  return _MemoryManager;
}


// ----- methods for class `Lattice::EdgeIterator' -----
//
Lattice::EdgeIterator::EdgeIterator(LatticePtr& lat)
{
  // place all edges on linked list
  for (Node::Iterator itr(lat->initial()); itr.more(); itr++) {
    EdgePtr& edge(Cast<EdgePtr>(itr.edge()));
    _edgeList.push_back(edge);
  }

  for (_NodeVectorIterator itr = lat->_allNodes().begin(); itr != lat->_allNodes().end(); itr++) {
    NodePtr node(*itr);
    if (node.isNull()) continue;
    for (Node::Iterator itr(node); itr.more(); itr++) {
      EdgePtr& edge(Cast<EdgePtr>(itr.edge()));
      _edgeList.push_back(edge);
    }
  }

  _itr = _edgeList.begin();
}

Lattice::EdgeIterator::EdgeIterator(Lattice& lat)
{
  // place all edges on linked list
  for (Node::Iterator itr(lat.initial()); itr.more(); itr++) {
    EdgePtr& edge(Cast<EdgePtr>(itr.edge()));
    _edgeList.push_back(edge);
  }

  for (_NodeVectorIterator itr = lat._allNodes().begin(); itr != lat._allNodes().end(); itr++) {
    NodePtr node(*itr);
    if (node.isNull()) continue;
    for (Node::Iterator itr(node); itr.more(); itr++) {
      EdgePtr& edge(Cast<EdgePtr>(itr.edge()));
      _edgeList.push_back(edge);
    }
  }

  _itr = _edgeList.begin();
}


// ----- methods for class `DepthFirstApplyConfidences' -----
//
void DepthFirstApplyConfidences::_expandNode(WFSTransducerPtr& A, WFSTransducer::NodePtr& node)
{
  _depth++;
  node->setColor(WFSAcceptor::Gray);
  for (Iterator itr(A, node); itr.more(); itr++) {
    EdgePtr& edge(Cast<EdgePtr>(itr.edge()));

    if (edge->output() != 0) {
      _wordDepth++;
      if (_confidenceList->word(_wordDepth) != A->outputLexicon()->symbol(edge->output()))
	throw jconsistency_error("Words (%s and %s) do not match.",
				 _confidenceList->word(_wordDepth).c_str(), A->outputLexicon()->symbol(edge->output()).c_str());
    }

    if (_wordDepth >= 0) {
      const Weight& conf(_confidenceList->weight(_wordDepth));
      edge->data().setGamma(float(conf) + edge->data().gamma());
    }

    NodePtr nextNode(Cast<NodePtr>(edge->next()));
    if (nextNode->color() == WFSAcceptor::White)
      _expandNode(A, nextNode);

    if (edge->output() != 0) _wordDepth--;
  }
  node->setColor(WFSAcceptor::Black);
  _depth--;
}

void applyConfidences(LatticePtr& A, const ConfidenceListPtr& confidenceList)
{
  DepthFirstApplyConfidences applyConfidences(confidenceList);
  applyConfidences.apply(A);
}


// ----- methods for class `ConsensusGraph::_EquivalenceClass' -----
//
ConsensusGraph::_EquivalenceClass::_EquivalenceClass(EdgePtr& edge)
  : _symbolX(edge->output()), _startX(edge->data().start()), _endX(edge->data().end()), _score(0.0)
{
  _edgeList.push_back(edge);
}

void ConsensusGraph::_EquivalenceClass::merge(_EquivalenceClass* equiv)
{
  for (_EdgeListIterator itr = equiv->_edgeList.begin(); itr != equiv->_edgeList.end(); itr++) {
    EdgePtr& edge(*itr);
    if (edge->data().start() < _startX) _startX = edge->data().start();
    if (edge->data().end() > _endX) _endX = edge->data().end();
    add(edge);
  }
}


// ----- methods for class `ConsensusGraph' -----
//
ConsensusGraph::ConsensusGraph(LatticePtr& lattice, WFSTLexiconPtr& lexicon)
  : Lattice(LexiconPtr(new Lexicon("Consensus State Lexicon")), lattice->outputLexicon(), lattice->outputLexicon()),
    _lexicon(lexicon)
{
  _intraWordClustering(lattice);
  _interWordClustering(lattice);
  _constructGraph(lattice);
}

ConsensusGraph::_MergeCandidate*
ConsensusGraph::_initializeIntraWordClusters(LatticePtr& lattice)
{
  // find all edges with common word labels, and the same start and end times
  lattice->_topoSort();
  for (_Iterator itr = lattice->_snodes().begin(); itr != lattice->_snodes().end(); itr++) {
    for (Node::Iterator nitr(*itr); nitr.more(); nitr++) {
      EdgePtr& edge(Cast<EdgePtr>(nitr.edge()));
      unsigned outputX = edge->output();
      _EquivalenceMapIterator equiv = _intraWordClusters.find(outputX);
      if (equiv == _intraWordClusters.end()) {
	_EquivalenceClass eclass(edge);
	_EquivalenceList elist;
	elist.push_back(eclass);
	_intraWordClusters[outputX] = elist;
      } else {
	_EquivalenceList& elist((*equiv).second);
	bool foundEquivalence = false;
	for (_EquivalenceListIterator eitr = elist.begin(); eitr != elist.end(); eitr++) {
	  if ((*eitr).start() == edge->data().start() && (*eitr).end() == edge->data().end()) {
	    elist.push_back(edge);
	    foundEquivalence = true;
	    break;
	  }
	}
	if (foundEquivalence == false) elist.push_back(_EquivalenceClass(edge));
      }
    }
  }

  // initialize the list of merge candidates
  _mergeList.clear();
  double           bestScore = -HUGE;
  _MergeCandidate* bestMerge = NULL;
  for (_EquivalenceMapIterator itr = _intraWordClusters.begin(); itr != _intraWordClusters.end(); itr++) {
    _EquivalenceList& elist((*itr).second);
    for (_EquivalenceListIterator eitr1 = elist.begin(); eitr1 != elist.end(); eitr1++) {
      for (_EquivalenceListIterator eitr2 = eitr1; eitr2 != elist.end(); eitr2++) {
	if (eitr2 == eitr1) eitr2++;
	_EquivalenceClass& equiv1(*eitr1);
	_EquivalenceClass& equiv2(*eitr2);
	_MergeCandidate merge(&equiv1, &equiv2);
	double score = _updateInterWordScore(merge);
	if (score > bestScore) {
	  bestScore = score;
	  bestMerge = &merge;
	}
	_mergeList.push_back(merge);
      }
    }
  }

  return bestMerge;
}

ConsensusGraph::_MergeCandidate* ConsensusGraph::_initializeInterWordClusters()
{
  // place all edges on a list
  for (_EquivalenceMapIterator eitr = _intraWordClusters.begin(); eitr != _intraWordClusters.end(); eitr++) {
    _EquivalenceList& elist((*eitr).second);
    for (_EquivalenceListIterator litr = elist.begin(); litr != elist.end(); litr++) {
      _EquivalenceClass& equiv(*litr);
      _interWordClusters.insert(_EquivalencePointerMapValueType(&equiv, equiv));
    }
  }

  // find all interword merge candidates
  _mergeList.clear();
  double bestScore = -HUGE;
  _MergeCandidate* bestMerge = NULL;
  for (_EquivalencePointerMapIterator eitr1 = _interWordClusters.begin(); eitr1 != _interWordClusters.end(); eitr1++) {
    _EquivalenceClass& equiv1((*eitr1).second);
    for (_EquivalencePointerMapIterator eitr2 = _interWordClusters.begin(); eitr2 != _interWordClusters.end(); eitr2++) {
      if (eitr2 == eitr1) eitr2++;
      _EquivalenceClass& equiv2((*eitr2).second);

      // determine if the segments overlap
      if ((equiv1.start() > equiv2.start() && equiv1.start() < equiv2.end()) ||
	  (equiv1.end() > equiv2.start() && equiv1.end() < equiv2.end()) ||
	  (equiv1.start() < equiv2.start() && equiv1.end() > equiv2.end()) ||
	  (equiv1.start() > equiv2.start() && equiv1.end() < equiv2.end())) {
	_MergeCandidate merge(&equiv1, &equiv2);
	double score = _updateInterWordScore(merge);
	if (score > bestScore) {
	  bestScore = score;
	  bestMerge = &merge;
	}
	_mergeList.push_back(merge);
      }
    }
  }

  return bestMerge;
}

ConsensusGraph::_MergeCandidate*
ConsensusGraph::_bestMerge(_EquivalenceClass* first, _EquivalenceClass* second, bool interWord)
{
  // remove the candidate to be merged from the list and perform merge
  for (_MergeListIterator itr = _mergeList.begin(); itr != _mergeList.end(); ++itr) {
    _MergeCandidate& merge(*itr);
    if (merge._first == first && merge._second == second) {
      _mergeList.erase(itr);
      break;
    }
  }
  first->merge(second);
  _EquivalencePointerMapIterator toRemove = _interWordClusters.find(second);
  _interWordClusters.erase(toRemove);

  // update all other candidates
  double bestScore = -HUGE;
  _MergeCandidate* bestMerge = NULL;
  for (_MergeListIterator itr = _mergeList.begin(); itr != _mergeList.end(); ++itr) {
    _MergeCandidate& merge(*itr);
    bool update = false;
    if (merge._first == first || merge._second == first) {
      update = true;
    } else if (merge._first == second) {
      merge._first = first;
      update = true;
    } else if (merge._second == second) {
      merge._second = first;
      update = true;
    }
    if (update) {
      double score = (interWord) ? _updateInterWordScore(merge) : _updateIntraWordScore(merge);
      if (score > bestScore) {
	bestScore = score;
	bestMerge = &merge;
      }
    }
  }

  return bestMerge;
}

// calculate the maximum overlap
double ConsensusGraph::_updateIntraWordScore(_MergeCandidate& merge)
{
  _EdgeList& elist1(merge._first->_edgeList);
  _EdgeList& elist2(merge._second->_edgeList);

  double maxOverlap = 0.0;
  for (_EdgeListIterator itr1 = elist1.begin(); itr1 != elist1.end(); itr1++) {
    EdgePtr& edge1(*itr1);
    for (_EdgeListIterator itr2 = elist2.begin(); itr2 != elist2.end(); itr2++) {
    EdgePtr& edge2(*itr2);
      int startOverlap = max(edge1->data().start(), edge2->data().start());
      int endOverlap   = min(edge1->data().end(), edge2->data().end());
      double overlap   = (endOverlap - startOverlap) /
	((edge1->data().end() - edge1->data().start()) + (edge2->data().end() - edge2->data().start()));
      overlap *= edge1->data().gamma() * edge2->data().gamma();

      if (overlap > maxOverlap)
	maxOverlap = overlap;
    }
  }
  merge._score = maxOverlap;
}

// calculate the (negative) average edit distance
double ConsensusGraph::_updateInterWordScore(_MergeCandidate& merge)
{
  _EdgeList& elist1(merge._first->_edgeList);
  _EdgeList& elist2(merge._second->_edgeList);

  double editDistance = 0.0;
  int    pairsN       = 0;
  for (_EdgeListIterator itr1 = elist1.begin(); itr1 != elist1.end(); itr1++) {
    EdgePtr& edge1(*itr1);
    for (_EdgeListIterator itr2 = elist2.begin(); itr2 != elist2.end(); itr2++) {
      EdgePtr& edge2(*itr2);
      editDistance += _lexicon->editDistance(edge1->output(), edge2->output());
      pairsN++;
    }
  }
  merge._score = - editDistance / pairsN;

  return merge._score;
}

void ConsensusGraph::_intraWordClustering(LatticePtr& lattice)
{
  bool interWord = false;

  _MergeCandidate* best = _initializeIntraWordClusters(lattice);
  do {
    best = _bestMerge(best->_first, best->_second, interWord);
  } while (best != NULL);
}

void ConsensusGraph::_interWordClustering(LatticePtr& lattice)
{
  bool interWord = true;

  _MergeCandidate* best = _initializeInterWordClusters();
  do {
    best = _bestMerge(best->_first, best->_second, interWord);
  } while (best != NULL);
}

void ConsensusGraph::_constructGraph(LatticePtr& lattice)
{
  // sort the equivalence classes according to start time
  for (_EquivalencePointerMapIterator itr = _interWordClusters.begin(); itr != _interWordClusters.end(); itr++)
    _sortedClusters.push_back((*itr).first);
  sort(_sortedClusters.begin(), _sortedClusters.end() - 1, LessThan());

  // construct the graph
  unsigned stateX = 0;
  bool create = true;
  _initial = _newNode(stateX++);
  NodePtr node(find(stateX++, create));
  unsigned symbol = outputLexicon()->index("<s>");
  EdgePtr edge(Cast<EdgePtr>(_newEdge(Cast<NodePtr>(_initial), node, symbol, symbol, Weight(0.0))));
  _initial->_addEdgeForce(edge);
  for (_SortedClustersIterator itr = _sortedClusters.begin(); itr != _sortedClusters.end(); itr++) {
    _EquivalenceClass& eclass(*(*itr));
    _EdgeList& elist(eclass._edgeList);
    
    PosteriorProbabilityMap posteriorProbs;
    for (_EdgeListIterator eitr = elist.begin(); eitr != elist.end(); eitr++) {
      EdgePtr& edge(*eitr);
      PosteriorProbabilityMapIterator pitr = posteriorProbs.find(edge->output());
      if (pitr == posteriorProbs.end()) {
	posteriorProbs[edge->output()] = edge->data().gamma();
      } else {
	(*pitr).second += edge->data().gamma();
      }
    }

    NodePtr nextNode(find(stateX++, create));
    for (PosteriorProbabilityMapIterator pitr = posteriorProbs.begin(); pitr != posteriorProbs.end(); pitr++) {
      unsigned symbol    = (*pitr).first;
      double   posterior = (*pitr).second;
      EdgePtr  edge(Cast<EdgePtr>(_newEdge(node, nextNode, symbol, symbol, Weight(posterior))));
      node->_addEdgeForce(edge);
    }
    node = nextNode;
  }
  _addFinal(stateX);
  NodePtr endNode(find(stateX));
  symbol = outputLexicon()->index("</s>");
  edge = Cast<EdgePtr>(_newEdge(node, endNode, symbol, symbol, Weight(0.0)));
}

ConsensusGraphPtr createConsensusGraph(LatticePtr& lattice, WFSTLexiconPtr& lexicon)
{
  ConsensusGraphPtr consensus(new ConsensusGraph(lattice, lexicon));
  return consensus;
}
