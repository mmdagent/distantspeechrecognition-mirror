//
//                                Enigma
//                    Finite State Transducer Library
//                                (fst)
//
//  Module:  fst.fst
//  Purpose: Representation and manipulation of finite state machines.
//  Author:  John McDonough and Emilian Stoimenov
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

#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include "common/mlist.h"
#include "fsm/fsm.h"
#include "dictionary/distribTree.h"


// add negative log-probabilities
LogDouble logAdd(LogDouble ap, LogDouble bp)
{
  if (ap > LogZero)
    throw jconsistency_error("ap (%g) > LogZero (%g)", ap, LogZero);

  if (bp > LogZero)
    throw jconsistency_error("bp (%g) > LogZero (%g)", bp, LogZero);

  if (ap > bp) { LogDouble temp = ap; ap = bp; bp = temp; }

  LogDouble diff = ap - bp;
  //  if (diff < -LogZero) return ap;

  double z = exp(diff);
  if (isnan(z))
    throw jconsistency_error("ap - bp = %g returned NaN.", z);

  return ap - log(1.0 + z);
}

Weight logAdd(Weight a, Weight b)
{
  LogDouble ap = float(a);
  LogDouble bp = float(b);

  if (ap > LogZero)
    throw jconsistency_error("ap (%g) > LogZero (%g)", ap, LogZero);

  if (bp > LogZero)
    throw jconsistency_error("bp (%g) > LogZero (%g)", bp, LogZero);

  if (ap > bp) { LogDouble temp = ap; ap = bp; bp = temp; }

  LogDouble diff = ap - bp;
  // if (diff < -LogZero) return Weight(float(ap));

  double z = exp(diff);
  if (isnan(z))
    throw jconsistency_error("ap - bp = %g returned NaN.", z);

  return Weight(float(ap - log(1.0 + z)));
}

// ----- methods for class `WFSAcceptor' -----
//
WFSAcceptor::WFSAcceptor(LexiconPtr& inlex)
  : _totalNodes(0), _totalFinalNodes(0), _totalEdges(0),
    _inputLexicon(inlex), _initial(NULL) { }

WFSAcceptor::WFSAcceptor(LexiconPtr& statelex, LexiconPtr& inlex, const String& name)
  : _name(name), _totalNodes(0), _totalFinalNodes(0), _totalEdges(0),
    _stateLexicon(statelex), _inputLexicon(inlex),
    _initial(NULL) { }

WFSAcceptor::~WFSAcceptor() { _clear(); }

// Note: Must explicitly clear nodes because all links hold smart
//	 pointers to their 'from' nodes.
void WFSAcceptor::_clear()
{
  // printf("Clearing WFSM\n");  fflush(stdout);

  if (_initial.isNull() == false) { _initial->_clear(); _initial = NULL; }

  for (_NodeVectorIterator itr = _nodes.begin(); itr != _nodes.end(); itr++) {
    NodePtr& node(*itr);
    if (node.isNull()) continue;
    node->_clear();
  }
  _nodes.erase(_nodes.begin(), _nodes.end());

  for (_NodeMapIterator itr = _final.begin(); itr != _final.end(); itr++)
    (*itr).second->_clear();
  _final.erase(_final.begin(), _final.end());

  _totalNodes = 0;  _totalFinalNodes = 0;  _totalEdges = 0;
}

bool WFSAcceptor::hasFinal(unsigned state)
{
  _NodeMapIterator itr = _final.find(state);
  return itr != _final.end();
}

void WFSAcceptor::_addFinal(unsigned state, Weight cost)
{
  if (hasFinal(state))
    throw jconsistency_error("Automaton already has final node %d.", state);

  if (state >= _nodes.size() || _nodes[state].isNull()) {
    NodePtr ptr(_newNode(state));
    _final.insert(_ValueType(state, ptr));
    ptr->_setCost(cost);
  } else {
    NodePtr& node(_nodes[state]);
    node->_setCost(cost);
    _final.insert(_ValueType(state, node));
    _nodes[state] = NULL;
  }
  _totalFinalNodes++;
}

void WFSAcceptor::_resize(unsigned state)
{
  unsigned initialSize = _nodes.size();
  unsigned sz          = initialSize;

  do {
    sz *= 2; sz += 1;
  } while (sz <= state);
  _nodes.reserve(sz);
  for (unsigned i = initialSize; i < sz; i++)
    _nodes.push_back(NodePtr(NULL));
}

WFSAcceptor::NodePtr WFSAcceptor::_find(unsigned state, bool create)
{
  if (_initial.isNull()) { _initial = _newNode(state); return _initial; }
    
  if (initial()->index() == state)
    return initial();

  _NodeMapIterator itr = _final.find(state);

  if (itr != _final.end()) {
    assert((*itr).second->isFinal() == true);
    return (*itr).second;
  }

  if (state < _nodes.size() && _nodes[state].isNull() == false)
    return _nodes[state];

  if (create == false)
    throw jkey_error("No state %u exists.", state);

  if (state >= _nodes.size()) _resize(state);

  _nodes[state] = _newNode(state);

  return _nodes[state];
}

WFSAcceptor::Node* WFSAcceptor::_newNode(unsigned state)
{
  return new Node(state);
}

WFSAcceptor::Edge* WFSAcceptor::_newEdge(NodePtr& from, NodePtr& to, unsigned input, unsigned output, Weight cost)
{
  return new Edge(from, to, input, output, cost);
}

static const unsigned MaxCharacters = 1024;
void WFSAcceptor::read(const String& fileName, bool noSelfLoops)
{
  if (fileName == "")
    jio_error("File name is null.");

  _clear();

  printf("\nReading WFSA from file %s\n", fileName.c_str());

  FILE*         fp     = fileOpen(fileName, "r");
  static size_t n      = 0;
  static char*  buffer = NULL;

  while(getline(&buffer, &n, fp) > 0) {
    static char* token[5];
    token[0] = strtok(buffer, " \t\n");

    char* p = NULL;
    unsigned s1 = strtoul(token[0], &p, 0);
    if (p == token[0])
      s1 = _stateLexicon->index(token[0]);

    unsigned i = 0;
    while((i < 4) && ((token[++i] = strtok(NULL, " \t\n")) != NULL) );
    
    if (i == 1) {			// add a final state with zero cost

      _addFinal(s1);

      // printf("Added final node %d.\n", s1);

    } else if (i == 2) {		// add a final state with non-zero cost

      float cost;
      sscanf(token[1], "%f", &cost);
      _addFinal(s1, Weight(cost));

      // printf("Added final node %d with cost %g.\n", s1, cost);

    } else if (i == 3 || i == 4) {	// add an arc

      bool create = true;
      unsigned s2;
      sscanf(token[1], "%u", &s2);

      if (s1 == s2 && noSelfLoops) continue;

      if (_initial.isNull())
	_initial = _newNode(s1);

      NodePtr from(_find(s1, create));
      NodePtr to(_find(s2, create));

      p = NULL;
      unsigned symbol = strtoul(token[2], &p, 0);
      if (p == token[2])
	symbol = _inputLexicon->index(token[2]);

      if (s1 == s2 && symbol == 0) continue;

      float cost = 0.0;
      if (i == 4)
	sscanf(token[3], "%f", &cost);

      EdgePtr edgePtr(_newEdge(from, to, symbol, symbol, Weight(cost)));
      from->_addEdgeForce(edgePtr);  _totalEdges++;

      /*
      printf("Added edge from %d to %d with symbol = %d, cost = %g.\n",
	     s1, s2, input, cost);  fflush(stdout);
      */

    } else
      throw jio_error("Transducer file %s is inconsistent.", fileName.chars());
  }
    
  fileClose( fileName, fp);
}

void WFSAcceptor::write(const String& fileName, bool useSymbols)
{
  FILE* fp;
  if (fileName == "")
    fp = stdout;
  else
    fp = fileOpen(fileName, "w");

  // write edges leaving from initial state
  for (Node::Iterator itr(_initial); itr.more(); itr++)
    if (useSymbols)
      itr.edge()->write(stateLexicon(), inputLexicon(), fp);
    else
      itr.edge()->write(fp);

  // write edges leaving from intermediate states
  for (_NodeVectorIterator itr = _nodes.begin(); itr != _nodes.end(); itr++) {
    NodePtr& node(*itr);
    if (node.isNull()) continue;
    for (Node::Iterator itr(node); itr.more(); itr++)
      if (useSymbols)
	itr.edge()->write(stateLexicon(), inputLexicon(), fp);
      else
	itr.edge()->write(fp);
  }

  // write final states
  for (_NodeMapIterator itr=_final.begin(); itr != _final.end(); itr++) {
    NodePtr& nd((*itr).second);

    // write edges
    for (Node::Iterator itr(nd); itr.more(); itr++)
      if (useSymbols)
	itr.edge()->write(stateLexicon(), inputLexicon(), fp);
      else
	itr.edge()->write(fp);

    // write nodes
    if (useSymbols)
      nd->write(stateLexicon(), fp);
    else
      nd->write(fp);
  }

  if (fp != stdout)
    fileClose( fileName, fp);
}

void WFSAcceptor::reverse(const WFSAcceptorPtr& wfsa)
{
  clear();
  bool create = true;

  // create super initial and final states
  NodePtr& rinitial(initial(WFSTSortedInput::Node::_MaximumIndex-3));
  _addFinal(wfsa->initial()->index());

  // add arcs from final (i.e., initial) node
  // printf("From final (i.e., initial) node:\n");
  NodePtr rfinal(find(wfsa->initial()->index()));
  for (Node::Iterator eitr(wfsa->initial()); eitr.more(); eitr++) {
    const EdgePtr& edge(eitr.edge());
    const NodePtr& node2(edge->next());
    NodePtr rnode2(find(node2->index(), create));
    EdgePtr redge(new Edge(rnode2, rfinal, edge->input(), edge->cost()));
    /*
    redge->write(stateLexicon(), inputLexicon());
    */
    rnode2->_addEdgeForce(redge);
  }  
  
  // add arcs from super initial node
  // printf("From super initial node:\n");
  for (_ConstNodeMapIterator nitr = wfsa->_finis().begin(); nitr != wfsa->_finis().end(); nitr++) {
    const NodePtr& node((*nitr).second);
    NodePtr rnode(find(node->index(), create));
    EdgePtr redge(new Edge(rinitial, rnode, 0, node->cost()));
    /*
    redge->write(stateLexicon(), inputLexicon());
    */
    rinitial->_addEdgeForce(redge);
  }

  // add arcs from final nodes
  // printf("From final nodes:\n");
  for (_ConstNodeMapIterator nitr = wfsa->_finis().begin(); nitr != wfsa->_finis().end(); nitr++) {
    const NodePtr& node1((*nitr).second);
    NodePtr rnode1(find(node1->index(), create));
    for (Node::Iterator eitr(node1); eitr.more(); eitr++) {
      const EdgePtr& edge(eitr.edge());
      const NodePtr& node2(edge->next());
      NodePtr rnode2(find(node2->index(), create));
      EdgePtr redge(new Edge(rnode2, rnode1, edge->input(), edge->cost()));
      /*
      redge->write(stateLexicon(), inputLexicon());
      */
      rnode2->_addEdgeForce(redge);
    }
  }

  // add arcs from internal nodes
  // printf("From internal nodes:\n");
  for (_ConstNodeVectorIterator nitr = wfsa->_allNodes().begin(); nitr != wfsa->_allNodes().end(); nitr++) {
    const NodePtr& node1(*nitr);
    if (node1.isNull()) continue;
    NodePtr rnode1(find(node1->index(), create));
    for (Node::Iterator eitr(node1); eitr.more(); eitr++) {
      const EdgePtr& edge(eitr.edge());
      const NodePtr& node2(edge->next());
      NodePtr rnode2(find(node2->index(), create));
      EdgePtr redge(new Edge(rnode2, rnode1, edge->input(), edge->output(), edge->cost()));
      /*
      redge->write(stateLexicon(), inputLexicon(), outputLexicon());
      */
      rnode2->_addEdgeForce(redge);
    }
  }  
}

bool WFSAcceptor::epsilonCycle(NodePtr& node)
{
  cout << "Searching for epsilon cycles from node " << node->index() << endl;

  set<unsigned> visited;
  return _visit(node, visited);
}

bool WFSAcceptor::_visit(NodePtr& node, set<unsigned>& visited)
{
  unsigned index = node->index();
  if (visited.find(index) != visited.end()) {
    printf("Found epsilon cycle involving node %d.", index);
    return true;
  }
  visited.insert(index);

  for (Node::Iterator itr(node); itr.more(); itr++) {
    EdgePtr& edge(itr.edge());

    if (edge->input() != 0) continue;

    if (_visit(edge->next(), visited)) return true;
  }
  return false;
}

void WFSAcceptor::printStats() const
{
  cout << endl;
  cout << "Weighted Finite-State Acceptor:"                   << endl;
  cout << "    Total Nodes       = " << _totalNodes           << endl;
  cout << "    Total Final Nodes = " << _totalFinalNodes      << endl;
  cout << "    Total Edges       = " << _totalEdges           << endl;
  cout << "    Lexicon Size      = " << _inputLexicon->size() << endl;
  cout << endl;
}

// these routines sort lattice nodes topologically
void WFSAcceptor::setColor(Color c)
{
  if (_initial.isNull() == false)
    _initial->setColor(c);
  for (_NodeVectorIterator itr = _allNodes().begin(); itr != _allNodes().end(); itr++) {
    NodePtr& node(*itr);
    if (node.isNull()) continue;
    node->setColor(c);
  }
  for (_NodeMapIterator itr = _finis().begin(); itr != _finis().end(); itr++) {
    NodePtr& nd((*itr).second);
    nd->setColor(c);
  }
}

const WFSAcceptor::EdgePtr& WFSAcceptor::edges(NodePtr& node) { return node->_edgeList; }


// ----- methods for class `WFSAcceptor::Edge' -----
//
const double WFSAcceptor::Edge::MinimumCost;

WFSAcceptor::Edge::Edge(NodePtr& prev, NodePtr& next, unsigned symbol, Weight cost)
  : _prev(prev), _next(next), _input(symbol), _output(symbol), _cost(cost)
{
  // printf("Adding Edge : Total Edges = %d\n", ++_totalEdges);  fflush(stdout);
}

WFSAcceptor::Edge::Edge(NodePtr& prev, NodePtr& next, unsigned input, unsigned output, Weight cost)
  : _prev(prev), _next(next), _input(input), _output(output), _cost(cost) { }

WFSAcceptor::Edge::~Edge()
{
  // printf("Deleting Edge : Total Edges = %d\n", --_totalEdges);  fflush(stdout);
}

void WFSAcceptor::Edge::write(FILE* fp)
{
  fprintf(fp, "%10d  %10d  %10d",
	  prev()->index(), next()->index(), input());

  if (fabs(float(cost())) < MinimumCost) fprintf(fp, "\n");
  else fprintf(fp, "  %12g\n", float(cost()));
}

void WFSAcceptor::Edge::write(LexiconPtr& statelex, LexiconPtr& arclex, FILE* fp)
{
  if (statelex.isNull() || statelex->size() == 0)
    fprintf(fp, "%10d  %10d  %20s",
	    prev()->index(), next()->index(),
	    (arclex->symbol(input())).c_str());
  else
    fprintf(fp, "%25s  %25s  %20s",
	    (statelex->symbol(prev()->index())).c_str(),
	    (statelex->symbol(next()->index())).c_str(),
	    (arclex->symbol(input())).c_str());

  if (fabs(float(cost())) < MinimumCost) fprintf(fp, "\n");
  else fprintf(fp, "  %12g\n", float(cost()));
}

void WFSAcceptor::Edge::write(LexiconPtr& stateLex, LexiconPtr& inputLex, LexiconPtr& outputLex, FILE* fp)
{
  throw jio_error("Don't use this method!");
}

MemoryManager<WFSAcceptor::Edge>& WFSAcceptor::Edge::memoryManager()
{
  static MemoryManager<Edge> _MemoryManager("WFSAcceptor::Edge");
  return _MemoryManager;
}


// ----- methods for class `WFSAcceptor::Node' -----
//
const unsigned WFSAcceptor::Node::_MaximumIndex = 536870911;

WFSAcceptor::Node::Node(unsigned idx, Weight cost)
  : _cost(0.0), _edgeList(NULL)
{
  static unsigned guid = 100;
  if (idx > _MaximumIndex)
    throw jindex_error("Node index (%d) > %d.", _MaximumIndex);

  _index.index = idx;
  _index.color = unsigned(White);
  _index.final = 0;
  _index.allowPurge = 1;

  if (float(cost) > 0.0) _setCost(cost);
  
  /*
  cout << "Node " << index() << " : Color " << color() << " : Final " << isFinal() << " : Purge " << canPurge() <<  " : ID " << _id << endl;
  */

  // printf("Total Nodes = %d\n", ++TotalNodes);  fflush(stdout);
}

WFSAcceptor::Node::~Node()
{
  _clear();
  // printf("Total Nodes = %d\n", --TotalNodes);  fflush(stdout);
}

void WFSAcceptor::Node::_addEdge(EdgePtr& ed)
{
  for (EdgePtr ptr = _edges(); ptr.isNull() == false; ptr = ptr->_edges())
    if (ptr->input() == ed->input()) return;

  ed->_edges() = _edges();
  _edges() = ed;
}

void WFSAcceptor::Node::_addEdgeForce(EdgePtr& ed)
{
  ed->_edges() = _edges();
  _edges() = ed;
}

void WFSAcceptor::Node::_clear()
{
  // _setCost();
  _edgeList = NULL;
}

void WFSAcceptor::Node::write(FILE* fp)
{
  if (float(_cost) == 0.0)
    fprintf(fp, "%10d\n", index());
  else
    fprintf(fp, "%10d  %12g\n", index(), float(_cost));
}

void WFSAcceptor::Node::write(LexiconPtr& statelex, FILE* fp)
{
  if (statelex.isNull() || statelex->size() == 0) { write(fp);  return; }

  if (float(_cost) == 0.0)
    if (statelex.isNull())
      fprintf(fp, "%10d\n", index());
    else
      fprintf(fp, "%10s\n", (statelex->symbol(index())).c_str());
  else
    if (statelex.isNull())
      fprintf(fp, "%10d  %12g\n", index(), float(_cost));
    else
      fprintf(fp, "%25s  %12g\n",
	      (statelex->symbol(index())).c_str(), float(_cost));
}

void WFSAcceptor::Node::writeArcs(FILE* fp)
{
  EdgePtr edge(_edges());
  while (edge.isNull() == false) {
    edge->write(fp);
    edge = edge->_edges();
  }
}

void WFSAcceptor::Node::writeArcs(LexiconPtr& statelex, LexiconPtr& arclex, FILE* fp)
{
  EdgePtr edge(_edges());
  while (edge.isNull() == false) {
    edge->write(statelex, arclex, fp);
    edge = edge->_edges();
  }
}

unsigned WFSAcceptor::Node::edgesN() const
{
  unsigned cnt = 0;
  EdgePtr elist = _edgeList;
  while (elist.isNull() == false) {
    elist = elist->_edgeList;
    cnt++;
  }
  return cnt;
}

MemoryManager<WFSAcceptor::Node>&  WFSAcceptor::Node::memoryManager()
{
  static MemoryManager<Node> _MemoryManager("WFSAcceptor::Node");
  return _MemoryManager;
}


// ----- methods for class `WFSTransducer' -----
//
WFSTransducer::WFSTransducer(const WFSAcceptorPtr& wfsa, bool convertWFSA, const String& name)
  : WFSAcceptor(wfsa->stateLexicon(), wfsa->inputLexicon(), name), _outputLexicon(wfsa->inputLexicon())
{
  if (convertWFSA) _convert(wfsa);
}

WFSTransducer::NodePtr& WFSTransducer::initial(int idx)
{
  if (_initial.isNull()) _initial = ((idx >= 0) ? new Node(idx) : new Node(0));

  return Cast<NodePtr>(_initial);
}

// this method must be called from the constructor of the *most* derived
// class, or else the wrong 'Node' type will be allocated
void WFSTransducer::_convert(const WFSAcceptorPtr& wfsa)
{
  _clear();

  bool create = true;

  // add edges from initial node  
  _initial = _newNode(wfsa->initial()->index());
  for (WFSAcceptor::Node::Iterator itr(wfsa->initial()); itr.more(); itr++) {
    WFSAcceptor::EdgePtr& oldEdge(itr.edge());
    NodePtr node(find(oldEdge->next()->index(), create));

    EdgePtr edge((Edge*) _newEdge(initial(), node, oldEdge->input(), oldEdge->output(), oldEdge->cost()));
    initial()->_addEdgeForce(edge);
  }

  for (_NodeVectorIterator itr = wfsa->_nodes.begin(); itr != wfsa->_nodes.end(); itr++) {
    WFSAcceptor::NodePtr& oldFrom(*itr);
    if (oldFrom.isNull()) continue;
    NodePtr from(find(oldFrom->index(), create));

    for (WFSAcceptor::Node::Iterator itr(oldFrom); itr.more(); itr++) {
      WFSAcceptor::EdgePtr& oldEdge(itr.edge());
      NodePtr to(find(oldEdge->next()->index(), create));

      EdgePtr edge((Edge*) _newEdge(from, to, oldEdge->input(), oldEdge->output(), oldEdge->cost()));
      from->_addEdgeForce(edge);
    }
  }

  for (_NodeMapIterator itr = wfsa->_final.begin(); itr != wfsa->_final.end(); itr++) {
    WFSAcceptor::NodePtr& oldFrom((*itr).second);
    NodePtr from(find(oldFrom->index(), create));

    for (WFSAcceptor::Node::Iterator itr(oldFrom); itr.more(); itr++) {
      WFSAcceptor::EdgePtr& oldEdge(itr.edge());
      NodePtr to(find(oldEdge->next()->index(), create));

      EdgePtr edge((Edge*) _newEdge(from, to, oldEdge->input(), oldEdge->output(), oldEdge->cost()));
      from->_addEdgeForce(edge);
    }
    _addFinal(oldFrom->index(), oldFrom->cost());
  }
}

void WFSTransducer::_replaceSym(NodePtr& node, unsigned fromX, unsigned toX, bool input)
{
  for (Node::Iterator itr(node); itr.more(); itr++) {
    EdgePtr& edge(itr.edge());
    if (input) {
      if (edge->input() == fromX) edge->_setInput(toX);
    } else{
      if (edge->output() == fromX) edge->_setOutput(toX);
    }
  }
}

void WFSTransducer::replaceSymbol(const String& fromSym, const String& toSym, bool input)
{
  unsigned fromX = (input ? inputLexicon()->index(fromSym) : outputLexicon()->index(fromSym));
  unsigned toX   = (input ? inputLexicon()->index(toSym)   : outputLexicon()->index(toSym));

  _replaceSym(initial(), fromX, toX, input);
  for (_NodeVector::iterator itr = _allNodes().begin(); itr != _allNodes().end(); itr++) {
    NodePtr& node(*itr);
    if (node.isNull()) continue;
    _replaceSym(node, fromX, toX, input);
  }
  for (_NodeMap::iterator itr = _finis().begin(); itr != _finis().end(); itr++)
    _replaceSym((*itr).second, fromX, toX, input);
}

list<String> WFSTransducer::_split(const String& words)
{
  /*
  if (words[0] == " " || words[words.size()-1] == " ")
    throw jconsistency_error("Word string cannot contain leading or trailing white space!");
  */
  if (words.size() == 0)
    throw jdimension_error("String cannot be zero length!");

  list<String> wordlist;
  String::size_type pos = 0, prevPos = 0;
  while ((pos = words.find_first_of(' ', pos)) != String::npos) {
    wordlist.push_back(words.substr(prevPos, pos - prevPos));
    prevPos = ++pos;
  }
  if (prevPos == 0)
    wordlist.push_back(words);
  else
    wordlist.push_back(words.substr(prevPos));

  return wordlist;
}

unsigned WFSTransducer::_highestIndex()
{
  unsigned idx = initial()->index();

  // iterate over non-final nodes
  for (_NodeVector::iterator itr = _allNodes().begin(); itr != _allNodes().end(); itr++) {
    NodePtr& nd(Cast<NodePtr>(*itr));
    if (nd.isNull()) continue;
    if (nd->index() > idx) idx = nd->index();
  }

  // iterate over final nodes
  for (_NodeMap::iterator itr = _finis().begin(); itr != _finis().end(); itr++) {
    NodePtr& nd(Cast<NodePtr>((*itr).second));
    if (nd->index() > idx) idx = nd->index();
  }

  return idx;
}

void WFSTransducer::fromWords(const String& words, const String& end, const String& filler, bool clear, float logprob,
			      const String& garbageWord, float garbageBeginPenalty, float garbageEndPenalty)
{
  if (clear) _clear();

  list<String> wordlist(_split(words));  wordlist.push_back(end);

  if (wordlist.size() == 1)
    throw jconsistency_error("Word string had zero length.");


  Weight initialC(logprob), C(0.0);
  NodePtr begNode(initial(/* index= */ 0));

  if (garbageWord != "") {
    Weight penalty(garbageBeginPenalty);
    EdgePtr edge(new Edge(begNode, begNode, inputLexicon()->index(garbageWord), inputLexicon()->index(garbageWord), penalty));
    begNode->_addEdgeForce(edge);
  }

  bool     create = true;
  unsigned cnt    = _highestIndex() + 1;
  for (list<String>::const_iterator itr = wordlist.begin(); itr != wordlist.end(); itr++) {
    bool     useFiller = inputLexicon()->isPresent(*itr) ? false : true;

    if (useFiller && filler == "")
      throw jkey_error("Could not find \"%s\" in lexicon.", (*itr).c_str());

    /*
    if (useFiller)
      cout << "Replacing " << (*itr) << " with " << filler << endl;
    */

    unsigned word      = useFiller ? inputLexicon()->index(filler) : inputLexicon()->index(*itr);
    NodePtr  endNode   = find(cnt++, create);
    EdgePtr edge(new Edge(begNode, endNode, word, word, (itr == wordlist.begin()) ? initialC : C));
    begNode->_addEdgeForce(edge);
    begNode = endNode;
  }

  if (garbageWord != "") {
    Weight penalty(garbageEndPenalty);
    EdgePtr edge(new Edge(begNode, begNode, inputLexicon()->index(garbageWord), inputLexicon()->index(garbageWord), penalty));
    begNode->_addEdgeForce(edge);
  }

  _addFinal(cnt-1);
}

String WFSTransducer::bestString()
{
  for (Node::Iterator itr(initial()); itr.more(); itr++)
    return _inputLexicon->symbol(itr.edge()->input());
}

WFSTransducer::WFSTransducer(LexiconPtr& statelex, LexiconPtr& inlex, LexiconPtr& outlex, const String& name)
  : WFSAcceptor(statelex, inlex, name), _outputLexicon(outlex) { }

const WFSTransducer::EdgePtr& WFSTransducer::edges(WFSAcceptor::NodePtr& node) { return Cast<EdgePtr>(node->_edgeList); }

void WFSTransducer::reverse(const WFSTransducerPtr& wfst)
{
  typedef _NodeMap::const_iterator ConstNodeIterator;

  clear();
  bool create = true;

  // create super initial and final states
  NodePtr& rinitial(initial(WFSTSortedInput::Node::_MaximumIndex-3));
  _addFinal(wfst->initial()->index());

  // add arcs from final (i.e., initial) node
  // printf("From final (i.e., initial) node:\n");
  NodePtr rfinal(find(wfst->initial()->index()));
  for (Node::Iterator eitr(wfst->initial()); eitr.more(); eitr++) {
    const EdgePtr& edge(eitr.edge());
    const NodePtr& node2(edge->next());
    NodePtr rnode2(find(node2->index(), create));
    EdgePtr redge(new Edge(rnode2, rfinal, edge->input(), edge->output(), edge->cost()));
    /*
    redge->write(stateLexicon(), inputLexicon(), outputLexicon());
    */
    rnode2->_addEdgeForce(redge);
  }  
  
  // add arcs from super initial node
  // printf("From super initial node:\n");
  for (ConstNodeIterator nitr = wfst->_finis().begin(); nitr != wfst->_finis().end(); nitr++) {
    const NodePtr& node((*nitr).second);
    NodePtr rnode(find(node->index(), create));
    EdgePtr redge(new Edge(rinitial, rnode, 0, 0, node->cost()));
    /*
    redge->write(stateLexicon(), inputLexicon(), outputLexicon());
    */
    rinitial->_addEdgeForce(redge);
  }

  // add arcs from final nodes
  // printf("From final nodes:\n");
  for (ConstNodeIterator nitr = wfst->_finis().begin(); nitr != wfst->_finis().end(); nitr++) {
    const NodePtr& node1((*nitr).second);
    NodePtr rnode1(find(node1->index(), create));
    for (Node::Iterator eitr(node1); eitr.more(); eitr++) {
      const EdgePtr& edge(eitr.edge());
      const NodePtr& node2(edge->next());
      NodePtr rnode2(find(node2->index(), create));
      EdgePtr redge(new Edge(rnode2, rnode1, edge->input(), edge->output(), edge->cost()));
      /*
      redge->write(stateLexicon(), inputLexicon(), outputLexicon());
      */
      rnode2->_addEdgeForce(redge);
    }
  }

  // add arcs from internal nodes
  // printf("From internal nodes:\n");
  for (_NodeVector::const_iterator nitr = wfst->_allNodes().begin(); nitr != wfst->_allNodes().end(); nitr++) {
    const NodePtr& node1(*nitr);
    if (node1.isNull()) continue;
    NodePtr rnode1(find(node1->index(), create));
    for (Node::Iterator eitr(node1); eitr.more(); eitr++) {
      const EdgePtr& edge(eitr.edge());
      const NodePtr& node2(edge->next());
      NodePtr rnode2(find(node2->index(), create));
      EdgePtr redge(new Edge(rnode2, rnode1, edge->input(), edge->output(), edge->cost()));
      /*
      redge->write(stateLexicon(), inputLexicon(), outputLexicon());
      */
      rnode2->_addEdgeForce(redge);
    }
  }  
}

WFSAcceptor::Node* WFSTransducer::_newNode(unsigned state)
{
  return new Node(state);
}

WFSAcceptor::Edge* WFSTransducer::_newEdge(NodePtr& from, NodePtr& to, unsigned input, unsigned output, Weight cost)
{
  return new Edge(from, to, input, output, cost);
}

void WFSTransducer::printStats() const
{
  cout << endl;
  cout << "Weighted Finite-State Transducer:"                    << endl;
  cout << "    Total Nodes         = " << _totalNodes            << endl;
  cout << "    Total Final Nodes   = " << _totalFinalNodes       << endl;
  cout << "    Total Edges         = " << _totalEdges            << endl;
  cout << "    Input Lexicon Size  = " << _inputLexicon->size()  << endl;
  cout << "    Output Lexicon Size = " << _outputLexicon->size() << endl;
  cout << endl;
}

void WFSTransducer::read(const String& fileName, bool noSelfLoops)
{
  if (fileName == "")
    jio_error("File name is null.");

  _clear();

  printf("\nReading WFST from file %s\n", fileName.c_str());

  FILE*         fp     = fileOpen(fileName, "r");
  static size_t n      = 1000;
  static char*  buffer = (char*) malloc(1000 * sizeof(char));

  while(getline(&buffer, &n, fp) > 0) {
    // printf("%s\n", buffer);
    static char* token[6];
    token[0] = strtok(buffer, " \t\n");

    char* p = NULL;
    unsigned s1 = strtoul(token[0], &p, 0);
    if (p == token[0])
      s1 = _stateLexicon->index(token[0]);

    unsigned i = 0;
    while((i < 5) && ((token[++i] = strtok(NULL, " \t\n")) != NULL) );
    
    if (i == 1) {			// add a final state with zero cost

      _addFinal(s1);

      // printf("Added final node %d.\n", s1);

    } else if (i == 2) {		// add a final state with non-zero cost

      float cost;
      sscanf(token[1], "%f", &cost);
      _addFinal(s1, Weight(cost));
      
      // printf("Added final node %d with cost %g.\n", s1, cost);

    } else if (i == 4 || i == 5) {	// add an arc

      bool create = true;

      p = NULL;
      unsigned s2 = strtoul(token[1], &p, 0);
      if (p == token[1])
	s2 = _stateLexicon->index(token[1]);

      if (s1 == s2 && noSelfLoops) continue;

      if (_initial.isNull())
	_initial = (Node*) _newNode(s1);

      NodePtr from(find(s1, create));
      NodePtr to(find(s2, create));

      p = NULL;
      unsigned input = strtoul(token[2], &p, 0);
      if (p == token[2])
	input = _inputLexicon->index(token[2]);

      p = NULL;
      unsigned output = strtoul(token[3], &p, 0);
      if (p == token[3])
	output = _outputLexicon->index(token[3]);

      if (s1 == s2 && input == 0 && output == 0) continue;

      float cost = 0.0;
      if (i == 5)
	sscanf(token[4], "%f", &cost);

      EdgePtr edgePtr((Edge*) _newEdge(from, to, input, output, Weight(cost)));
      from->_addEdgeForce(edgePtr);  _totalEdges++;

      /*
      printf("Added edge from %d to %d with input = %s, output = %s, cost = %g.\n",
      s1, s2, inputLexicon()->symbol(input).c_str(), outputLexicon()->symbol(output).c_str(), cost);  fflush(stdout); */
      
    } else
      throw jio_error("Transducer file %s is inconsistent.", fileName.chars());
  }

  fileClose( fileName, fp);
}

// read a transducer in reverse order
void WFSTransducer::reverseRead(const String& fileName, bool noSelfLoops)
{
  if (fileName == "")
    jio_error("File name is null.");

  _clear();

  // create super initial and final states
  NodePtr& rinitial(initial(WFSTSortedInput::Node::_MaximumIndex-3));
  bool initialFlag = false;

  printf("\nReverse reading WFST from file %s\n", fileName.c_str());

  FILE*         fp     = fileOpen(fileName, "r");
  static size_t n      = 1000;
  static char*  buffer = (char*) malloc(1000 * sizeof(char));

  while(getline(&buffer, &n, fp) > 0) {
    // printf("%s\n", buffer);
    static char* token[6];
    token[0] = strtok(buffer, " \t\n");

    char* p = NULL;
    unsigned s1 = strtoul(token[0], &p, 0);
    if (p == token[0])
      s1 = _stateLexicon->index(token[0]);

    unsigned i = 0;
    while((i < 5) && ((token[++i] = strtok(NULL, " \t\n")) != NULL) );
    
    if (i == 1) {			// add a final state with zero cost

      NodePtr rnode(find(s1));
      EdgePtr redge((Edge*) _newEdge(rinitial, rnode, 0, 0));
      rinitial->_addEdgeForce(redge);

      // printf("Added final node %d.\n", s1);

    } else if (i == 2) {		// add a final state with non-zero cost

      float cost;
      sscanf(token[1], "%f", &cost);

      NodePtr rnode(find(s1));
      EdgePtr redge((Edge*) _newEdge(rinitial, rnode, 0, 0, Weight(cost)));
      rinitial->_addEdgeForce(redge);

      // printf("Added final node %d with cost %g.\n", s1, cost);

    } else if (i == 4 || i == 5) {	// add an arc

      bool create = true;

      p = NULL;
      unsigned s2 = strtoul(token[1], &p, 0);
      if (p == token[1])
	s2 = _stateLexicon->index(token[1]);

      if (s1 == s2 && noSelfLoops) continue;

      if (initialFlag == false) { _addFinal(s1);  initialFlag = true; }

      NodePtr from(find(s1, create));
      NodePtr to(find(s2, create));

      p = NULL;
      unsigned input = strtoul(token[2], &p, 0);
      if (p == token[2])
	input = _inputLexicon->index(token[2]);

      p = NULL;
      unsigned output = strtoul(token[3], &p, 0);
      if (p == token[3])
	output = _outputLexicon->index(token[3]);

      if (s1 == s2 && input == 0) continue;

      float cost = 0.0;
      if (i == 5)
	sscanf(token[4], "%f", &cost);

      EdgePtr edgePtr((Edge*) _newEdge(to, from, input, output, Weight(cost)));
      to->_addEdgeForce(edgePtr);  _totalEdges++;

      /*
      printf("Added edge from %d to %d with input = %s, output = %s, cost = %g.\n",
	     s1, s2, inputLexicon()->symbol(input).c_str(), outputLexicon()->symbol(output).c_str(), cost);  fflush(stdout);
      */

    } else
      throw jio_error("Transducer file %s is inconsistent.", fileName.chars());
  }

  fileClose( fileName, fp);
}

// reassign consecutive indices to all nodes
void WFSTransducer::_reindex()
{
  unsigned nodeN = 0;
  initial()->_setIndex(nodeN++);

  _NodeVector newMap(_allNodes());
  _allNodes().erase(_allNodes().begin(), _allNodes().end());
  for (_NodeVector::iterator itr = newMap.begin(); itr != newMap.end(); itr++) {
    NodePtr& node(*itr);
    if (node.isNull()) continue;
    node->_setIndex(nodeN);
    _allNodes()[nodeN++] = node;
  }

  _NodeMap newFinal(_finis());
  _finis().erase(_finis().begin(), _finis().end());
  for (_NodeMap::iterator itr = newFinal.begin(); itr != newFinal.end(); itr++) {
    NodePtr& node((*itr).second);
    node->_setIndex(nodeN);
    _finis().insert(_NodeMap::value_type(nodeN++, node));
  }
}

void WFSTransducer::write(const String& fileName, bool useSymbols)
{
  // _reindex();

  FILE* fp;
  if (fileName == "")
    fp = stdout;
  else
    fp = fileOpen(fileName, "w");

  // write edges leaving from initial state
  for (Node::Iterator itr(initial()); itr.more(); itr++)
    if (useSymbols)
      itr.edge()->write(stateLexicon(), inputLexicon(), outputLexicon(), fp);
    else
      itr.edge()->write(fp);

  // write edges leaving from intermediate states
  for (_NodeVectorIterator itr = _nodes.begin(); itr != _nodes.end(); itr++) {
    NodePtr& nd(Cast<NodePtr>(*itr));
    if (nd.isNull()) continue;

    for (Node::Iterator itr(nd); itr.more(); itr++)
      if (useSymbols)
	itr.edge()->write(stateLexicon(), inputLexicon(), outputLexicon(), fp);
      else
	itr.edge()->write(fp);
  }

  // write final states
  for (_NodeMapIterator itr = _final.begin(); itr != _final.end(); itr++) {
    NodePtr& nd(Cast<NodePtr>((*itr).second));

    // write edges
    for (Node::Iterator itr(nd); itr.more(); itr++)
      if (useSymbols)
	itr.edge()->write(stateLexicon(), inputLexicon(), outputLexicon(), fp);
      else
	itr.edge()->write(fp);

    // write nodes
    if (useSymbols)
      nd->write(stateLexicon(), fp);
    else
      nd->write(fp);
  }

  if (fp != stdout)
    fileClose( fileName, fp);
}

WFSTransducerPtr reverse(const WFSTransducerPtr& wfst)
{
  WFSTransducerPtr reverse(new WFSTransducer(wfst->stateLexicon(), wfst->inputLexicon(), wfst->outputLexicon()));
  reverse->reverse(wfst);

  return reverse;
}


// ----- methods for class `WFSTransducer::Edge' -----
//
void WFSTransducer::Edge::write(FILE* fp)
{
  fprintf(fp, "%10d  %10d  %10d  %10d",
	  prev()->index(), next()->index(), input(), output());

  if (fabs(float(cost())) < MinimumCost) fprintf(fp, "\n");
  else fprintf(fp, "  %12g\n", float(cost()));
}

void WFSTransducer::Edge::write(LexiconPtr& statelex, LexiconPtr& inlex, LexiconPtr& outlex, FILE* fp)
{
  if (statelex.isNull() || statelex->size() == 0)
    fprintf(fp, "%10d  %10d  %10s  %20s",
	    prev()->index(), next()->index(),
	    (inlex->symbol(input())).c_str(),
	    (outlex->symbol(output())).c_str());
  else {
    cout << "quering" << endl;
    cout <<  "Input: " << inlex->symbol(input()) << " Output: " << output() << endl;
    fprintf(fp, "%25s  %25s  %10s  %20s",
	    (statelex->symbol(prev()->index())).c_str(),
	    (statelex->symbol(next()->index())).c_str(),
	    (inlex->symbol(input())).c_str(),
	    (outlex->symbol(output())).c_str());
  }

  if (fabs(float(cost())) < MinimumCost) fprintf(fp, "\n");
  else fprintf(fp, "  %12g\n", float(cost()));
}

MemoryManager<WFSTransducer::Edge>&  WFSTransducer::Edge::memoryManager()
{
  static MemoryManager<Edge> _MemoryManager("WFSTransducer::Edge");
  return _MemoryManager;
}


// ----- methods for class `WFSTransducer::Node' -----
//
void WFSTransducer::Node::writeArcs(FILE* fp)
{
  EdgePtr edge(_edges());
  while (edge.isNull() == false) {
    edge->write(fp);
    edge = edge->_edges();
  }
}

void WFSTransducer::Node::writeArcs(LexiconPtr& statelex, LexiconPtr& inlex, LexiconPtr& outlex, FILE* fp)
{
  EdgePtr edge(_edges());
  while (edge.isNull() == false) {
    edge->write(statelex, inlex, outlex, fp);
    edge = edge->_edges();
  }
}

MemoryManager<WFSTransducer::Node>&  WFSTransducer::Node::memoryManager()
{
  static MemoryManager<Node> _MemoryManager("WFSTransducer::Node");
  return _MemoryManager;
}


// ----- methods for class `ConfidenceList' -----
//
Weight ConfidenceList::weight(int depth) const
{
  if (depth < 0)
    throw jconsistency_error("Depth (%d) less than zero.", depth);

  ConfidenceEntry entry((*this)[unsigned(depth)]);

  return entry._weight;
}

String ConfidenceList::word(int depth) const
{
  if (depth < 0)
    throw jconsistency_error("Depth (%d) less than zero.", depth);

  ConfidenceEntry entry((*this)[unsigned(depth)]);

  return entry._word;
}

void ConfidenceList::binarize(float threshold)
{
  if (threshold < 0.0)
    throw jparameter_error("Negative threshold (%f).", threshold);

  threshold = -log(threshold);
  for (_Iterator itr(*this); itr.more(); itr++) {
    if (float((*itr)._weight) < threshold)
      (*itr)._weight = ZeroWeight;
    else
      (*itr)._weight = LogZeroWeight;
  }
}

void ConfidenceList::write(const String& fileName)
{
  FILE* fp = (fileName == "") ? stdout : fileOpen(fileName, "w");

  fprintf(fp, "Confidence List: %s\n", name().c_str());
  for (_Iterator itr(*this); itr.more(); itr++)
    fprintf(fp, "%20s : %10.4f\n", (*itr)._word.c_str(), float((*itr)._weight));
  fprintf(fp, "\n");  fflush(fp);

  if (fp != stdout)
    fileClose(fileName, fp);
}


// ----- methods for class `WFSTSortedInput' -----
//
const WFSTSortedInput::NodePtr& WFSTSortedInput::whiteNode() {
  static const NodePtr WhiteNode(new Node(WFSTSortedInput::Node::_MaximumIndex-2, White));
  return WhiteNode;
}

const WFSTSortedInput::NodePtr& WFSTSortedInput::grayNode() {
  static const NodePtr GrayNode(new Node(WFSTSortedInput::Node::_MaximumIndex-1, Gray));
  return GrayNode;
}

const WFSTSortedInput::NodePtr& WFSTSortedInput::blackNode() {
  static const NodePtr BlackNode(new Node(WFSTSortedInput::Node::_MaximumIndex, Black));
  return BlackNode;
}

WFSTSortedInput::WFSTSortedInput(const WFSAcceptorPtr& wfsa)
  : WFSTransducer(wfsa, /* convertWFSA= */ false, wfsa->name()+String(" Sorted Input")),
    _findCount(0), _dynamic(false), _A(NULL)
{
  /*
  if (_dynamic) {
    printf("Dynamic = true.\n");  fflush(stdout);
  }
  */

  _convert(wfsa);
}

WFSTSortedInput::WFSTSortedInput(const WFSTransducerPtr& A, bool dynamic)
  : WFSTransducer(A->stateLexicon(), A->inputLexicon(), A->outputLexicon(), A->name()+String(" Sorted Input")),
    _findCount(0), _dynamic(dynamic), _A(A)
{
  /*
  if (_dynamic) {
    printf("Dynamic = true.\n");  fflush(stdout);
  }
  */
}

WFSTSortedInput::WFSTSortedInput(LexiconPtr& statelex, LexiconPtr& inlex, LexiconPtr& outlex,
				 bool dynamic, const String& name)
  : WFSTransducer(statelex, inlex, outlex, name),
    _findCount(0), _dynamic(dynamic), _A(NULL)
{
  /*
  if (_dynamic) {
    printf("Dynamic = true.\n");  fflush(stdout);
  }
  */
}

WFSTSortedInput::NodePtr& WFSTSortedInput::initial(int idx)
{
  if (_initial.isNull())
    if (idx >= 0)
      _initial = new Node(idx);
    else if (_A.isNull())
      _initial = new Node(WFSTSortedInput::Node::_MaximumIndex-3);	// "super" initial state for minimization
    else
      _initial = new Node(_A->initial());				// normal initial state

  return Cast<NodePtr>(_initial);
}

ConfidenceListPtr WFSTSortedInput::_splitConfs(const String& confs)
{
  list<String>     confentries;
  splitList(confs, confentries);

  ConfidenceListPtr conflist(new ConfidenceList("Confidence Entries"));
  for (list<String>::const_iterator itr = confentries.begin(); itr != confentries.end(); itr++) {
    // cout << (*itr) << endl;

    list<String> entry;
    splitList(*itr, entry);

    list<String>::const_iterator eitr = entry.begin();
    const String& word(*eitr);  eitr++;
    float wgt;
    sscanf((*eitr).c_str(), "%f", &wgt);

    /*
    cout << "entry[0] = " << word << endl;
    cout << "entry[1] = " << wgt  << endl;
    */

    wgt = max(double(0.0), double(-log(wgt)));
    conflist->push(ConfidenceEntry(word, Weight(wgt)));
  }

  return conflist;
}

ConfidenceListPtr WFSTSortedInput::fromConfs(const String& confs, const String& end, const String& filler)
{
  _clear();

  ConfidenceListPtr conflist(_splitConfs(confs));
  conflist->push(ConfidenceEntry(end, ZeroWeight));

  bool     create = true;
  unsigned cnt    = 1;
  NodePtr begNode(initial(/* index= */ 0));

  for (ConfidenceList::Iterator itr(conflist); itr.more(); itr++) {
    bool     useFiller = inputLexicon()->isPresent((*itr)._word) ? false : true;

    if (useFiller && filler == "")
      throw jkey_error("Could not find \"%s\" in lexicon.", (*itr)._word.c_str());

    /*
    if (useFiller)
      cout << "Replacing " << (*itr) << " with " << filler << endl;
    */

    unsigned word      = useFiller ? inputLexicon()->index(filler) : inputLexicon()->index((*itr)._word);
    NodePtr  endNode   = find(cnt++, create);
    EdgePtr edge(new Edge(begNode, endNode, word, word));
    begNode->_addEdgeForce(edge);
    begNode = endNode;
  }

  if (cnt == 0)
    throw jconsistency_error("Word string had zero length.");

  _addFinal(cnt-1);

  return conflist;
}

const WFSTSortedInput::EdgePtr& WFSTSortedInput::edges(WFSAcceptor::NodePtr& nd)
{
  bool create = true;

  NodePtr node(Cast<NodePtr>(nd));

  // are edges already expanded?
  if (node->_edgeList.isNull() == false || node->_nodeA.isNull())
    return Cast<const EdgePtr>(node->_edgeList);

  for (WFSTransducer::Node::Iterator itr(_A, node->_nodeA); itr.more(); itr++) {
    WFSTransducer::EdgePtr& edge(itr.edge());
    unsigned input  = edge->input();
    unsigned output = edge->output();
    Weight    c     = edge->cost();
    NodePtr nextNode(find(edge->next()->index(), create));

    EdgePtr newEdge(new Edge(node, nextNode, input, output, c));
    if (edge->next()->isFinal())
      nextNode->_setCost(edge->next()->cost());

    node->_addEdgeForce(edge);
  }

  return Cast<const EdgePtr>(node->_edgeList);
}

void WFSTSortedInput::_unique(unsigned count)
{
  // loop over nodes to purge "old" edges
  for (_NodeVectorIterator itr = _nodes.begin(); itr != _nodes.end(); itr++) {
    NodePtr& node(Cast<NodePtr>(*itr));
    if (node.isNull()) continue;
    
    if (_findCount - node->_lastEdgesCall < count) continue;

    if (node->_edges().isNull() == false) {
      node->_expanded = false;
      node->_edges() = NULL;
    }   
  }

  // loop again to purge "unique" nodes
  unsigned active = 0;
  unsigned purged = 0;
  for (_NodeVectorIterator itr = _nodes.begin(); itr != _nodes.end(); itr++) {
    NodePtr& node(Cast<NodePtr>(*itr));
    if (node.isNull()) continue;

    if (node.unique() == false) {
      if (node != whiteNode() && node != grayNode() && node != blackNode()) active++;
      continue;
    }

    if (node->color() == White)
      node = whiteNode();
    else if (node->color() == Gray)
      node = grayNode();
    else
      node = blackNode();

    purged++;
  }

  /*
  printf("State lexicon %s : Input lexicon %s : Output lexicon %s\n",
	 stateLexicon()->name().c_str(), inputLexicon()->name().c_str(), outputLexicon()->name().c_str());
  */

  printf("%15s : %8d : %d Purged Nodes : %d Active Nodes\n",
	 name().c_str(), _findCount, purged, active);  fflush(stdout);
}

void WFSTSortedInput::purgeUnique(unsigned count)
{
  if (_dynamic) _unique(count);

  if (_A.isNull() == false) _A->purgeUnique(count);
}

WFSTSortedInput::NodePtr WFSTSortedInput::
find(const WFSTransducer::NodePtr& node, bool create)
{
  unsigned index = node->index();

  _NodeMapIterator itr = _final.find(index);

  if (itr != _final.end()) {
    assert((*itr).second->isFinal() == true);
    return Cast<NodePtr>((*itr).second);
  }

  if (index < _nodes.size() && _nodes[index].isNull() == false)
    return Cast<NodePtr>(_nodes[index]);

  if (create == false)
    throw jkey_error("Could not find sorted input node %d.", index);

  NodePtr newNode((Node*) _newNode(node, White, node->cost()));
  if (node->isFinal()) {
    newNode->_setCost(node->cost());
    _final.insert(WFSAcceptor::_ValueType(index, newNode));
    itr = _final.find(index);
    return Cast<NodePtr>((*itr).second);
  } else {
    if (index >= _nodes.size()) _resize(index);
    _nodes[index] = newNode;
    return Cast<NodePtr>(_nodes[index]);
  }
}

WFSAcceptor::Node* WFSTSortedInput::_newNode(unsigned state)
{
  return new Node(state);
}

WFSAcceptor::Node* WFSTSortedInput::_newNode(const WFSTransducer::NodePtr& node, Color col, Weight cost)
{
  return new Node(node, col, cost);
}

WFSAcceptor::Edge* WFSTSortedInput::
_newEdge(NodePtr& from, NodePtr& to, unsigned input, unsigned output, Weight cost)
{
  return new Edge(from, to, input, output, cost);
}


// ----- methods for class `WFSTSortedInput::Edge' -----
//
MemoryManager<WFSTSortedInput::Edge>&  WFSTSortedInput::Edge::memoryManager()
{
  static MemoryManager<Edge> _MemoryManager("WFSTSortedInput::Edge");
  return _MemoryManager;
}


// ----- methods for class `WFSTSortedInput::Node' -----
//
// insert edge based on input symbol, if not present
void WFSTSortedInput::Node::_addEdge(WFSAcceptor::EdgePtr& newEdge)
{
  EdgePtr ptr    = Cast<const EdgePtr>(_edgeList);
  EdgePtr oldPtr = ptr;
  while (ptr.isNull() == false && ptr->input() < newEdge->input()) {
    oldPtr = ptr;
    ptr    = ptr->_edges();
  }
  while (ptr.isNull() == false && ptr->input() == newEdge->input() && ptr->output() < newEdge->output()) {
    oldPtr = ptr;
    ptr    = ptr->_edges();
  }

  if (ptr.isNull() == false && ptr->input() == newEdge->input()) return;

  if (ptr == oldPtr) {

    newEdge->_edges() = _edgeList;
    _edgeList  = newEdge;

  } else {

    newEdge->_edges() = ptr;
    oldPtr->_edges()  = Cast<EdgePtr>(newEdge);

  }
}

// insert edge based on input symbol
void WFSTSortedInput::Node::_addEdgeForce(WFSAcceptor::EdgePtr& newEdge)
{
  EdgePtr ptr    = Cast<const EdgePtr>(_edges());
  EdgePtr oldPtr = ptr;
  while (ptr.isNull() == false && ptr->input() < newEdge->input()) {
    oldPtr = ptr;
    ptr    = ptr->_edges();
  }
  while (ptr.isNull() == false && ptr->input() == newEdge->input() && ptr->output() < newEdge->output()) {
    oldPtr = ptr;
    ptr    = ptr->_edges();
  }

  if (ptr == oldPtr) {
    newEdge->_edges() = _edgeList;
    _edgeList  = newEdge;

  } else {

    newEdge->_edges() = ptr;
    oldPtr->_edges()  = Cast<EdgePtr>(newEdge);

  }
}

MemoryManager<WFSTSortedInput::Node>&  WFSTSortedInput::Node::memoryManager()
{
  static MemoryManager<Node> _MemoryManager("WFSTSortedInput::Node");
  return _MemoryManager;
}


// ----- definition for class `WeightPusher' -----
//
void WeightPusher::push()
{
  WFSTransducerPtr reverse(new WFSTransducer(_wfst->stateLexicon(), _wfst->inputLexicon(), _wfst->outputLexicon()));
  reverse->reverse(_wfst);

  _resetPotentials(reverse);
  _calculatePotentials(reverse);
  /*
  _printPotentials();
  */
  _reweightNodes();
}

void WeightPusher::_resetPotentials(const WFSTransducerPtr& reverse)
{
  _potentialMap.erase(_potentialMap.begin(), _potentialMap.end());

  _Potential zero(reverse->initial(), ZeroWeight, ZeroWeight);
  _potentialMap.insert(_ValueType(reverse->initial()->index(), zero));

  for (_NodeVectorConstIterator nitr = reverse->_allNodes().begin(); nitr != reverse->_allNodes().end(); nitr++) {
    const _NodePtr& node(*nitr);
    if (node.isNull()) continue;

    _Potential zero(node, LogZeroWeight, LogZeroWeight);
    _potentialMap.insert(_ValueType(node->index(), zero));
  }

  for (_NodeMapConstIterator nitr = reverse->_finis().begin(); nitr != reverse->_finis().end(); nitr++) {
    const _NodePtr& node((*nitr).second);

    _Potential zero(node, LogZeroWeight, LogZeroWeight);
    _potentialMap.insert(_ValueType(node->index(), zero));
  }
}

void WeightPusher::_calculatePotentials(const WFSTransducerPtr& reverse)
{
  while (_queue.empty() == false) _queue.pop();
  _set.erase(_set.begin(), _set.end());

  // insert initial node on queue
  _queue.push(reverse->initial()->index());
  _set.insert(reverse->initial()->index());

  // iterate until queue is empty
  unsigned cnt = 0;
  while (_queue.empty() == false) {

    if (++cnt % 100000 == 0) { printf("%6d nodes on stack\n", _queue.size());  fflush(stdout); }

    // if (cnt % 1000000 == 0) return;

    unsigned index = _queue.front();  _queue.pop();
    _SetIterator sitr = _set.find(index);  _set.erase(sitr);

    _Iterator pitr = _potentialMap.find(index);
    _Potential& potential((*pitr).second);

    Weight     R = potential._r;
    potential._r = LogZeroWeight;
    const _NodePtr& node(potential._node);

    for (WFSTransducer::Node::Iterator eitr(node); eitr.more(); eitr++) {
      WFSTransducer::EdgePtr& edge(eitr.edge());
      Weight arcWeight(edge->cost());

      if (edge->next() == node && arcWeight == ZeroWeight) continue;

      _Iterator nitr = _potentialMap.find(edge->next()->index());
      _Potential& nextPotential((*nitr).second);
      Weight newWeight = _Semiring::oplus(nextPotential._d, _Semiring::otimes(R, arcWeight));
      if (_isEqual(newWeight, nextPotential._d) == false) {
	nextPotential._d = newWeight;
	nextPotential._r = _Semiring::oplus(nextPotential._r, _Semiring::otimes(R, arcWeight));

	if (_set.find(edge->next()->index()) == _set.end()) {
	  _queue.push(edge->next()->index());
	  _set.insert(edge->next()->index());
	}
      }
    }
  }
}

void WeightPusher::_printPotentials() const
{
  for (_ConstIterator itr = _potentialMap.begin(); itr != _potentialMap.end(); itr++)
    (*itr).second.print();
}

void WeightPusher::_reweightNodes()
{
  // assign zero weight to all final nodes and reweight arcs
  for (_NodeMapIterator nitr = _wfst->_finis().begin(); nitr != _wfst->_finis().end(); nitr++) {
    _NodePtr& node((*nitr).second);
    node->_setCost(ZeroWeight);
    _reweightArcs(node);
  }

  // reweight arcs from intermediate nodes
  for (_NodeVectorIterator nitr = _wfst->_allNodes().begin(); nitr != _wfst->_allNodes().end(); nitr++) {
    _NodePtr& node(*nitr);
    if (node.isNull()) continue;
    _reweightArcs(node);
  }

  // reweight arcs from initial node
  _reweightArcs(_wfst->initial(), /* initFlag= */ true);
}

void WeightPusher::_reweightArcs(WFSTransducer::NodePtr& node, bool initFlag)
{
  _ConstIterator witr = _potentialMap.find(node->index());

  if (witr == _potentialMap.end())
    throw jconsistency_error("Could not find potential for node %d.\n", node->index());

  Weight origWeight(_Semiring::inverse((*witr).second._d));
  for (WFSTransducer::Node::Iterator eitr(node); eitr.more(); eitr++) {
    WFSTransducer::EdgePtr& edge(eitr.edge());

    if (edge->next() == node && edge->cost() == ZeroWeight) continue;

    witr = _potentialMap.find(edge->next()->index());

    if (witr == _potentialMap.end())
      throw jconsistency_error("Could not find potential for node %d.\n", edge->next()->index());

    Weight nextPotential((*witr).second._d);
    Weight arcWeight(_Semiring::otimes(edge->cost(), nextPotential));
    if (initFlag == false)
      arcWeight = _Semiring::otimes(arcWeight, origWeight);

    Cast<Weight>(edge->_cost) = arcWeight;

    /*
    printf("inv(V(p[e])) = %g : w[e] = %g : V(n[e]) = %g\n", float(origWeight), float(edge->cost()), float(nextPotential));
    edge->write(_wfst->stateLexicon(), _wfst->inputLexicon(), _wfst->outputLexicon());
    */    
  }
}

void pushWeights(WFSTransducerPtr& wfst)
{
  WeightPusher pusher(wfst);  pusher.push();
}


// ----- definition for class `WFSTEpsilonRemoval' -----
//
void WFSTEpsilonRemoval::_discoverClosure(const _NodePtr& node)
{
  _potentialMap.erase(_potentialMap.begin(), _potentialMap.end());

  _Potential zero(node, ZeroWeight, ZeroWeight);
  _potentialMap.insert(_ValueType(node->index(), zero));

  _Queue fifo;  fifo.push(node->index());
  _Set   set;
  while (fifo.empty() == false) {
    unsigned index = fifo.front();  fifo.pop();
    _NodePtr searchNode(_A->find(index));

    for (WFSTransducer::Node::Iterator itr(searchNode); itr.more(); itr++) {
      if (itr.edge()->input() != 0) continue;
      const _NodePtr nextNode(itr.edge()->next());
      if (set.find(nextNode->index()) != set.end()) continue;

      fifo.push(nextNode->index());  set.insert(nextNode->index());
      _Potential logZero(nextNode, LogZeroWeight, LogZeroWeight);
      _potentialMap.insert(_ValueType(nextNode->index(), logZero));
    }
  }
}

unsigned WFSTEpsilonRemoval::_String(unsigned strq, unsigned syme)
{
  if (strq == 0)
    return syme;
  if (syme == 0)
    return strq;

  // create a new multi-word symbol
  bool create = true;
  String string = outputLexicon()->symbol(strq) + ":" + outputLexicon()->symbol(syme);

  printf("Created new symbol %s\n", string.c_str());

  return outputLexicon()->index(string, create);
}

void WFSTEpsilonRemoval::_calculateClosure(const _NodePtr& node)
{
  while (_queue.empty() == false) _queue.pop();
  _set.erase(_set.begin(), _set.end());

  // insert initial node on queue
  _queue.push(node->index());
  _set.insert(node->index());

  // iterate until stack is empty
  while (_queue.empty() == false) {

    unsigned index = _queue.front();  _queue.pop();
    _SetIterator sitr = _set.find(index);  _set.erase(sitr);

    _Iterator pitr = _potentialMap.find(index);

    if (pitr == _potentialMap.end())
      throw jconsistency_error("Could not find potential for node %d.", index);

    _Potential& potential((*pitr).second);

    Weight     R = potential._r;
    unsigned   s = potential._s;
    potential._r = LogZeroWeight;
    const _NodePtr node(potential._node);

    for (WFSTransducer::Node::Iterator eitr(node); eitr.more(); eitr++) {
      if (eitr.edge()->input() != 0) continue;
      WFSTransducer::EdgePtr& edge(eitr.edge());
      Weight arcWeight(edge->cost());

      _Iterator nitr = _potentialMap.find(edge->next()->index());
      _Potential& nextPotential((*nitr).second);
      Weight newWeight = _Semiring::oplus(nextPotential._d, _Semiring::otimes(R, arcWeight));
      if (_isEqual(newWeight, nextPotential._d) == false) {
	nextPotential._d = newWeight;
	nextPotential._r = _Semiring::oplus(nextPotential._r, _Semiring::otimes(R, arcWeight));
	nextPotential._s = _String(s, edge->output());

	if (_set.find(edge->next()->index()) == _set.end()) {
	  _queue.push(edge->next()->index());
	  _set.insert(edge->next()->index());
	}
      }
    }
  }
}

const WFSTEpsilonRemoval::EdgePtr& WFSTEpsilonRemoval::edges(WFSAcceptor::NodePtr& nd)
{
  NodePtr node(Cast<NodePtr>(nd));

  // are edges already expanded?
  if (node->_expanded || node->_nodeA.isNull())
    return node->_edges();

  _discoverClosure(node->_nodeA);
  _calculateClosure(node->_nodeA);

  bool create = true;
  for (_Iterator itr = _potentialMap.begin(); itr != _potentialMap.end(); itr++) {
    const _Potential& potential((*itr).second);
    const _NodePtr pnode(potential._node);
    const Weight& d(potential._d);
    unsigned s(potential._s);

    if (pnode->isFinal()) {
      NodePtr nextNode(find(pnode, create));
      if (node != nextNode) {
	EdgePtr edgePtr(new Edge(node, nextNode, /* input= */ 0, /* output= */ 0, /* cost= */ d));
	node->_addEdgeForce(edgePtr);
      }
    }

    for (WFSTransducer::Node::Iterator eitr(pnode); eitr.more(); eitr++) {
      WFSTransducer::EdgePtr& edge(eitr.edge());

      if (edge->input() == 0 && edge->next()->isFinal() == false) continue;

      Weight arcWeight(_Semiring::otimes(d, edge->cost()));
      unsigned strne(_String(s, edge->output()));

      NodePtr nextNode(find(edge->next(), create));
      EdgePtr edgePtr(new Edge(node, nextNode, edge->input(), strne, arcWeight));
      node->_addEdgeForce(edgePtr);
    }
  }

  node->_expanded = true;  node->_lastEdgesCall = _findCount;
  return node->_edges();
}

WFSTEpsilonRemoval::NodePtr& WFSTEpsilonRemoval::initial(int idx)
{
  if (_initial.isNull())
    /*
    if (idx >= 0)
      _initial = new Node(idx);
    else
    */
      _initial = new Node(_A->initial());	// normal initial state

  return Cast<NodePtr>(_initial);
}

WFSAcceptor::Node* WFSTEpsilonRemoval::_newNode(unsigned state)
{
  return new Node(state);
}

WFSAcceptor::Node* WFSTEpsilonRemoval::_newNode(const WFSTransducer::NodePtr& node, Color col, Weight cost)
{
  return new Node(node, col, cost);
}

WFSAcceptor::Edge* WFSTEpsilonRemoval::
_newEdge(NodePtr& from, NodePtr& to, unsigned input, unsigned output, Weight cost)
{
  return new Edge(from, to, input, output, cost);
}

WFSTEpsilonRemovalPtr removeEpsilon(WFSTransducerPtr& wfst)
{
  WFSTEpsilonRemovalPtr remove(new WFSTEpsilonRemoval(wfst));
  breadthFirstSearch(remove);

  return remove;
}


// ----- methods for class `WFSTEpsilonRemoval::Edge' -----
//
MemoryManager<WFSTEpsilonRemoval::Edge>&  WFSTEpsilonRemoval::Edge::memoryManager()
{
  static MemoryManager<Edge> _MemoryManager("WFSTEpsilonRemoval::Edge");
  return _MemoryManager;
}


// ----- methods for class `WFSTEpsilonRemoval::Node' -----
//
MemoryManager<WFSTEpsilonRemoval::Node>& WFSTEpsilonRemoval::Node::memoryManager()
{
  static MemoryManager<Node> _MemoryManager("WFSTEpsilonRemoval::Node");
  return _MemoryManager;
}


// ----- methods for class `WFSTEpsilonMinusOneRemoval' -----
//
const WFSTEpsilonMinusOneRemoval::EdgePtr& WFSTEpsilonMinusOneRemoval::edges(WFSAcceptor::NodePtr& nd)
{
  NodePtr node(Cast<NodePtr>(nd));

  // are edges already expanded?
  if (node->_expanded || node->_nodeA.isNull())
    return node->_edges();

  _discoverClosure(node->_nodeA);
  _calculateClosure(node->_nodeA);

  // add non-epsilon arcs from current node
  bool create = true;
  for (WFSTransducer::Node::Iterator eitr(node->_nodeA); eitr.more(); eitr++) {
    WFSTransducer::EdgePtr& edge(eitr.edge());

    if (edge->input() == 0) continue;

    NodePtr nextNode(find(edge->next(), create));
    EdgePtr edgePtr(new Edge(node, nextNode, edge->input(), edge->output(), edge->cost()));
    node->_addEdgeForce(edgePtr);
  }

  // now add arcs to all nodes in the epsilon closure
  unsigned epsilon = 0;
  for (_Iterator itr = _potentialMap.begin(); itr != _potentialMap.end(); itr++) {
    const _Potential& potential((*itr).second);
    const _NodePtr pnode(potential._node);

    if (pnode == node->_nodeA) continue;

    const Weight& d(potential._d);
    unsigned s(potential._s);
    NodePtr nextNode(find(pnode, create));
    EdgePtr edgePtr(new Edge(node, nextNode, epsilon, s, d));
    node->_addEdgeForce(edgePtr);
  }

  node->_expanded = true;  node->_lastEdgesCall = _findCount;
  return node->_edges();
}

WFSTEpsilonMinusOneRemoval::NodePtr& WFSTEpsilonMinusOneRemoval::initial(int idx)
{
  if (_initial.isNull())
    /*
    if (idx >= 0)
      _initial = new Node(idx);
    else
    */
      _initial = new Node(_A->initial());	// normal initial state

  return Cast<NodePtr>(_initial);
}

void WFSTEpsilonMinusOneRemoval::_findAllEpsilonOutgoing()
{
  unsigned epsilon = 0;
  for (_NodeVectorIterator nitr = _allNodes().begin(); nitr != _allNodes().end(); nitr++) {
    NodePtr node(*nitr);

    if (node.isNull()) continue;

    bool allEpsilon = true;
    for (Node::Iterator eitr(this, node); eitr.more(); eitr++) {
      WFSTransducer::EdgePtr& edge(eitr.edge());
      if (edge->input() != epsilon) { allEpsilon = false;  break; }
    }
    if (allEpsilon)
      _purgeList.insert(_PurgeListValueType(node->index(), node));
  }
}

void WFSTEpsilonMinusOneRemoval::_checkArcs(NodePtr& node)
{
  unsigned epsilon = 0;
  for (WFSTransducer::Node::Iterator eitr(node); eitr.more(); eitr++) {
    WFSTransducer::EdgePtr& edge(eitr.edge());
    if (edge->input() != epsilon) {
      _PurgeListIterator itr = _purgeList.find(edge->next()->index());
      if (itr != _purgeList.end())
	_purgeList.erase(itr, ++itr);
    }
  }
}

void WFSTEpsilonMinusOneRemoval::_removeLinks(NodePtr& node)
{
  WFSAcceptor::EdgePtr edge = node->_edgeList;
  WFSAcceptor::EdgePtr last = NULL;
  while (edge.isNull() == false) {
    if (_purgeList.find(edge->next()->index()) != _purgeList.end()) {
      if (edge == node->_edgeList) {
	node->_edgeList = edge->_edgeList;  last = NULL;  edge = node->_edgeList;
      } else {
	last->_edgeList = edge->_edgeList;  edge = last->_edgeList;
      }
    } else {
      last = edge;  edge = last->_edgeList;
    }
  }
}

void WFSTEpsilonMinusOneRemoval::_purge()
{
  // remove all links into nodes marked for purging
  _removeLinks(initial());
  for (_NodeVectorIterator nitr = _allNodes().begin(); nitr != _allNodes().end(); nitr++) {
    NodePtr node(*nitr);
    
    if (node.isNull()) continue;

    _removeLinks(node);
  }
  for (_NodeMapIterator nitr = _finis().begin(); nitr != _finis().end(); nitr++) {
    NodePtr& node((*nitr).second);
    _removeLinks(node);
  }

  // remove the actual nodes
  _NodeVectorIterator nitr(_allNodes().begin());
  NodePtr node(*nitr);
  printf("Purging %d nodes.\n", _purgeList.size());
  for (_PurgeListIterator itr = _purgeList.begin(); itr != _purgeList.end(); itr++) {
    unsigned index((*itr).first);
    while (node.isNull() || node->index() < index) {
      nitr++;  node = *nitr;
    }

    if (index == node->index()) *nitr = NULL;
  }
}

void WFSTEpsilonMinusOneRemoval::_findAllEpsilonIncoming()
{
  // check the initial node
  _checkArcs(initial());

  // check all intermediate nodes
  for (_NodeVectorIterator nitr = _allNodes().begin(); nitr != _allNodes().end(); nitr++) {
    NodePtr node(*nitr);

    if (node.isNull()) continue;

    _checkArcs(node);
  }

  // check all final nodes
  for (_NodeMapIterator nitr = _finis().begin(); nitr != _finis().end(); nitr++) {
    NodePtr& node((*nitr).second);
    _checkArcs(node);
  }
}

void WFSTEpsilonMinusOneRemoval::purgeAllEpsilon()
{
  _clearList();
  _findAllEpsilonOutgoing();
  _findAllEpsilonIncoming();
  _purge();
}

WFSTEpsilonMinusOneRemovalPtr removeEpsilonMinusOne(WFSTransducerPtr& wfst)
{
  WFSTEpsilonMinusOneRemovalPtr remove(new WFSTEpsilonMinusOneRemoval(wfst));
  breadthFirstSearch(remove);
  remove->purgeAllEpsilon();

  return remove;
}


// ----- methods for class `WFSTEpsilonMinusOneRemoval::Edge' -----
//
MemoryManager<WFSTEpsilonMinusOneRemoval::Edge>&  WFSTEpsilonMinusOneRemoval::Edge::memoryManager()
{
  static MemoryManager<Edge> _MemoryManager("WFSTEpsilonMinusOneRemoval::Edge");
  return _MemoryManager;
}


// ----- methods for class `WFSTEpsilonMinusOneRemoval::Node' -----
//
MemoryManager<WFSTEpsilonMinusOneRemoval::Node>& WFSTEpsilonMinusOneRemoval::Node::memoryManager()
{
  static MemoryManager<Node> _MemoryManager("WFSTEpsilonMinusOneRemoval::Node");
  return _MemoryManager;
}


// ----- methods for class `WFSTProjection' -----
//
WFSTProjection::WFSTProjection(WFSTransducerPtr& wfst, Side side)
  : WFSTSortedInput(wfst), _side(side)
{
  if (_side == Input)
    _outputLexicon = _inputLexicon;
  else
    _inputLexicon  = _outputLexicon;
}

const WFSTProjection::EdgePtr& WFSTProjection::edges(WFSAcceptor::NodePtr& nd)
{
  NodePtr node(Cast<NodePtr>(nd));

  // are edges already expanded?
  if (node->_expanded)
    return node->_edges();

  bool create = true;
  for (WFSTransducer::Node::Iterator itr(node->_nodeA); itr.more(); itr++) {
    WFSTransducer::EdgePtr& edge(itr.edge());

    unsigned symbol = (_side == Input) ? edge->input() : edge->output();
    NodePtr nextNode(find(edge->next(), create));
    EdgePtr edgePtr(new Edge(node, nextNode, symbol, symbol, edge->cost()));
    node->_addEdgeForce(edgePtr);
  }

  return node->_edges();
}

WFSTProjection::NodePtr& WFSTProjection::initial(int idx)
{
  if (_initial.isNull())
    /*
    if (idx >= 0)
      _initial = new Node(idx);
    else
    */
      _initial = new Node(_A->initial());	// normal initial state

  return Cast<NodePtr>(_initial);
}

WFSAcceptor::Node* WFSTProjection::_newNode(const WFSTransducer::NodePtr& node, Color col, Weight cost)
{
  return new Node(node, col, cost);
}

WFSAcceptor::Edge* WFSTProjection::
_newEdge(NodePtr& from, NodePtr& to, unsigned input, unsigned output, Weight cost)
{
  return new Edge(from, to, input, output, cost);
}

WFSTProjectionPtr project(WFSTransducerPtr& wfst, const String& side)
{
  WFSTProjection::Side pside;
  if (side == "Output" || side == "output" || side == "OUTPUT")
      pside = WFSTProjection::Output;
  else if (side == "Input" || side == "input" || side == "INPUT")
      pside = WFSTProjection::Input;
  else
      throw jconsistency_error("Cannot determine projection side.");
	
  WFSTProjectionPtr projector(new WFSTProjection(wfst, pside));
  breadthFirstSearch(projector);

  return projector;
}


// ----- methods for class `WFSTProjection::Edge' -----
//
MemoryManager<WFSTProjection::Edge>&  WFSTProjection::Edge::memoryManager()
{
  static MemoryManager<Edge> _MemoryManager("WFSTProjection::Edge");
  return _MemoryManager;
}


// ----- methods for class `WFSTProjection::Node' -----
//
MemoryManager<WFSTProjection::Node>& WFSTProjection::Node::memoryManager()
{
  static MemoryManager<Node> _MemoryManager("WFSTProjection::Node");
  return _MemoryManager;
}


// ----- methods for class `WFSTRemoveEndMarkers' -----
//
WFSTRemoveEndMarkers::WFSTRemoveEndMarkers(WFSTSortedInputPtr& A, const String& end, bool dynamic, bool input, const String& name)
  : WFSTSortedInput(A->stateLexicon(), A->inputLexicon(), A->outputLexicon(), dynamic, name), _end(end), _A(A), _input(input) { }

WFSTRemoveEndMarkers::NodePtr& WFSTRemoveEndMarkers::initial(int idx)
{
  if (_initial.isNull()) _initial = new Node(_A->initial());

  return Cast<NodePtr>(_initial);
}

WFSAcceptor::Node* WFSTRemoveEndMarkers::_newNode(unsigned state)
{
  return new Node(state);
}

WFSAcceptor::Edge* WFSTRemoveEndMarkers::
_newEdge(NodePtr& from, NodePtr& to, unsigned input, unsigned output, Weight cost)
{
  return new Edge(from, to, input, output, cost);
}

WFSTRemoveEndMarkers::NodePtr
WFSTRemoveEndMarkers::find(const WFSTSortedInput::NodePtr& nd, bool create)
{
  unsigned state = nd->index();

  if (initial()->index() == state)
    return initial();

  _NodeMapIterator itr = _final.find(state);

  if (itr != _final.end()) {
    assert((*itr).second->isFinal() == true);
    return Cast<NodePtr>((*itr).second);
  }

  if (state < _nodes.size() && _nodes[state].isNull() == false)
    return Cast<NodePtr>(_nodes[state]);

  if (create == false)
    throw jkey_error("No state %u exists.", state);

  NodePtr newNode(new Node(nd));
  if (nd->isFinal()) {
    newNode->_setCost(nd->cost());
    _final.insert(WFSAcceptor::_ValueType(state, newNode));
    itr = _final.find(state);
    return Cast<NodePtr>((*itr).second);
  } else {
    if (state >= _nodes.size()) _resize(state);
    _nodes[state] = newNode;
    return Cast<NodePtr>(_nodes[state]);
  }
}

const WFSTRemoveEndMarkers::EdgePtr& WFSTRemoveEndMarkers::edges(WFSAcceptor::NodePtr& nd)
{
  bool create = true;

  NodePtr node(Cast<NodePtr>(nd));

  /*
  printf("\nNode %d\n", node->index());
  if (node->_edges().isNull())
    printf("node->_edges() is Null\n");
  if (node->_nodeA.isNull())
    printf("node->_nodeA is Null\n");
  fflush(stdout);
  */

  // are edges already expanded?
  if (node->_edges().isNull() == false || node->_nodeA.isNull())
    return node->_edges();

  for (WFSTSortedInput::Node::Iterator itr(_A, Cast<WFSTSortedInput::NodePtr>(node->_nodeA)); itr.more(); itr++) {
    WFSTSortedInput::EdgePtr& edge(itr.edge());
    unsigned input  = edge->input();
    unsigned output = edge->output();
    Weight   c      = edge->cost();

    NodePtr nextNode(find(edge->next(), create));

    if (_input == true) {
      if (inputLexicon()->symbol(input).substr(0, 1) == _end) {
	// printf("Replacing %s with eps.", inputLexicon()->symbol(input).c_str());  fflush(stdout);
	input = 0;
      }
    } else {
      if (outputLexicon()->symbol(output).substr(0, 1) == _end) {
	// printf("Replacing %s with eps.", inputLexicon()->symbol(input).c_str());  fflush(stdout);
	output = 0;
      }
    }

    EdgePtr newEdge(new Edge(node, nextNode, input, output, c));
    node->_addEdgeForce(newEdge);
  }

  return node->_edges();
}


// ----- methods for class `WFSTRemoveEndMarkers::Edge' -----
//
MemoryManager<WFSTRemoveEndMarkers::Edge>&  WFSTRemoveEndMarkers::Edge::memoryManager()
{
  static MemoryManager<Edge> _MemoryManager("WFSTRemoveEndMarkers::Edge");
  return _MemoryManager;
}


// ----- methods for class `WFSTRemoveEndMarkers::Node' -----
//
WFSTRemoveEndMarkers::Node::Node(unsigned idx, Color col, Weight cost)
  : WFSTSortedInput::Node(idx, col, cost) { }

WFSTRemoveEndMarkers::Node::Node(const WFSTSortedInput::NodePtr& nodeA, Color col, Weight cost)
  : WFSTSortedInput::Node(nodeA, col, cost)
{
  /*
  printf("Allocating node %d\n", nodeA->index());
  fflush(stdout);
  */
}

MemoryManager<WFSTRemoveEndMarkers::Node>&  WFSTRemoveEndMarkers::Node::memoryManager()
{
  static MemoryManager<Node> _MemoryManager("WFSTRemoveEndMarkers::Node");
  return _MemoryManager;
}

WFSTSortedInputPtr removeEndMarkers(WFSTSortedInputPtr& A, bool input)
{
  WFSTRemoveEndMarkersPtr remove(new WFSTRemoveEndMarkers(A, "#", false, input));

  breadthFirstSearch(remove);
  return remove;
}


// ----- methods for class `WFSTSortedOutput' -----
//
WFSTSortedOutput::WFSTSortedOutput(const WFSAcceptorPtr& wfsa, bool convertWFSA)
  : WFSTransducer(wfsa, /* convertWFSA= */ false, wfsa->name()+String(" Sorted Output"))
{
  if (convertWFSA) _convert(wfsa);
}

WFSTSortedOutput::WFSTSortedOutput(const WFSTransducerPtr& A)
  : WFSTransducer(A->stateLexicon(), A->inputLexicon(), A->outputLexicon(), A->name()+String(" Sorted Output")), _A(A) { }

WFSTSortedOutput::WFSTSortedOutput(LexiconPtr& statelex, LexiconPtr& inlex, LexiconPtr& outlex, const String& name)
  : WFSTransducer(statelex, inlex, outlex, name), _A(NULL) { }

WFSTSortedOutput::NodePtr& WFSTSortedOutput::initial(int idx)
{
  if (_initial.isNull())
    if (idx >= 0)
      _initial = new Node(idx);
    else if (_A.isNull())
      _initial = new Node(WFSTSortedInput::Node::_MaximumIndex-3);	// "super" initial state for minimization
    else
      _initial = new Node(_A->initial());				// normal initial state

  return Cast<NodePtr>(_initial);
}

const WFSTSortedOutput::EdgePtr& WFSTSortedOutput::edges(WFSAcceptor::NodePtr& nd)
{
  bool create = true;

  NodePtr node(find(nd->index()));

  // are edges already expanded?
  if (node->_edges().isNull() == false || node->_nodeA.isNull())
    return node->_edges();

  for (WFSTransducer::Node::Iterator itr(_A, node->_nodeA); itr.more(); itr++) {
    WFSTransducer::EdgePtr& edge(itr.edge());
    unsigned input  = edge->input();
    unsigned output = edge->output();
    Weight   c      = edge->cost();
    NodePtr  nextNode(find(edge->next()->index(), create));

    EdgePtr newEdge(new Edge(node, nextNode, input, output, c));
    if (edge->next()->isFinal())
      nextNode->_setCost(edge->next()->cost());

    node->_addEdgeForce(edge);
  }

  return node->_edges();
}

const WFSTSortedOutput::EdgePtr& WFSTSortedOutput::edges(WFSAcceptor::NodePtr& nd, WFSTSortedInput::NodePtr& comp)
{
  return edges(nd);
}

// insert edge based on output symbol, if not present
void WFSTSortedOutput::Node::_addEdge(WFSAcceptor::EdgePtr& newEdge)
{
  EdgePtr ptr    = Cast<const EdgePtr>(_edgeList);
  EdgePtr oldPtr = ptr;
  while (ptr.isNull() == false && ptr->output() < newEdge->output()) {
    oldPtr = ptr;
    ptr    = Cast<const EdgePtr>(ptr->_edges());
  }
  while (ptr.isNull() == false && ptr->output() == newEdge->output() && ptr->input() < newEdge->input()) {
    oldPtr = ptr;
    ptr    = Cast<const EdgePtr>(ptr->_edges());
  }

  if (ptr.isNull() == false && ptr->output() == newEdge->output()) return;

  if (ptr == oldPtr) {

    newEdge->_edges() = _edgeList;
    _edgeList  = newEdge;

  } else {

    newEdge->_edges() = ptr;
    oldPtr->_edges()  = Cast<EdgePtr>(newEdge);

  }
}

// insert edge based on output symbol
void WFSTSortedOutput::Node::_addEdgeForce(WFSAcceptor::EdgePtr& newEdge)
{
  EdgePtr ptr    = Cast<const EdgePtr>(_edges());
  EdgePtr oldPtr = ptr;
  while (ptr.isNull() == false && ptr->output() < newEdge->output()) {
    oldPtr = ptr;
    ptr    = Cast<const EdgePtr>(ptr->_edges());
  }
  while (ptr.isNull() == false && ptr->output() == newEdge->output() && ptr->input() < newEdge->input()) {
    oldPtr = ptr;
    ptr    = Cast<const EdgePtr>(ptr->_edges());
  }

  if (ptr == oldPtr) {

    newEdge->_edges() = _edgeList;
    _edgeList  = newEdge;

  } else {

    newEdge->_edges() = ptr;
    oldPtr->_edges()  = Cast<EdgePtr>(newEdge);

  }
}

WFSTSortedOutput::NodePtr WFSTSortedOutput::
find(const WFSTransducer::NodePtr& node, bool create)
{
  unsigned index = node->index();

  _NodeMapIterator itr = _final.find(index);

  if (itr != _final.end()) {
    assert((*itr).second->isFinal() == true);
    return Cast<NodePtr>((*itr).second);
  }

  if (index < _nodes.size() && _nodes[index].isNull() == false)
    return Cast<NodePtr>(_nodes[index]);

  if (create == false)
    throw jkey_error("Could not find sorted output node %d.", index);

  NodePtr newNode(new Node(node, node->cost()));
  if (node->isFinal()) {
    _final.insert(WFSAcceptor::_ValueType(index, newNode));
    itr = _final.find(index);
    return Cast<NodePtr>((*itr).second);
  } else {
    if (index >= _nodes.size()) _resize(index);
    _nodes[index] = newNode;
    return Cast<NodePtr>(_nodes[index]);
  }
}

WFSAcceptor::Node* WFSTSortedOutput::_newNode(unsigned state)
{
  return new Node(state);
}

WFSAcceptor::Edge* WFSTSortedOutput::
_newEdge(NodePtr& from, NodePtr& to, unsigned input, unsigned output, Weight cost)
{
  return new Edge(from, to, input, output, cost);
}


// ----- methods for class `WFSTSortedOutput::Edge' -----
//
MemoryManager<WFSTSortedOutput::Edge>&  WFSTSortedOutput::Edge::memoryManager()
{
  static MemoryManager<Edge> _MemoryManager("WFSTSortedOutput::Edge");
  return _MemoryManager;
}


// ----- methods for class `WFSTSortedOutput::Node' -----
//
MemoryManager<WFSTSortedOutput::Node>& WFSTSortedOutput::Node::memoryManager()
{
  static MemoryManager<Node> _MemoryManager("WFSTSortedOutput::Node");
  return _MemoryManager;
}


// ----- methods for class `WFSTComposition' -----
//
WFSTComposition::
WFSTComposition(WFSTSortedOutputPtr& A, WFSTSortedInputPtr& B, LexiconPtr& stateLex,
		bool dynamic, const String& name)
  : WFSTSortedInput(stateLex, A->inputLexicon(), B->outputLexicon(), dynamic, name), _A(A), _B(B) /*,
												    _sequentialNodes(50000000) */
{
  /*
  for (_SequentialNodesIterator itr = _sequentialNodes.begin(); itr != _sequentialNodes.end(); itr++)
    (*itr) = NULL;
  */
}

WFSTComposition::NodePtr& WFSTComposition::initial(int idx)
{
  if (_initial.isNull()) {
    unsigned initialA = _A->initial()->index();
    WFSTSortedInput::NodePtr nodeB(_B->initial());
    unsigned initialB = nodeB->index();

    unsigned newIndex = _findIndex(_A->initial()->index(), _B->initial()->index(), /* filter= */ 0);
    _initial = new Node(/* filter= */ 0, _A->initial(), _B->initial(), newIndex);
  }

  return Cast<NodePtr>(_initial);
}

void WFSTComposition::reportMemoryUsage()
{
  WFSTComposition::Edge::report();
  WFSTComposition::Node::report();
}

void WFSTComposition_reportMemoryUsage() { WFSTComposition::reportMemoryUsage(); }

unsigned WFSTComposition::_findIndex(unsigned stateA, unsigned stateB, unsigned short filter)
{
  _State state(stateA, filter, stateB);
  return _indexMap.insert(state);
}

void barf()
{
  printf("This makes me want to barf!\n");  fflush(stdout);
}

void WFSTComposition::purgeUnique(unsigned count)
{
  if (_dynamic) _unique(count);

  _A->purgeUnique(count);  _B->purgeUnique(count);
}

WFSAcceptor::Node* WFSTComposition::_newNode(unsigned state)
{
  throw jconsistency_error("'WFSTComposition' should not be loaded statically.");

  return NULL;
}

WFSAcceptor::Edge* WFSTComposition::
_newEdge(NodePtr& from, NodePtr& to, unsigned input, unsigned output, Weight cost)
{
  throw jconsistency_error("'WFSTComposition' should not be loaded statically.");

  return NULL;
}


// ----- methods for class `WFSTComposition::Edge' -----
//
WFSTComposition::Edge::Edge(NodePtr& prev, NodePtr& next,
			    unsigned input, unsigned output, Weight cost)
  : WFSTSortedInput::Edge(prev, next, input, output, cost) { }

WFSTComposition::Edge::~Edge() { }

MemoryManager<WFSTComposition::Edge>&  WFSTComposition::Edge::memoryManager()
{
  static MemoryManager<Edge> _MemoryManager("WFSTComposition::Edge");
  return _MemoryManager;
}


// ----- methods for class `WFSTComposition::Node' -----
//
WFSTComposition::Node::Node(unsigned short filter, const WFSTSortedOutput::NodePtr& nodeA, const WFSTSortedInput::NodePtr& nodeB,
			    unsigned idx, Color col, Weight cost)
  : WFSTSortedInput::Node(idx, col, cost), _filter(filter), _nodeA(nodeA), _nodeB(nodeB) { }

WFSTComposition::Node::~Node() { }

MemoryManager<WFSTComposition::Node>& WFSTComposition::Node::memoryManager()
{
  static MemoryManager<Node> _MemoryManager("WFSTComposition::Node");
  return _MemoryManager;
}

WFSTSortedInputPtr compose(WFSTSortedOutputPtr& sortedA, WFSTSortedInputPtr& sortedB,
			   const String& semiring, const String& name)
{
  LexiconPtr stateLex(new Lexicon("Composition State Lexicon"));

  WFSTComposition* comp = NULL;
  if (semiring == "Tropical")
    comp = new WFSTComp<TropicalSemiring>(sortedA, sortedB, stateLex, /* dynamic= */ false, name);
  else if (semiring == "LogProb")
    comp = new WFSTComp<LogProbSemiring>(sortedA, sortedB, stateLex, /* dynamic= */ false, name);
  else
    throw jtype_error("Could not determinine type of semiring for %s.", semiring.c_str());

  WFSTCompositionPtr composition(comp);

  breadthFirstSearch(composition);
  return purgeWFST(composition);
}


// ----- methods for class `BreadthFirstSearch' -----
//
BreadthFirstSearch::BreadthFirstSearch(unsigned purgeCount) :
  _purgeCount(purgeCount) { _clear(); }

BreadthFirstSearch::~BreadthFirstSearch() { _clear(); }

void BreadthFirstSearch::_clear() { _nodeQueue.clear(); }

void BreadthFirstSearch::search(WFSTransducerPtr& A)
{
  A->setColor(WFSAcceptor::White);
  _nodeQueue.push(A->initial());

  unsigned purge = 0;
  while (_nodeQueue.more()) {
    NodePtr node(_nodeQueue.pop());
    _expandNode(A, node);

    if (++purge % _purgeCount == 0) {
      A->purgeUnique(_purgeCount);
    }
  }
}

void BreadthFirstSearch::_expandNode(WFSTransducerPtr& A, NodePtr& node)
{
  for (Iterator itr(A, node); itr.more(); itr++) {
    NodePtr& nextNode(itr.edge()->next());
    if (nextNode->color() != WFSAcceptor::White) continue;

    nextNode->setColor(WFSAcceptor::Gray);
    _nodeQueue.push(nextNode);
  }
  node->setColor(WFSAcceptor::Black);
}

void breadthFirstSearch(WFSTransducerPtr& A, unsigned purgeCount)
{
  BreadthFirstSearch bfs(purgeCount);
  bfs.search(A);
}


// ----- methods for class `BreadthFirstWrite' -----
//
BreadthFirstWrite::BreadthFirstWrite(const String& fileName, bool useSymbols, unsigned purgeCount)
  : BreadthFirstSearch(purgeCount), _fileName(fileName), _fp(stdout), _useSymbols(useSymbols)
{
  if (fileName != "") _fp = fileOpen(fileName, "w");
}

BreadthFirstWrite::~BreadthFirstWrite()
{
  fileClose(_fileName, _fp);
}

void BreadthFirstWrite::_expandNode(WFSTransducerPtr& A, NodePtr& node)
{
  for (Iterator itr(A, node); itr.more(); itr++) {
    EdgePtr& edge(itr.edge());

    if (_useSymbols)
      edge->write(A->stateLexicon(), A->inputLexicon(), A->outputLexicon(), _fp);
    else
      edge->write(_fp);

    NodePtr& nextNode(edge->next());
    if (nextNode->color() != WFSAcceptor::White) continue;
    nextNode->setColor(WFSAcceptor::Gray);
    _nodeQueue.push(nextNode);
  }
  node->setColor(WFSAcceptor::Black);

  if (node->isFinal()) {
    if (_useSymbols)
      node->write(A->stateLexicon(), _fp);
    else
      node->write(_fp);
  }
}

void breadthFirstWrite(WFSTransducerPtr& A, const String& fileName, bool useSymbols, unsigned purgeCount)
{
  BreadthFirstWrite bfw(fileName, useSymbols, purgeCount);
  bfw.write(A);
}


// ----- methods for class `DepthFirstSearch' -----
//
void DepthFirstSearch::search(WFSTransducerPtr& A)
{
  _depth = 0;
  A->setColor(WFSAcceptor::White);
  _expandNode(A, A->initial());
}

void DepthFirstSearch::_expandNode(WFSTransducerPtr& A, NodePtr& node)
{
  _depth++;
  node->setColor(WFSAcceptor::Gray);
  for (Iterator itr(A, node); itr.more(); itr++) {
    NodePtr& nextNode(itr.edge()->next());

    if (nextNode->color() != WFSAcceptor::White) continue;

    _expandNode(A, nextNode);
  }
  node->setColor(WFSAcceptor::Black);
  _depth--;
}

void depthFirstSearch(WFSTransducerPtr& A)
{
  DepthFirstSearch dfs;
  dfs.search(A);
}


// ----- methods for helper class `WFSTDeterminization::StateEntry' -----
//
WFSTDeterminization::StateEntry::StateEntry()
  : _state(NULL), _residualWeight(ZeroWeight), _residualStringSize(0)
{
  for (unsigned i = 0; i < MaxResidualSymbols; i++)
    _residualString[i] = 0;
}

WFSTDeterminization::StateEntry::StateEntry(const vector<Index>& residualString)
  : _state(NULL), _residualWeight(ZeroWeight), _residualStringSize(residualString.size())
{
  if (residualString.size() > MaxResidualSymbols)
    throw jdimension_error("Length of residual string (%d) is greater than maximum allowable (%d).",
			   residualString.size(), MaxResidualSymbols);

  for (unsigned i = 0; i < MaxResidualSymbols; i++)
    _residualString[i] = 0;

  unsigned i = 0;
  for (vector<Index>::const_iterator itr = residualString.begin(); itr != residualString.end(); itr++)
    _residualString[i++] = *itr;
}

WFSTDeterminization::StateEntry::
StateEntry(const WFSTSortedInput::NodePtr& node, Weight residualWeight)
  : _state(node), _residualWeight(residualWeight), _residualStringSize(0)
{
  for (unsigned i = 0; i < MaxResidualSymbols; i++)
    _residualString[i] = 0;
}

WFSTDeterminization::StateEntry::
StateEntry(const WFSTSortedInput::NodePtr& node, Weight residualWeight, const vector<Index>& residualString)
  : _state(node), _residualWeight(residualWeight), _residualStringSize(residualString.size())
{
  if (residualString.size() > MaxResidualSymbols)
    throw jdimension_error("Length of residual string (%d) is greater than maximum allowable (%d).",
			   residualString.size(), MaxResidualSymbols);

  for (unsigned i = 0; i < MaxResidualSymbols; i++)
    _residualString[i] = 0;

  unsigned i = 0;
  for (vector<Index>::const_iterator itr = residualString.begin(); itr != residualString.end(); itr++)
    _residualString[i++] = *itr;
}

const String& WFSTDeterminization::StateEntry::name(const LexiconPtr& outlex) const
{
  static String nm;
  static char buffer[MaxCharacters];
  static char temp[MaxCharacters];

  // cout << "Examining " << _state.the_p << endl;

  sprintf(buffer, "%d:", (_state.isNull() ? -1 : int(_state->index())));
  if (_residualStringSize > 0)
    sprintf(temp,   "%5.8e:", float(_residualWeight));
  else
    sprintf(temp,   "%5.8e", float(_residualWeight));
  strcat(buffer, temp);

  for (unsigned i = 0; i < _residualStringSize; i++) {
    const String& sym = outlex->symbol(_residualString[i]);
    if (i == _residualStringSize - 1)
      sprintf(temp, "%s", sym.c_str());
    else
      sprintf(temp, "%s;", sym.c_str());
    strcat(buffer, temp);
  }

  nm = buffer;

  return nm;
}

WFSTDeterminization::StateEntry::Index WFSTDeterminization::StateEntry::residualString(unsigned i) const
{
  if (i >= _residualStringSize)
    throw jindex_error("Index 'i' %d >= '_residualStringSize'.", i, _residualStringSize);
  return _residualString[i];
}

void WFSTDeterminization::StateEntry::write(const LexiconPtr& outlex, FILE* fp) const
{
  printf("State Entry : %s\n", name(outlex).c_str());
}


// ----- methods for helper class `WFSTDeterminization::Substates' -----
//
WFSTDeterminization::Substates::Cursor
WFSTDeterminization::Substates::size(u8 partition) const
{
  if (partition == 0xff) return 0;
  return (partition >> 6) + 1 + (partition & 0x10 ? sizeof(Weight) : 0) + (partition & 0x0f);
}

u32 WFSTDeterminization::Substates::hash(Cursor pos) const
{
  u32 value = 0;
  for (Vector<u8>::const_iterator i = _substates.begin() + pos + 2 * sizeof(Cursor); *i != 0xff;) {
    Vector<u8>::const_iterator end = i + 1 + size(*i);
    for (; i != end; ++i) value = 337 * value + *i;
  }
  return value;
}

bool WFSTDeterminization::Substates::equal(Cursor pos1, Cursor pos2) const
{
  Vector<u8>::const_iterator i1 = _substates.begin() + pos1 + 2 * sizeof(Cursor);
  Vector<u8>::const_iterator i2 = _substates.begin() + pos2 + 2 * sizeof(Cursor);
  for (; *i1 != 0xff;) {
    u8 partition = *(i1++);
    if (partition != *(i2)) return false;
    assert(*i2 != 0xff);
    ++i2;
    Cursor len = size(partition);
    if (memcmp(&*i1, &*i2, len) != 0) return false;
    i1 += len;
    i2 += len;
  }
  if (*i2 != 0xff) return false;
  return true;
}

void WFSTDeterminization::Substates::appendSubstates(const StateEntryListBase* state)
{
  appendBytes(_substates, 0, sizeof(Cursor));
  appendBytes(_substates, _substatesN, sizeof(Cursor));
  Weight previous = ZeroWeight;

  for (StateEntryListBase::Iterator itr(state); itr.more(); itr++) {
    const StateEntry& entry(itr.entry());

    unsigned stateIndex = entry.state().isNull() ? (UINT_MAX - 1) : entry.state()->index();
    u8 partition = (estimateBytes(stateIndex) - 1) << 6;

    Weight hashWeight(entry.residualWeight().hash());
    if (previous == hashWeight) partition |= 0x20;
    else if (hashWeight != ZeroWeight) partition |= 0x10;

    if (entry.residualStringSize() > MaxResidualSymbols)
      throw jconsistency_error("Residual string size (%d) not supported.", entry.residualStringSize());

    /*
    if (entry.residualStringSize() == 2) {
      printf("String size = 2\n");  fflush(stdout);
    }
    */

    u32 stringBytes = 0;
    if (entry.residualStringSize() > 0)
      stringBytes = u8(estimateBytes(entry.residualString(0)));
    if (entry.residualStringSize() == 2)
      stringBytes += 4;
    partition |= stringBytes;

    _substates.push_back(partition);
    appendBytes(_substates, stateIndex, (partition >> 6) + 1);
    if (partition & 0x10) appendBytes(_substates, hashWeight, sizeof(Weight));

    if (entry.residualStringSize() > 0) {
      if (stringBytes > 4) stringBytes -= 4;
      appendBytes(_substates, entry.residualString(0), stringBytes);
    }
    if (entry.residualStringSize() == 2)
      appendBytes(_substates, entry.residualString(1), 4);

    previous = hashWeight;
  }
  _substates.push_back(0xff);
}

WFSTDeterminization::Substates::Cursor
WFSTDeterminization::Substates::append(const StateEntryListBase* state, bool create)
{
  // append substates
  Cursor start = _substates.size();
  appendSubstates(state);

  // get hash value
  u32 key = hash(start);
  Cursor i = _bins[key % _bins.size()];
  for (; (i != UINT_MAX) && (!equal(start, i));
       i = Cursor(getBytes(_substates.begin() + i, sizeof(Cursor))));

  // if existing: resize vector, else: add to hash table
  if (i != UINT_MAX) {
    /*
      printf("Found previous %d.\n", u32(getBytes(_substates.begin() + i + sizeof(Cursor), sizeof(Cursor))));  fflush(stdout);
    */
    _substates.resize(start);
    return u32(getBytes(_substates.begin() + i + sizeof(Cursor), sizeof(Cursor)));
  }

  if (create == false)
    throw jkey_error("Could not find substate.");

  // resize hash table on demand
  if (_substatesN > 2 * _bins.size()) {
    /*
    printf("Rehashing to size %d\n", 2 * _bins.size() - 1);
    printf("Substates size %d\n", _substates.size());  fflush(stdout);
    */

    std::fill(_bins.begin(), _bins.end(), UINT_MAX);
    _bins.grow(2 * _bins.size() - 1, UINT_MAX);

    for (Cursor i = 0; i != start; ++i) {
      u32 key = hash(i) % _bins.size();
      setBytes(_substates.begin() + i, _bins[key], sizeof(Cursor));
      _bins[key] = i;
      for (i += 2 * sizeof(Cursor); _substates[i] != 0xff; i += 1 + size(_substates[i]));
    }
  }
  setBytes(_substates.begin() + start, _bins[key % _bins.size()], sizeof(Cursor));
  _bins[key % _bins.size()] = start;
  ++_substatesN;
  return (_substatesN - 1);
}


// ----- methods for helper class `WFSTDeterminization::StateEntryListBase' -----
//
WFSTDeterminization::StateEntryListBase::~StateEntryListBase() { }

const String& WFSTDeterminization::StateEntryListBase::name(const LexiconPtr& outlex) const
{
  static String nm;

  nm = "";
  int idx = -1;
  bool firstEntry = true;
  for (_StateEntryListConstIterator itr = _stateEntryList.begin(); itr != _stateEntryList.end(); itr++) {
    if (idx > (*itr).index()) {
      (*itr).write(outlex);
      throw jconsistency_error("Indices (%d, %d) do not increase.", idx, (*itr).index());
    }
    idx = (*itr).index();
    if (firstEntry == true)
      firstEntry = false;
    else
      nm += "|";
    nm += (*itr).name(outlex);
  }

  return nm;
}

void WFSTDeterminization::StateEntryListBase::write(const LexiconPtr& outlex, FILE* fp) const
{
  for (_StateEntryListConstIterator itr = _stateEntryList.begin(); itr != _stateEntryList.end(); itr++)
    (*itr).write(outlex, fp);
}

bool WFSTDeterminization::StateEntryListBase::isFinal() const
{
  for (_StateEntryListConstIterator itr = _stateEntryList.begin(); itr != _stateEntryList.end(); itr++)
    if ((*itr).isFinal()) return true;

  bool allNull = true;
  for (_StateEntryListConstIterator itr = _stateEntryList.begin(); itr != _stateEntryList.end(); itr++)
    if ((*itr).state().isNull() == false) { allNull = false;  break; }

  return allNull;
}

unsigned WFSTDeterminization::StateEntryListBase::residualStringSize(vector<unsigned>* rstr) const
{
  static unsigned residualString[MaxResidualSymbols];
  for (unsigned i = 0; i < MaxResidualSymbols; i++)
    residualString[i] = UINT_MAX;

  unsigned sz       = MaxResidualSymbols;
  bool     foundEnd = false;
  for (_StateEntryListConstIterator itr = _stateEntryList.begin(); itr != _stateEntryList.end(); itr++) {
    const StateEntry& entry(*itr);
    if (entry.isFinal() == false) continue;

    foundEnd = true;
    sz = min(sz, entry.residualStringSize());

    for (unsigned i = 0; i < sz; i++) {
      if (residualString[i] == UINT_MAX) {
	residualString[i] = entry.residualString(i);
      } else if (residualString[i] != entry.residualString(i)) {
	sz = i;  break;
      }
    }

    if (sz == 0) break;
  }

  if (foundEnd == false)
    throw jconsistency_error("Could not find end node.");

  if (rstr != NULL)
    for (unsigned i = 0; i < MaxResidualSymbols; i++)
      (*rstr)[i] = (i < sz) ? residualString[i] : 0;

  return sz;
}

vector<unsigned> WFSTDeterminization::StateEntryListBase::residualString() const
{
  vector<unsigned> residual(MaxResidualSymbols);
  residualStringSize(&residual);
  
  unsigned residualSize = MaxResidualSymbols;
  for (unsigned i = 0; i < MaxResidualSymbols; i++)
    if (residual[i] == 0) { residualSize = i; break; }
      
  residual.resize(residualSize);
  return residual;
}

MemoryManager<WFSTDeterminization::StateEntryListBase>& WFSTDeterminization::StateEntryListBase::memoryManager()
{
  static MemoryManager<StateEntryListBase> _MemoryManager("WFSTDeterminization::StateEntryListBase");
  return _MemoryManager;
}


// ----- methods for helper class `WFSTDeterminization::ArcEntry' -----
//
unsigned WFSTDeterminization::ArcEntry::firstOutputSymbol() const
{
  if (_state.residualStringSize() > 0)
    return _state.residualString(0);

  return _edge->output();
}


// ----- methods for helper class `WFSTDeterminization::ArcEntryList' -----
//
WFSTDeterminization::ArcEntryList::~ArcEntryList()
{
  for (_ArcEntryListConstIterator itr = _arcEntryList.begin(); itr != _arcEntryList.end(); itr++)
    delete *itr;
}


// ----- methods for helper class `WFSTDeterminization::ArcEntryMap' -----
//
WFSTDeterminization::ArcEntryMap::~ArcEntryMap()
{
  for (_ArcEntryMapConstIterator itr = _arcEntryMap.begin(); itr != _arcEntryMap.end(); itr++)
    delete *itr;
}

void WFSTDeterminization::ArcEntryMap::add(const ArcEntry* arcEntry)
{
  ArcEntryList* l = new ArcEntryList(arcEntry->symbol());
  _HashPair pair(_arcEntryMap.insertExisting(l));
  
  ArcEntryList* arcList = _arcEntryMap[pair.first];
  arcList->add(arcEntry);

  if (l != arcList) delete l;
}


// ----- methods for class `WFSTDeterminization' -----
//
const unsigned WFSTDeterminization::MaxResidualSymbols;

WFSTDeterminization::WFSTDeterminization(WFSTSortedInputPtr& A, LexiconPtr& stateLex, bool dynamic)
  : WFSTSortedInput(stateLex, A->inputLexicon(), A->outputLexicon(), dynamic, String("det(")+A->name()+String(")")), _A(A)
{ }

WFSTDeterminization::NodePtr& WFSTDeterminization::initial(int idx)
{
  if (_initial.isNull()) {
    bool create = true;
    const StateEntryListBase* stateEntryList = _initialStateEntryList(_A);
    _initial = new Node(stateEntryList, _substates.append(stateEntryList, create));
  }

  return Cast<NodePtr>(_initial);
}

const WFSTDeterminization::EdgePtr& WFSTDeterminization::edges(WFSAcceptor::NodePtr& nd)
{
  NodePtr node(Cast<NodePtr>(nd));

  // are edges already expanded?
  if (node->_expanded)
    return node->_edges();

  ArcEntryMap* arcEntryMap(node->arcEntryMap(_A));

  for (ArcEntryMap::Iterator itr(arcEntryMap); itr.more(); itr++)
    _addArc(node, itr.list());

  if (node->_stateEntryList->isFinal() && node->_stateEntryList->residualStringSize() > 0)
    _addArcToEnd(node);

  static EdgePtr ptr;
  ptr = node->_edges();

  node->_expanded = true;  node->_lastEdgesCall = _findCount;
  if (_dynamic == false) {
    delete node->_stateEntryList;  node->_stateEntryList = NULL;
  }

  delete arcEntryMap;

  return ptr;
}

void WFSTDeterminization::reportMemoryUsage()
{
  /*
  StateEntry::memoryManager().report();
  */
  StateEntryListBase::report();

  ArcEntry::report();
  ArcEntryList::report();
  ArcEntryMap::report();

  WFSTDeterminization::Edge::report();
  WFSTDeterminization::Node::report();  
}

void WFSTDeterminization_reportMemoryUsage() { WFSTDeterminization::reportMemoryUsage(); }

WFSTDeterminization::NodePtr WFSTDeterminization::find(const StateEntryListBase* state, bool create)
{
  ++_findCount;

  unsigned index = _substates.append(state, create);

  if (initial()->index() == index) { delete state; return initial(); }

  _NodeMapIterator itr = _final.find(index);

  if (itr != _final.end()) {
    delete state;
    assert((*itr).second->isFinal() == true);
    return Cast<NodePtr>((*itr).second);
  }

  if (index < _nodes.size() && _nodes[index].isNull() == false) {
    NodePtr& node(Cast<NodePtr>(_nodes[index]));
    if (node == whiteNode())
      node = new Node(state, index, White);
    else if (node == grayNode())
      node = new Node(state, index, Gray);
    else if (node == blackNode())
      node = new Node(state, index, Black);
    else
      delete state;
    return node;
  }

  if (create == false)
    throw jkey_error("Could not find state for node %d.", index);

  NodePtr newNode(new Node(state, index));

  if (newNode->color() != White)
    throw jconsistency_error("Color of newly created node should be White.");

  if (state->isFinal() && state->residualStringSize() == 0) {
    newNode->_setCost(cost(state));
    _final.insert(WFSAcceptor::_ValueType(index, newNode));
    itr = _final.find(index);
    return Cast<NodePtr>((*itr).second);
  } else {
    if (index >= _nodes.size()) _resize(index);
    _nodes[index] = newNode;
    return Cast<NodePtr>(_nodes[index]);
  }
}

unsigned WFSTDeterminization::_arcSymbol(const StateEntryListBase* selist) const
{
  int asym = -1;
  for (StateEntryListBase::Iterator itr(selist); itr.more(); itr++) {
    const StateEntry& entry(itr.entry());
    unsigned symbol = entry.residualString();
    if (asym == -1)
      asym = symbol;
    else if (asym != symbol)
      throw jconsistency_error("Entries in 'StateEntryList' have inconsistent symbols (%d vs. %d).",
			       asym, symbol);
  }
  if (asym == -1)
    throw jconsistency_error("No valid output symbol found in 'StateEntryList'.");

  return unsigned(asym);
}

unsigned WFSTDeterminization::_arcSymbol(const ArcEntryList* aelist) const
{
  int asym = -1;
  for (ArcEntryList::Iterator itr(aelist); itr.more(); itr++) {
    const ArcEntry* entry(itr.entry());
    unsigned symbol = entry->firstOutputSymbol();
    if (asym == -1)
      asym = symbol;
    else if (asym != symbol)
      asym = 0;
  }
  if (asym == -1) asym = 0;

  return unsigned(asym);
}


// ----- methods for class `WFSTDeterminization::Edge' -----
//
MemoryManager<WFSTDeterminization::Edge>&  WFSTDeterminization::Edge::memoryManager()
{
  static MemoryManager<Edge> _MemoryManager("WFSTDeterminization::Edge");
  return _MemoryManager;
}


// ----- methods for class `WFSTDeterminization::Node' -----
//
WFSTDeterminization::ArcEntryMap* WFSTDeterminization::Node::arcEntryMap(WFSTSortedInputPtr& wfst)
{
  ArcEntryMap* arcEntryMap = new ArcEntryMap();

  for (StateEntryListBase::Iterator itr(_stateEntryList); itr.more(); itr++) {
    StateEntry& stateEntry(Cast<StateEntry>(itr.entry()));
    WFSTSortedInput::NodePtr& node(stateEntry.state());
    if (node.isNull()) continue;
    for (WFSTSortedInput::Node::Iterator nitr(wfst, node); nitr.more(); nitr++) {
      ArcEntry* arcEntry = new ArcEntry(stateEntry, nitr.edge());
      arcEntryMap->add(arcEntry);
    }
  }

  return arcEntryMap;
}

MemoryManager<WFSTDeterminization::Node>& WFSTDeterminization::Node::memoryManager()
{
  static MemoryManager<Node> _MemoryManager("WFSTDeterminization::Node");
  return _MemoryManager;
}


WFSTDeterminizationPtr determinize(WFSTransducerPtr& A, const String& semiring, unsigned count)
{
  LexiconPtr stateLex(new Lexicon("Composition State Lexicon"));
  WFSTSortedInputPtr sortedA(new WFSTSortedInput(A));

  WFSTDeterminization* det = NULL;
  if (semiring == "Tropical") {
    det = new WFSTDet<TropicalSemiring>(sortedA, stateLex, /* dynamic= */ false);
    printf("Determinizing with the 'Tropical' semiring.\n");
  } else if (semiring == "LogProb") {
    det = new WFSTDet<LogProbSemiring>(sortedA, stateLex, /* dynamic= */ false);
    printf("Determinizing with the 'LogProb' semiring.\n");
  } else {
    throw jtype_error("Could not determinine type of semiring for %s.", semiring.c_str());
  }

  WFSTDeterminizationPtr determinization(det);

  breadthFirstSearch(determinization, count);

  printf("Finished determinization.\n");  fflush(stdout);

  return determinization;
}

WFSTDeterminizationPtr determinize(WFSTSortedInputPtr& sortedA, const String& semiring, unsigned count)
{
  LexiconPtr stateLex(new Lexicon("Composition State Lexicon"));

  WFSTDeterminization* det = NULL;
  if (semiring == "Tropical") {
    det = new WFSTDet<TropicalSemiring>(sortedA, stateLex);
    printf("Determinizing with the 'Tropical' semiring.\n");
  } else if (semiring == "LogProb") {
    det = new WFSTDet<LogProbSemiring>(sortedA, stateLex);
    printf("Determinizing with the 'LogProb' semiring.\n");
  } else {
    throw jtype_error("Could not determinine type of semiring for %s.", semiring.c_str());
  }

  WFSTDeterminizationPtr determinization(det);

  breadthFirstSearch(determinization, count);
  return determinization;
}

// ----- methods for class `WFSTIndexed' -----
//
WFSTIndexed::WFSTIndexed(const WFSAcceptorPtr& wfsa, bool convertWFSA)
  : WFSTSortedOutput(wfsa, /* convertFlag= */ false)
{
  if (convertWFSA) _convert(wfsa);
}

WFSTIndexed::
WFSTIndexed(LexiconPtr& statelex, LexiconPtr& inlex, const String& grammarFile, const String& name)
  : WFSTSortedOutput(statelex, inlex, inlex, name)
{
  if (grammarFile != "") read(grammarFile);
}

WFSTIndexed::
WFSTIndexed(LexiconPtr& statelex, LexiconPtr& inlex, LexiconPtr& outlex, const String& name)
  : WFSTSortedOutput(statelex, inlex, outlex, name) { }

WFSTIndexed::NodePtr& WFSTIndexed::initial(int idx)
{
  if (_initial.isNull())
    throw jconsistency_error("'WFSTIndexed' has not been loaded.");

  return Cast<NodePtr>(_initial);
}

void WFSTIndexed::_convert(const WFSAcceptorPtr& wfsa)
{
  WFSTransducer::_convert(wfsa);  _indexEdges();
}

void WFSTIndexed::read(const String& fileName, bool noSelfLoops)
{
  WFSTransducer::read(fileName, noSelfLoops);  _indexEdges();
}

const WFSTIndexed::EdgePtr& WFSTIndexed::edges(WFSAcceptor::NodePtr& nd)
{
  throw jconsistency_error("This method is not supported for 'WFSTIndexed'.");
}

const WFSTIndexed::EdgePtr& WFSTIndexed::edges(WFSAcceptor::NodePtr& nd, WFSTSortedInput::NodePtr& lat)
{
  NodePtr node(Cast<NodePtr>(nd));

  // only branch node need be expanded
  if (node->_edges().isNull() == false && node->_latticeNode == lat)
    return node->_edges();

  node->_latticeNode = lat;  node->_edges() = NULL;

  _expandNode(node, node->_latticeNode);

  static EdgePtr retList;
  retList = node->_edges();

  return retList;
}

void WFSTIndexed::_expandNode(NodePtr& node, WFSTSortedInput::NodePtr& latticeNode)
{
  set<unsigned> elist;

  // add all epsilon arcs
  unsigned output = 0;
  _EdgeMapIterator epsitr = node->_edgeMap.find(output);
  if (epsitr != node->_edgeMap.end() && elist.find(output) == elist.end()) {
    elist.insert(output);
    _EdgeList& edgesList((*epsitr).second);
    for (_EdgeListIterator eeitr = edgesList.begin(); eeitr != edgesList.end(); eeitr++) {
      EdgePtr& nextEdge(*eeitr);
      EdgePtr newEdge(new Edge(node, nextEdge->next(), nextEdge->input(), nextEdge->output(), nextEdge->cost()));
      node->_addEdgeForce(newEdge);
    }
  }

  // add arcs to match grammar arcs on the output side
  for (WFSTSortedInput::Node::Iterator itr(_lattice, latticeNode); itr.more(); itr++) {
    WFSTSortedInput::EdgePtr& latticeEdge(itr.edge());
    output = latticeEdge->input();

    if (elist.find(output) != elist.end()) continue;

    elist.insert(output);

    _EdgeMapIterator eitr = node->_edgeMap.find(output);
    if (eitr == node->_edgeMap.end()) continue;
    _EdgeList& edgesList((*eitr).second);
    for (_EdgeListIterator eeitr = edgesList.begin(); eeitr != edgesList.end(); eeitr++) {
      EdgePtr& nextEdge(*eeitr);
      EdgePtr newEdge(new Edge(node, nextEdge->next(), nextEdge->input(), nextEdge->output(), nextEdge->cost()));
      node->_addEdgeForce(newEdge);
    }
  }
}

void WFSTIndexed::_indexEdges()
{
  // index adjacency lists of all nodes
  initial()->_indexEdges();

  for (_NodeVectorIterator itr = _allNodes().begin(); itr != _allNodes().end(); itr++) {
    NodePtr& node(*itr);
    if (node.isNull()) continue;
    node->_indexEdges();
  }

  for (_NodeMapIterator itr = _finis().begin(); itr != _finis().end(); itr++) {
    NodePtr& node((*itr).second);
    node->_indexEdges();
  }
}

WFSAcceptor::Node* WFSTIndexed::_newNode(unsigned state)
{
  return new Node(state);
}

WFSAcceptor::Edge* WFSTIndexed::_newEdge(NodePtr& from, NodePtr& to, unsigned input, unsigned output, Weight cost)
{
  return new Edge(from, to, input, output, cost);
}


// ----- methods for class `WFSTIndexed::Edge' -----
//
WFSTIndexed::Edge::Edge(NodePtr& prev, NodePtr& next, unsigned input, unsigned output, Weight cost)
  : WFSTSortedOutput::Edge(prev, next, input, output, cost)
{
  _prev.disable();
  if (_next == _prev) _next.disable();
}

WFSTIndexed::Edge::~Edge()
{
  /*
  printf("Deleting WFSTIndexed::Edge from %d to %d\n",
	 _prev->index(), _next->index());  fflush(stdout);
  */
}

MemoryManager<WFSTIndexed::Edge>& WFSTIndexed::Edge::memoryManager()
{
  static MemoryManager<Edge> _MemoryManager("WFSTIndexed::Edge");
  return _MemoryManager;
}


// ----- methods for class `WFSTIndexed::Node' -----
//
void WFSTIndexed::Node::_indexEdges()
{
  EdgePtr edge(_edges());
  while (edge.isNull() == false) {
    _EdgeMapIterator eitr = _edgeMap.find(edge->output());
    if (eitr == _edgeMap.end()) {
      _EdgeList elist;
      elist.push_back(edge);
      _edgeMap.insert(_EdgeMapValueType(edge->output(), elist));
    } else {
      (*eitr).second.push_back(edge);
    }

    EdgePtr tmp(edge->_edges()); edge = tmp;
  }
}

MemoryManager<WFSTIndexed::Node>& WFSTIndexed::Node::memoryManager()
{
  static MemoryManager<Node> _MemoryManager("WFSTIndexed::Node");
  return _MemoryManager;
}


// ----- methods for class `WFSTLexicon' -----
//
WFSTLexicon::
WFSTLexicon(LexiconPtr& statelex, LexiconPtr& inlex, LexiconPtr& outlex,
	    const String& sil, const String& breath, const String& sos, const String& eos, const String& end,
	    const String& backOff, double backOffWeight,
	    unsigned maxWordN, const String& lexiconFile, bool epsilonToBranch, const String& name)
  : WFSTSortedOutput(statelex, inlex, outlex, name),
    _sil(inlex->index(sil)), _breath(_breathSymbols(breath)),
    _sosOutput(((sos == "") ? 0 : outputLexicon()->index(sos))), _eosOutput(outputLexicon()->index(eos)),
    _backOffInput(((backOff == "") ? 0 : inputLexicon()->index(backOff))),
    _backOffOutput(((backOff == "") ? 0 : outputLexicon()->index(backOff))), _backOffWeight(backOffWeight),
    _end(end), _maxWordN(maxWordN), _maxEndX(0), _nodesN(1), _epsilonToBranch(epsilonToBranch),
    _branchList(new _EdgeList[_maxWordN]), _branch(new Node(_nodesN++, /* isBranch= */ true)),
    _endButOne(new Node(_nodesN++)), _endNode(new Node(_nodesN++))
{
  _endNode->_setCost();
  EdgePtr edge(new Edge(_endButOne, _endNode, inputLexicon()->index(eos), outputLexicon()->index(eos)));
  _endButOne->_addEdgeForce(edge);

  if (lexiconFile != "")
    read(lexiconFile);
}

WFSTLexicon::~WFSTLexicon()
{
  delete[] _branchList;
}

unsigned WFSTLexicon::editDistance(unsigned word1, unsigned word2) const
{
  if (word1 == word2) return 0;

  _EdgeList& edgeList1(_branchList[word1]);
  _EdgeList& edgeList2(_branchList[word2]);

  _EdgeListIterator itr1 = edgeList1.begin();
  _EdgeListIterator itr2 = edgeList2.begin();

  EdgePtr edge1(*itr1);
  EdgePtr edge2(*itr2);

  unsigned edit    = 0;
  unsigned symbol1 = edge1->input();
  unsigned symbol2 = edge2->input();
  
  String string1   = _inputLexicon->symbol(symbol1);
  String string2   = _inputLexicon->symbol(symbol2);

  while (string1.find('#') == String::npos && string2.find('#') == String::npos); {

    if (symbol1 != symbol2) edit++;

    EdgePtr tmp1 = edge1; edge1 = tmp1->next()->_edges();
    EdgePtr tmp2 = edge2; edge2 = tmp2->next()->_edges();

    symbol1 = edge1->input();
    symbol2 = edge2->input();
  
    string1 = _inputLexicon->symbol(symbol1);
    string2 = _inputLexicon->symbol(symbol2);
  } 

  while (string1.find('#') == String::npos) {
    edit++;

    EdgePtr tmp1 = edge1; edge1 = tmp1->next()->_edges();

    symbol1 = edge1->input();
    string1 = _inputLexicon->symbol(symbol1);
  }

  while (string2.find('#') == String::npos) {
    edit++;

    EdgePtr tmp2 = edge2; edge2 = tmp2->next()->_edges();

    symbol2 = edge2->input();
    string2 = _inputLexicon->symbol(symbol2);
  }

  return edit;
}

void WFSTLexicon::_clear()
{
  _pronunciationList.clear();  _maxEndX = 0;
  for (unsigned i = 0; i < _maxWordN; i++)
    _branchList[i].clear();
  _nodesN = 4;
}

void WFSTLexicon::read(const String& fileName)
{
  _clear();

  FILE*         fp     = fileOpen(fileName, "r");
  static size_t n      = 1000;
  static char*  buffer = (char*) malloc(1000 * sizeof(char));
  unsigned      nsyms  = 0;

  while(getline(&buffer, &n, fp) > 0) {
    String line(buffer);
    String::size_type posEnd = line.find_first_of('\n');
    if (posEnd == String::npos)
      throw jconsistency_error("Expected a null-terminated string.");

    _addWord(line.substr(0, posEnd));
  }
  fileClose(fileName, fp);

  _checkMissingPronunciations();

  printf("\n>>> Maximum word boundary markers = %d <<<\n\n", _maxEndX);
}

WFSTLexicon::NodePtr& WFSTLexicon::initial(int idx)
{
  if (_grammar.isNull())
    throw jconsistency_error("Must set grammar before using 'WFSTLexicon'");

  if (_initial.isNull()) {
    NodePtr initialNode(new Node(/* idx= */ 0, /* isBranch= */ false));

    EdgePtr edge(new Edge(initialNode, _branch, _sil, _sosOutput));
    initialNode->_addEdgeForce(edge);

    _initial = initialNode;
  }

  return Cast<NodePtr>(_initial);
}

list<String> WFSTLexicon::_parsePronunciation(const String& pronunciation)
{
  list<String> phones;

  String::size_type pos     = 0;
  String::size_type prevPos = 0;
  while ((pos = pronunciation.find_first_of(' ', pos)) != String::npos) {
    String subString(pronunciation.substr(prevPos, pos - prevPos));

    /*
    cout << "Found : " << subString << endl;
    */

    phones.push_back(subString);
    prevPos = ++pos;
  }
  if (prevPos == 0)
    phones.push_back(pronunciation);
  else
    phones.push_back(pronunciation.substr(prevPos));

  return phones;
}

unsigned WFSTLexicon::_determineEndSymbol(const String& pronunciation)
{
  // determine end symbol marker
  unsigned endX = 0;
  _PronunciationListIterator pitr = _pronunciationList.find(pronunciation);
  if (pitr == _pronunciationList.end()) {
    endX = 1;
    _pronunciationList.insert(_PronunciationListValueType(pronunciation, endX));
  } else {
    endX = _pronunciationList[pronunciation] = ((*pitr).second + 1);
  }
  if (endX > _maxEndX)
    _maxEndX = endX;

  static char buffer[10];
  sprintf(buffer, "%d", endX);
  String endSymbol(_end + buffer);

  return inputLexicon()->index(endSymbol);
}

void WFSTLexicon::_addWord(const String& line)
{
  // parse dictionary entry
  String::size_type pos = line.find_first_of(' ');
  if (pos == 0)
    throw jconsistency_error("Dictionary entries may not begin with white space.");
  String word(line.substr(0, pos));
  String pronunciation(line.substr(pos+1));

  list<String> phones(_parsePronunciation(pronunciation));

  // build pronunciation
  NodePtr node(_branch);
  for (list<String>::iterator itr = phones.begin(); itr != phones.end(); itr++) {
    unsigned output = 0;
    if (itr == phones.begin())
      output = outputLexicon()->index(word);
    if (output >= _maxWordN)
      throw jindex_error("Index (%d) of word '%s' >= '_maxWordN (%d).", output, word.c_str(), _maxWordN);
    unsigned input = inputLexicon()->index(*itr);
    NodePtr nextNode(new Node(_nodesN++));
    EdgePtr edge(new Edge(node, nextNode, input, output));
    if (itr == phones.begin()) {
      _EdgeList& elist(_branchList[output]);
      elist.push_back(edge);
    } else {
      node->_addEdgeForce(edge);
    }
    node = nextNode;
  }

  // add word boundary marker
  unsigned endX = _determineEndSymbol(pronunciation);
  bool isBranch = _epsilonToBranch ? false : true;
  NodePtr nextNode(new Node(_nodesN++, isBranch));
  EdgePtr edge(new Edge(node, nextNode, endX, 0));
  node->_addEdgeForce(edge);

  if (_epsilonToBranch == false) return;

  static Weight logOneThird(log(3.0));

  // add silence and breath self loops
  unsigned arcsN = 2;
  if (_backOffInput != 0) arcsN++;
  Weight logWeight(log(arcsN + _breath.size()));
  if (_breath.size() != 0) {
    EdgePtr silLoop(new Edge(nextNode, nextNode, _sil, 0, logWeight));
    nextNode->_addEdgeForce(silLoop);

    for (_BreathSymbolsConstIterator itr = _breath.begin(); itr != _breath.end(); itr++) {
      unsigned symbol(*itr);
      EdgePtr breathLoop(new Edge(nextNode, nextNode, symbol, 0, logWeight));
      nextNode->_addEdgeForce(breathLoop);
    }
  }

  if (_backOffInput != 0) {
    // EdgePtr backOffLoop(new Edge(node, node, _backOffInput, _backOffOutput, logOneThird));
    // EdgePtr backOffLoop(new Edge(node, node, _backOffInput, _backOffOutput, _backOffWeight));
    EdgePtr backOffLoop(new Edge(nextNode, nextNode, _backOffInput, _backOffOutput, _backOffWeight));
    nextNode->_addEdgeForce(backOffLoop);
  }

  EdgePtr edgeToBranch(new Edge(nextNode, _branch, 0, 0, logOneThird));
  nextNode->_addEdgeForce(edgeToBranch);
}

void WFSTLexicon::_expandNode(NodePtr& node, WFSTSortedInput::NodePtr& grammarNode)
{
  static Weight logOneThird(log(3.0));
  set<unsigned> elist;

  // add arcs to match grammar arcs on the output side
  EdgePtr tail(NULL);
  for (WFSTSortedInput::Node::Iterator itr(_grammar, grammarNode); itr.more(); itr++) {
    WFSTSortedInput::EdgePtr& grammarEdge(itr.edge());

    unsigned output = grammarEdge->input();

    if (output == 0 || output == _eosOutput || output == _backOffOutput || elist.find(output) != elist.end()) continue;

    elist.insert(output);

    const _EdgeList& edgeList(_branchList[output]);

    if (edgeList.empty())
      throw jconsistency_error("No valid pronunciations for '%s'.", outputLexicon()->symbol(output).c_str());

    for (_EdgeListConstIterator eitr = edgeList.begin(); eitr != edgeList.end(); eitr++) {
      const EdgePtr& nextEdge(*eitr);
      Weight arcWeight(nextEdge->cost());

      if (node != _branch) arcWeight = TropicalSemiring::otimes(logOneThird, arcWeight);

      EdgePtr newEdge(new Edge(node, nextEdge->next(), nextEdge->input(), nextEdge->output(), arcWeight));
      if (tail.isNull())
	node->_addEdgeForce(newEdge);
      else
	tail->_edges() = newEdge;
      tail = newEdge;
    }
  }

  // add arc to 'endButOne'
  EdgePtr newEdge(new Edge(node, _endButOne, _sil, 0, logOneThird));
  node->_addEdgeForce(newEdge);

  // 'branch' requires no 'SIL' self loop
  // JMcD: is the backoff self-loop required?
  if (node == _branch) return;

  // add silence and breath self loops
  unsigned arcsN = 1;
  if (_backOffInput != 0) arcsN++;
  Weight logWeight(log(arcsN + _breath.size()));
  if (_breath.size() != 0) {
    EdgePtr silLoop(new Edge(node, node, _sil, 0, logWeight));
    node->_addEdgeForce(silLoop);

    for (_BreathSymbolsConstIterator itr = _breath.begin(); itr != _breath.end(); itr++) {
      unsigned symbol(*itr);
      EdgePtr breathLoop(new Edge(node, node, symbol, 0, logWeight));
      node->_addEdgeForce(breathLoop);
    }
  }

  if (_backOffInput != 0) {
    EdgePtr backOffLoop(new Edge(node, node, _backOffInput, _backOffOutput, logWeight));
    node->_addEdgeForce(backOffLoop);
  }
}

// check that all words in the output lexicon have valid pronunciations
void WFSTLexicon::_checkMissingPronunciations()
{
  bool missingWords = false;
  for (Lexicon::Iterator itr(outputLexicon()); itr.more(); itr++) {
    const String& word(itr.name());
    unsigned wordIndex = outputLexicon()->index(word);

    if (wordIndex == 0 || wordIndex == _sosOutput || wordIndex == _eosOutput) continue;

    const _EdgeList& edgeList(_branchList[wordIndex]);
    if (edgeList.empty()) {
      printf("No valid pronunciation for %s\n", word.c_str());
      missingWords = true;
    }
  }
  if (missingWords)
    throw jconsistency_error("Words missing from 'WFSTLexicon'.");
}

WFSTLexicon::_BreathSymbols WFSTLexicon::_breathSymbols(String breath)
{
  _BreathSymbols symbols;
  String::size_type pos;
  while((pos = breath.find_first_of(" ")) != String::npos) {
    String symbol(breath.substr(0, pos));
    symbols.push_back(inputLexicon()->index(symbol));

    breath = breath.substr(pos+1);
  }
  if (breath != "")
    symbols.push_back(inputLexicon()->index(breath));

  return symbols;
}

const WFSTLexicon::EdgePtr& WFSTLexicon::edges(WFSAcceptor::NodePtr& nd)
{
  throw jconsistency_error("This method is not supported for 'WFSTLexicon'.");  
}

const WFSTLexicon::EdgePtr& WFSTLexicon::edges(WFSAcceptor::NodePtr& nd, WFSTSortedInput::NodePtr& gram)
{
  NodePtr node(Cast<NodePtr>(nd));

  /*
  printf("Edge request for node %d : Dict node %d\n", node->index(), comp->index());  fflush(stdout);
  */

  // branch node?
  if (node->_isBranch == false) return node->_edges();

  node->_grammarNode = gram;  node->_edges() = NULL;
  _expandNode(node, node->_grammarNode);

  /*
  printf("Arcs for node %d : Dict node %d\n", node->index(), node->_dictNode->index()),
  node->writeArcs(stateLexicon(), inputLexicon(), outputLexicon());
  fflush(stdout);
  */

  static EdgePtr retList;
  retList = node->_edges();

  if (node == initial()) node->_edges() = NULL;

  return retList;
}


// ----- methods for class `WFSTLexicon::Edge' -----
//
WFSTLexicon::Edge::Edge(NodePtr& prev, NodePtr& next, unsigned input, unsigned output, Weight cost)
  : WFSTSortedOutput::Edge(prev, next, input, output, cost)
{
  _prev.disable();
  if (_next == _prev) _next.disable();
}

WFSTLexicon::Edge::~Edge()
{
  /*
  printf("Deleting WFSTLexicon::Edge from %d to %d\n",
	 _prev->index(), _next->index());  fflush(stdout);
  */
}

MemoryManager<WFSTLexicon::Edge>& WFSTLexicon::Edge::memoryManager()
{
  static MemoryManager<Edge> _MemoryManager("WFSTLexicon::Edge");
  return _MemoryManager;
}


// ----- methods for class `WFSTLexicon::Node' -----
//
MemoryManager<WFSTLexicon::Node>& WFSTLexicon::Node::memoryManager()
{
  static MemoryManager<Node> _MemoryManager("WFSTLexicon::Node");
  return _MemoryManager;
}


// ----- methods for helper class `WFSTContextDependency::StateName' -----
//
const unsigned WFSTContextDependency::StateName::MaxContextLength = 3;

WFSTContextDependency::
StateName::StateName(WFSTSortedInputPtr& dict, unsigned contextLen)
  : _dict(dict), _len(contextLen)
{
  if (contextLen > MaxContextLength)
    jdimension_error("Context length (%d) is greater than maximum (%d).",
		     contextLen > MaxContextLength);

  for (unsigned short i = 0; i < 2 * _len; i++)
    _names[i] = 0;
}

WFSTContextDependency::StateName::StateName(const StateName& src)
  : _dict(src._dict), _len(src._len)
{
  for (unsigned short i = 0; i < 2 * _len; i++)
    _names[i] = src._names[i];
}

WFSTContextDependency::StateName::~StateName() { }

WFSTContextDependency::StateName& WFSTContextDependency::
StateName::operator=(const StateName& src)
{
  if (_len != src._len)
    throw jconsistency_error("Context lengths (%d vs. %d) are not equal",
			     _len, src._len );

  for (unsigned short i = 0; i < 2 * _len; i++)
    _names[i] = src._names[i];

  return *this;
}

WFSTContextDependency::StateName WFSTContextDependency::StateName::
operator+(unsigned shift) const
{
  StateName stateName(*this);

  unsigned short ln = (2 * _len) - 1;
  for (unsigned short i = 0; i < ln; i++)
    stateName._names[i] = _names[i + 1];
  stateName._names[ln] = shift;

  return stateName;
}

WFSTContextDependency::StateName WFSTContextDependency::StateName::
operator+(const String& shift) const
{
  StateName stateName(*this);

  unsigned short ln = (2 * _len) - 1;
  for (unsigned short i = 0; i < ln; i++)
    stateName._names[i] = _names[i + 1];
  stateName._names[ln] = _dict->inputLexicon()->index(shift);

  return stateName;
}

unsigned WFSTContextDependency::StateName::left(unsigned l) const
{
  if (l == 0)
    throw jindex_error("Length is zero\n");

  if (l > _len)
    throw jindex_error("Length %d > %d\n", l, _len);
  return _names[_len - l];
}

unsigned WFSTContextDependency::StateName::right(unsigned l) const
{
  if (l == 0)
    throw jindex_error("Length is zero\n");

  if (l >= _len)
    throw jindex_error("Length %d >= %d\n", l, _len);
  return _names[_len + l];
}

String WFSTContextDependency::StateName::name(unsigned rc) const
{
  // base name
  String nm(_dict->inputLexicon()->symbol(phone()));

  // add left contexts
  nm += "/" + _dict->inputLexicon()->symbol(left(1));
  for (unsigned short i = 2; i <= _len; i++)
    nm += ";" + _dict->inputLexicon()->symbol(left(i));

  // add right contexts
  if (_len > 1) {
    nm += "_" + _dict->inputLexicon()->symbol(right(1));
    for (unsigned short i = 2; i < _len; i++)
      nm += ";" + right(i);

    nm += ";" + _dict->inputLexicon()->symbol(rc);
  } else
    nm += "_" + _dict->inputLexicon()->symbol(rc);

  return nm;
}

String WFSTContextDependency::StateName::name(const String& rc) const
{
  // base name
  String nm(_dict->inputLexicon()->symbol(phone()));

  // add left contexts
  nm += "/" + _dict->inputLexicon()->symbol(left(1));
  for (unsigned short i = 2; i <= _len; i++)
    nm += ";" + _dict->inputLexicon()->symbol(left(i));

  // add right contexts
  if (_len > 1) {
    nm += "_" + _dict->inputLexicon()->symbol(right(1));
    for (unsigned short i = 2; i < _len; i++)
      nm += ";" + _dict->inputLexicon()->symbol(right(i));

    nm += (rc == "") ? ";*" : ";" + rc;
  } else
    nm += (rc == "") ? "_*" : "_" + rc;

  return nm;
}

unsigned WFSTContextDependency::StateName::index() const
{
  if (_len > 2)
    throw jconsistency_error("Context length (%d) > 2.", _len);

  unsigned idx = 0;
  for (unsigned i = 0; i < 2 * _len; i++)
    idx = (idx << 7) + _names[i];

  return idx;
}


// ----- methods for class `WFSTContextDependency' -----
//
WFSTContextDependency::
WFSTContextDependency(LexiconPtr& statelex, LexiconPtr& inlex, WFSTSortedInputPtr& dict,
		      unsigned contextLen, const String& sil, const String& eps, const String& eos,
		      const String& end, const String& wb, const String& name)
  : WFSTSortedOutput(statelex, inlex, dict->inputLexicon(), name),
    _dict(dict), _contextLen(contextLen),
    _silInput(0), _silOutput(outputLexicon()->index(sil)), _eosOutput(outputLexicon()->index(eos)),
    _eps(eps), _end(end), _wb(wb)
{
  if (_contextLen == 0)
    throw jdimension_error("contextLen must be greater than 0");

  bool create = true;
  if (inputLexicon()->index(_eps, create) != 0)
    throw jindex_error("Input epsilon index (%d) should be zero.",
		       inputLexicon()->index(_eps));

  unsigned* ptr = (unsigned*) &_silInput;
  *ptr = inputLexicon()->index(sil, create);

  if (outputLexicon()->index(_eps) != 0)
    throw jindex_error("Output epsilon index (%d) should be zero.",
		       outputLexicon()->index(_eps));
}

WFSTContextDependency::NodePtr& WFSTContextDependency::initial(int idx)
{
  if (_initial.isNull())
    _initial = new Node(StateName(_dict, _contextLen));

  return Cast<NodePtr>(_initial);
}

const WFSTContextDependency::EdgePtr& WFSTContextDependency::edges(WFSAcceptor::NodePtr& nd)
{
  throw jconsistency_error("This method is not supported for 'WFSTContextDependency'.");
}

const WFSTContextDependency::EdgePtr& WFSTContextDependency::edges(WFSAcceptor::NodePtr& nd, WFSTSortedInput::NodePtr& comp)
{
  NodePtr node(Cast<NodePtr>(nd));

  /*
  printf("Edge request for node %d : Dict node %d\n", node->index(), comp->index());  fflush(stdout);
  */

  // are edges already expanded?
  if ((node->_edges().isNull() == false || node->isFinal() == true) && node->_dictNode == comp)
    return node->_edges();

  node->_dictNode = comp;  node->_edges() = NULL;
  _expandNode(node, node->_dictNode);

  /*
  printf("Arcs for node %d : Dict node %d\n", node->index(), node->_dictNode->index()),
  node->writeArcs(stateLexicon(), inputLexicon(), outputLexicon());
  fflush(stdout);
  */

  static EdgePtr retList;
  retList = node->_edges();

  if (node == initial()) node->_edges() = NULL;

  return retList;
}

void WFSTContextDependency::_expandNode(NodePtr& node, WFSTSortedInput::NodePtr& dictNode)
{
  bool create = true;

  /*
  printf("Expanding node %d : Dict node %d\n", node->index(), dictNode->index());  fflush(stdout);
  */

  // end node in dictionary: simulate null self-transitions
  if (dictNode->isFinal() && node->isFinal() == false) {
    unsigned phone = 0;

    StateName stateName(node->_stateName + phone);

    /*
    printf("State Name %s\n", stateName.name().c_str());  fflush(stdout);
    */

    unsigned multiphone;
    if (node->_stateName.phone() == 0) {
      multiphone = 0;
    } else if (node->_stateName.phone() == _silOutput) {
      multiphone = _silInput;
    } else
      multiphone = inputLexicon()->index(node->_stateName.name(phone), create);

    NodePtr newNode(new Node(stateName));

    if (stateName.phone() == _eosOutput)
      newNode->_setCost();

    EdgePtr edge(new Edge(node, newNode, multiphone, phone));

    /*
    printf("To End : Node A = %d : Node B = %d\n", node->index(), newNode->index());  fflush(stdout);
    edge->write(stateLexicon(), inputLexicon(), outputLexicon());
    */

    node->_addEdge(edge);
  }

  // normal case
  bool addedSelfLoops = false;
  bool addedEdgeToEnd = false;
  for (WFSTSortedInput::Node::Iterator itr(_dict, dictNode); itr.more(); itr++) {
    WFSTSortedInput::EdgePtr& dictEdge(itr.edge());
    unsigned phone = dictEdge->input();

    if (phone == 0) continue;

    if (node->_stateName.phone() == _eosOutput)
      throw jconsistency_error("Phone cannot be %s", outputLexicon()->symbol(_eosOutput).c_str());

    const String& trans(outputLexicon()->symbol(phone));
    if (trans.substr(0, 1) == _end) {	// handle word boundary makers
      if (addedSelfLoops == false) {
	_addSelfLoops(node, dictNode);
	addedSelfLoops = true;
      }
      continue;
    }

    StateName stateName(node->_stateName + phone);

    unsigned multiphone;
    if (node->_stateName.phone() == 0) {
      multiphone = 0;
    } else if (node->_stateName.phone() == _silOutput) {
      multiphone = _silInput;
    } else
      multiphone = inputLexicon()->index(node->_stateName.name(phone), create);

    NodePtr newNode(new Node(stateName));

    if (stateName.phone() == _eosOutput)
      newNode->_setCost();

    EdgePtr edge(new Edge(node, newNode, multiphone, phone));

    /*
    printf("Node A = %d : Node B = %d\n", node->index(), newNode->index());
    if (newNode->isFinal())
      printf("Node B is final\n");
    edge->write(stateLexicon(), inputLexicon(), outputLexicon());
    fflush(stdout);
    */

    node->_addEdgeForce(edge);
  }
}

void WFSTContextDependency::_addSelfLoops(NodePtr& node, WFSTSortedInput::NodePtr& dictNode)
{
  bool create = true;
  for (WFSTSortedInput::Node::Iterator itr(_dict, dictNode); itr.more(); itr++) {
    unsigned input = itr.edge()->input();

    static String trans;
    trans = outputLexicon()->symbol(input);

    if (trans.substr(0, 1) != _end) continue;

    unsigned  in  = inputLexicon()->index(trans,  create);
    EdgePtr   edgePtr(new Edge(node, node, in, input));
    node->_addEdge(edgePtr);
  }
}

WFSAcceptor::Node* WFSTContextDependency::_newNode(unsigned state)
{
  throw jconsistency_error("'WFSTContextDependency' should not be loaded statically.");

  return NULL;
}

WFSAcceptor::Edge* WFSTContextDependency::
_newEdge(NodePtr& from, NodePtr& to, unsigned input, unsigned output, Weight cost)
{
  throw jconsistency_error("'WFSTContextDependency' should not be loaded statically.");

  return NULL;
}


// ----- methods for class `WFSTContextDependency::Edge' -----
//
WFSTContextDependency::Edge::Edge(NodePtr& prev, NodePtr& next,
				  unsigned input, unsigned output, Weight cost)
  : WFSTSortedOutput::Edge(prev, next, input, output, cost)
{
  _prev.disable();
  if (_next == _prev) _next.disable();
}

WFSTContextDependency::Edge::~Edge()
{
  /*
  printf("Deleting WFSTContextDependency::Edge from %d to %d\n",
	 _prev->index(), _next->index());  fflush(stdout);
  */
}

MemoryManager<WFSTContextDependency::Edge>& WFSTContextDependency::Edge::memoryManager()
{
  static MemoryManager<Edge> _MemoryManager("WFSTContextDependency::Edge");
  return _MemoryManager;
}


// ----- methods for class `WFSTContextDependency::Node' -----
//
MemoryManager<WFSTContextDependency::Node>& WFSTContextDependency::Node::memoryManager()
{
  static MemoryManager<Node> _MemoryManager("WFSTContextDependency::Node");
  return _MemoryManager;
}


// ----- methods for class `WFSTHiddenMarkovModel' -----
//
WFSTHiddenMarkovModel::
WFSTHiddenMarkovModel(LexiconPtr& statelex, LexiconPtr& inlex, LexiconPtr& outlex, const DistribTreePtr& distribTree,
		      bool caching, const String& sil, const String& eps, const String& end, const String& name)
  : WFSTSortedOutput(statelex, inlex, outlex, name),
    _distribTree(distribTree), _caching(caching),
  _silInput(0), _silOutput(0), _eps(eps), _end(end)
{
  bool create = true;
  if (inputLexicon()->index(_eps, create) != 0)
    throw jindex_error("Input epsilon index (%d) should be zero.",
		       inputLexicon()->index(_eps));

  unsigned* ptr = (unsigned*) &_silInput;
  *ptr = inputLexicon()->index(sil, create);

  if (outputLexicon()->index(_eps, create) != 0)
    throw jindex_error("Output epsilon index (%d) should be zero.",
		       outputLexicon()->index(_eps));

   ptr = (unsigned*) &_silOutput;
  *ptr = outputLexicon()->index(sil, create);
}

WFSTHiddenMarkovModel::~WFSTHiddenMarkovModel() { }

WFSTHiddenMarkovModel::NodePtr& WFSTHiddenMarkovModel::initial(int idx)
{
  if (_context.isNull())
    throw jconsistency_error("Must set context before using 'WFSTHiddenMarkovModel'.");

  if (_initial.isNull())
    _initial = new Node(/* output= */0);

  return Cast<NodePtr>(_initial);
}

void WFSTHiddenMarkovModel::
_addSelfLoops(NodePtr& node, WFSTSortedInput::NodePtr& contextNode)
{
  bool create = true;
  for (WFSTSortedInput::Node::Iterator itr(_context, contextNode); itr.more(); itr++) {

    WFSTSortedInput::EdgePtr& contextEdge(itr.edge());
    // cout << "Looking at edge " << contextEdge.the_p << " in _addSelfLoops." << endl;

    unsigned input = itr.edge()->input();

    const String& trans(outputLexicon()->symbol(input));

    if (trans.substr(0, 1) != _end) continue;

    unsigned  in  = inputLexicon()->index(trans,  create);
    EdgePtr   edgePtr(new Edge(node, node, in, input));
    node->_addEdge(edgePtr);
  }
}

const String& WFSTHiddenMarkovModel::_transSymbol(const NodePtr& node)
{
  const String& outputSymbol(outputLexicon()->symbol(node->_output));

  /*
  cout << "Searching for symbol " << outputSymbol << endl;
  */

  if (outputSymbol == "SIL" || outputSymbol == "{SIL:WB}")
    return _distribTree->middle(outputSymbol);

  /*
  String begSymbol(_distribTree->begin(outputSymbol));
  String midSymbol(_distribTree->middle(outputSymbol));
  String endSymbol(_distribTree->end(outputSymbol));
  printf("%40s --> %10s %10s %10s\n", outputSymbol.c_str(), begSymbol.c_str(), midSymbol.c_str(), endSymbol.c_str());
  fflush(stdout);
  */

  switch (node->_state) {
  case 1: return _distribTree->begin(outputSymbol);  break;
  case 2: return _distribTree->middle(outputSymbol); break;
  case 3: return _distribTree->end(outputSymbol);    break;
  default:
    throw jindex_error("Node '_state' (%d) should be in [1, 3].\n", node->_state);
  }
}

void WFSTHiddenMarkovModel::_expandPhone(NodePtr& node)
{
  if (_caching && node->_edges().isNull() == false) return;

  static Weight logOneHalf(log(2.0));

  const String& trans(_transSymbol(node));
  unsigned      input = inputLexicon()->index(trans);

  /*
  printf("Node %d : Adding Self Loop %s\n", node->index(), trans.c_str());  fflush(stdout);
  */

  EdgePtr selfLoop(new Edge(node, node, input, 0, logOneHalf));
  node->_addEdgeForce(selfLoop);

  if (node->_state != 3) {
    bool isSilence = false;
    NodePtr       newNode(new Node(node->_output, isSilence, node->_state + 1));
    const String& newTrans(_transSymbol(newNode));
    unsigned      newInput = inputLexicon()->index(newTrans);

    /*
    printf("Node %d : Adding Transition %s to Node %d\n", node->index(), newTrans.c_str(), newNode->index());  fflush(stdout);
    */

    EdgePtr edge(new Edge(node, newNode, newInput, 0, logOneHalf));
    node->_addEdgeForce(edge);
  }
}

void WFSTHiddenMarkovModel::_expandSilence(NodePtr& node, unsigned statesN)
{
  if (_caching && node->_edges().isNull() == false) return;
    
  bool create = true;
  static Weight logOneHalf(log(2.0));

  static const String trans("SIL-m");
  unsigned     input = inputLexicon()->index(trans, create);

  EdgePtr selfLoop(new Edge(node, node, input, 0, logOneHalf));
  node->_addEdgeForce(selfLoop);

  if (node->_state != statesN) {
    bool isSilence = true;
    NodePtr newNode(new Node(node->_output, isSilence, node->_state + 1));

    EdgePtr edge(new Edge(node, newNode, input, 0, logOneHalf));
    node->_addEdgeForce(edge);
  }
}

const WFSTHiddenMarkovModel::EdgePtr& WFSTHiddenMarkovModel::edges(WFSAcceptor::NodePtr& nd)
{
  throw jconsistency_error("This method is not supported for 'WFSTHiddenMarkovModel'.");
}

const WFSTHiddenMarkovModel::EdgePtr& WFSTHiddenMarkovModel::edges(WFSAcceptor::NodePtr& nd, WFSTSortedInput::NodePtr& comp)
{
  NodePtr node(Cast<NodePtr>(nd));

  // only branch node need be expanded
  if (node->_edges().isNull() == false && node->_context == comp)
    return node->_edges();

  node->_context = comp;  node->_edges() = NULL;
  _expandNode(node, node->_context);

  static EdgePtr retList;
  retList = node->_edges();

  if (node->_state == 0) node->_edges() = NULL;

  return retList;
}

void WFSTHiddenMarkovModel::_expandNode(NodePtr& node, WFSTSortedInput::NodePtr& contextNode)
{
  bool create = true;

  //cout << "Expanding node " << node->index() << endl;

  if (node->_state != 0) {
    if (node->_isSilence)
      _expandSilence(node);
    else
      _expandPhone(node);
  }

  if (node->_state == 0 || (node->_isSilence && node->_state == 4) || (node->_isSilence == false && node->_state == 3)) {
    static Weight arcWeight(log(2.0));

    bool addedSelfLoops = false;
    for (WFSTSortedInput::Node::Iterator itr(_context, contextNode); itr.more(); itr++) {
      WFSTSortedInput::EdgePtr& contextEdge(itr.edge());

      /*
      cout << "edge from " << contextEdge->prev()->index() << " to " << contextEdge->next()->index()
	   << ": input symbol " << _context->inputLexicon()->symbol(contextEdge->input())
	   << ": output symbol " << _context->outputLexicon()->symbol(contextEdge->output()) << endl;
      */

      unsigned polyphone = contextEdge->input();

      // handle null transition
      if (polyphone == 0) continue;

      // handle word boundary makers
      const String& trans(outputLexicon()->symbol(polyphone));
      if (trans.substr(0, 1) == _end) {
	if (addedSelfLoops == false) {
	  _addSelfLoops(node, contextNode);
	  addedSelfLoops = true;
	}
	continue;
      }

      if (_caching) {
	_EdgeMapIterator eitr = _edgeMap.find(polyphone);
	if (eitr != _edgeMap.end()) {
	  EdgePtr& cacheEdge((*eitr).second);
	  EdgePtr edge(new Edge(node, cacheEdge->next(), cacheEdge->input(), polyphone, arcWeight));
	  node->_addEdgeForce(edge);
	  continue;
	}
      }

      // normal case
      bool isSilence = (polyphone == _silOutput);
      NodePtr newNode(new Node(polyphone, isSilence, /* idx= */ 1));
      unsigned newInput = inputLexicon()->index(_transSymbol(newNode), create);
      EdgePtr edge(new Edge(node, newNode, newInput, polyphone, arcWeight));
      node->_addEdgeForce(edge);

      /*
      printf("Node %d : Adding Transition %s to Node %d\n", node->index(), _transSymbol(newNode).c_str(), newNode->index());  fflush(stdout);
      */

      if (_caching) _edgeMap.insert(_EdgeMapValueType(polyphone, edge));
    }
  }
}


// ----- methods for class `WFSTHiddenMarkovModel::Edge' -----
//
WFSTHiddenMarkovModel::Edge::Edge(NodePtr& prev, NodePtr& next,
				  unsigned input, unsigned output, Weight cost)
  : WFSTSortedOutput::Edge(prev, next, input, output, cost)
{
  _prev.disable();
  if (_next == _prev) _next.disable();
}

WFSTHiddenMarkovModel::Edge::~Edge()
{
  /*
  printf("Deleting WFSTHiddenMarkovModel::Edge from %d to %d\n",
	 _prev->index(), _next->index());  fflush(stdout);
  */
}

MemoryManager<WFSTHiddenMarkovModel::Edge>& WFSTHiddenMarkovModel::Edge::memoryManager()
{
  static MemoryManager<Edge> _MemoryManager("WFSTHiddenMarkovModel::Edge");
  return _MemoryManager;
}


// ----- methods for class `WFSTHiddenMarkovModel::Node' -----
//
WFSTHiddenMarkovModel::Node::
Node(unsigned output, bool isSilence, unsigned short idx)
  : WFSTSortedOutput::Node(_convertIndex(output, idx)),
  _isSilence(isSilence), _output(output), _state(idx), _context(NULL)
{
  if (idx == 4) _setCost();
}

unsigned WFSTHiddenMarkovModel::Node::_convertIndex(unsigned polyphoneX, unsigned short stateX)
{
  static unsigned short staticStatesN = 1;
  static unsigned long  MaxPolyphoneX = UINT_MAX / 8;

  if (polyphoneX >= MaxPolyphoneX)
    throw jindex_error("polyphoneX (%d) >= %d.\n", polyphoneX, MaxPolyphoneX);

  if (stateX > 4)
    throw jindex_error("stateX (%d) > 4.\n", stateX);

  unsigned idx = polyphoneX; idx = idx << 3;
  return idx + stateX + staticStatesN;
}

MemoryManager<WFSTHiddenMarkovModel::Node>& WFSTHiddenMarkovModel::Node::memoryManager()
{
  static MemoryManager<Node> _MemoryManager("WFSTHiddenMarkovModel::Node");
  return _MemoryManager;
}


// ----- methods for class `WFSTCompare' -----
//
WFSTCompare::NodePtr& WFSTCompare::initial(int idx)
{
  NodePtr& initial(Cast<NodePtr>(_initial));

  if (initial->_compNode.isNull())
    initial->setComp(_comp->initial());

  return initial;
}

const double WFSTCompare::CostTolerance = 1.0E-02;

const WFSTCompare::EdgePtr& WFSTCompare::edges(WFSAcceptor::NodePtr& nd)
{
  NodePtr node(Cast<NodePtr>(nd));

  bool edgesN = false, inputSymbol = false, outputSymbol = false, arcCost = false, nodeFinal = false, nodeCost = false;

  if (node->isFinal() != node->_compNode->isFinal()) nodeFinal = true;

  float nodeC     = float(node->cost());
  float compNodeC = float(node->_compNode->cost());
  if ((fabs(nodeC - compNodeC) / fabs(nodeC + compNodeC)) > CostTolerance) nodeCost = true;

  static EdgePtr edgeList;
  edgeList = Cast<EdgePtr>(WFSTSortedInput::edges(node));
  EdgePtr edge(edgeList);
  for (WFSTSortedInput::Node::Iterator itr(_comp, node->_compNode); itr.more(); itr++) {

    if (edge.isNull()) { edgesN = true; break; }

    unsigned input      = edge->input();
    unsigned output     = edge->output();
    float    c          = edge->cost();

    WFSTSortedInput::EdgePtr& compEdge(itr.edge());
    unsigned compInput  = compEdge->input();
    unsigned compOutput = compEdge->output();
    float    compC      = compEdge->cost();

    if (input  != compInput)  inputSymbol  = true;
    if (output != compOutput) outputSymbol = true;
    if (((fabs(c - compC) / fabs(c + compC)) > CostTolerance) && (fabs(c - compC) > 1.0E-04)) arcCost = true;

    edge->next()->setComp(compEdge->next());
    edge = edge->_edges();
  }

  if (edge.isNull() == false) edgesN = true;

  if (edgesN) {
    printf("Error: Number of edges does not match for nodes %d and %d.\n",
	   node->index(), node->_compNode->index());  fflush(stdout);
  }
  if (inputSymbol) {
    printf("Error: Input symbols do not match for nodes %d and %d.\n",
	   node->index(), node->_compNode->index());  fflush(stdout);
  }
  if (outputSymbol) {
    printf("Error: Output symbols do not match for nodes %d and %d.\n",
	   node->index(), node->_compNode->index());  fflush(stdout);
  }
  if (arcCost) {
    printf("Error: Arc costs do not match for nodes %d and %d.\n",
	   node->index(), node->_compNode->index());  fflush(stdout);
  }
  if (nodeFinal) {
    printf("Error: Node final is not the same for nodes %d and %d.\n",
	   node->index(), node->_compNode->index());  fflush(stdout);
  }
  if (nodeCost) {
    printf("Error: Node costs (%g vs. %g) do not match for nodes %d and %d.\n",
	   nodeC, compNodeC, node->index(), node->_compNode->index());  fflush(stdout);
  }

  if (edgesN || inputSymbol || outputSymbol || arcCost || nodeCost) {
    printf("Arcs from node:\n");
      node->writeArcs(stateLexicon(), inputLexicon(), outputLexicon());
      printf("\nArcs from comparison node:\n");
      node->_compNode->writeArcs(_comp->stateLexicon(), _comp->inputLexicon(), _comp->outputLexicon());
      fflush(stdout);
  }

  return edgeList;
}

WFSAcceptor::Node* WFSTCompare::_newNode(unsigned state)
{
  return new Node(state);
}

WFSAcceptor::Edge* WFSTCompare::
_newEdge(NodePtr& from, NodePtr& to, unsigned input, unsigned output, Weight cost)
{
  return new Edge(from, to, input, output, cost);
}


// ----- methods for class `WFSTCompare::Node' -----
//
MemoryManager<WFSTCompare::Node>& WFSTCompare::Node::memoryManager()
{
  static MemoryManager<Node> _MemoryManager("WFSTCompare::Node");
  return _MemoryManager;
}


// ----- methods for class `WFSTCompare::Edge' -----
//
MemoryManager<WFSTCompare::Edge>& WFSTCompare::Edge::memoryManager()
{
  static MemoryManager<Edge> _MemoryManager("WFSTCompare::Edge");
  return _MemoryManager;
}


// ----- methods for helper class `ContextDependencyTransducer::StateName' -----
//
const unsigned ContextDependencyTransducer::StateName::MaxContextLength = 3;

ContextDependencyTransducer::StateName::StateName(unsigned contextLen, const String& beg)
  : _len(contextLen), _depth(0), _next(NULL)
{
  if (contextLen > MaxContextLength)
    jdimension_error("Context length (%d) is greater than maximum (%d).",
		     contextLen > MaxContextLength);

  unsigned idx = _phoneLexicon->index(beg);
  for (unsigned short i = 0; i < 2 * _len; i++)
    _names[i] = idx;
}

ContextDependencyTransducer::StateName::StateName(const StateName& src)
  : _len(src._len), _depth(src._depth), _next(NULL)
{
  for (unsigned short i = 0; i < 2 * _len; i++)
    _names[i] = src._names[i];
}

ContextDependencyTransducer::StateName::~StateName() { }

ContextDependencyTransducer::StateName& ContextDependencyTransducer::
StateName::operator=(const StateName& src)
{
  if (_len != src._len)
    throw jconsistency_error("Context lengths (%d vs. %d) are not equal",
			     _len, src._len );

  for (unsigned short i = 0; i < 2 * _len; i++)
    _names[i] = src._names[i];

  return *this;
}

ContextDependencyTransducer::StateName ContextDependencyTransducer::StateName::
operator+(unsigned shift) const
{
  StateName stateName(*this);
  stateName.incDepth();

  unsigned short ln = (2 * _len) - 1;
  for (unsigned short i = 0; i < ln; i++)
    stateName._names[i] = _names[i + 1];
  stateName._names[ln] = shift;

  return stateName;
}

ContextDependencyTransducer::StateName ContextDependencyTransducer::StateName::
operator+(const String& shift) const
{
  StateName stateName(*this);
  stateName.incDepth();

  unsigned short ln = (2 * _len) - 1;
  for (unsigned short i = 0; i < ln; i++)
    stateName._names[i] = _names[i + 1];
  stateName._names[ln] = _phoneLexicon->index(shift);

  return stateName;
}

const String& ContextDependencyTransducer::StateName::left(unsigned l) const
{
  if (l == 0)
    throw jindex_error("Length is zero\n");

  if (l > _len)
    throw jindex_error("Length %d > %d\n", l, _len);
  return _phoneLexicon->symbol(_names[_len - l]);
}

const String& ContextDependencyTransducer::StateName::right(unsigned l) const
{
  if (l == 0)
    throw jindex_error("Length is zero\n");

  if (l >= _len)
    throw jindex_error("Length %d >= %d\n", l, _len);
  return _phoneLexicon->symbol(_names[_len + l]);
}

String ContextDependencyTransducer::StateName::name(unsigned rc) const
{
  // base name
  String nm(phone());

  // add left contexts
  nm += "/" + left(1);
  for (unsigned short i = 2; i <= _len; i++)
    nm += ";" + left(i);

  // add right contexts
  if (_len > 1) {
    nm += "_" + right(1);
    for (unsigned short i = 2; i < _len; i++)
      nm += ";" + right(i);

    nm += ";" + _phoneLexicon->symbol(rc);
  } else
    nm += "_" + _phoneLexicon->symbol(rc);
  
  return nm;
}

String ContextDependencyTransducer::StateName::name(const String& rc) const
{
  // base name
  String nm(phone());

  // add left contexts
  nm += "/" + left(1);
  for (unsigned short i = 2; i <= _len; i++)
    nm += ";" + left(i);

  // add right contexts
  if (_len > 1) {
    nm += "_" + right(1);
    for (unsigned short i = 2; i < _len; i++)
      nm += ";" + right(i);

    nm += (rc == "") ? ";*" : ";" + rc;
  } else
    nm += (rc == "") ? "_*" : "_" + rc;
  
  return nm;
}

bool ContextDependencyTransducer::StateName::rightContextContains(const String& sym) const
{
  for (unsigned short i = 1; i < _len; i++)
    if (right(i) == sym) return true;

  return false;
}

bool ContextDependencyTransducer::StateName::rightMostContextContains(const String& sym) const
{
  return right(_len-1).find(sym) != String::npos;
}


// ----- methods for helper class `ContextDependencyTransducer::StateNameList' -----
//
ContextDependencyTransducer::StateNameList::StateNameList()
  : _stateName(NULL) { }

ContextDependencyTransducer::StateNameList::~StateNameList() { clear(); }

void ContextDependencyTransducer::StateNameList::clear()
{
  while (_stateName != NULL) {
    StateName* state = _stateName;
    _stateName = _stateName->_next;
    delete state;
  }
}

void ContextDependencyTransducer::StateNameList::push(const StateName& stateName, unsigned index)
{
  if (isPresent(index))
    throw jindex_error("Index %d is already present on 'StateNameList'.");

  _indexSet.insert(index);

  StateName* state = new StateName(stateName);
  state->_next = _stateName;
  _stateName = state;
}

ContextDependencyTransducer::StateName* ContextDependencyTransducer::StateNameList::pop()
{
  StateName* state = _stateName;

  if (_stateName != NULL)
    _stateName = _stateName->_next;

  return state;
}

bool ContextDependencyTransducer::StateNameList::isPresent(unsigned index) const
{
  _IndexSetConstIterator itr = _indexSet.find(index);

  return itr != _indexSet.end();
}


// ----- methods for class `ContextDependencyTransducer' -----
//
LexiconPtr		ContextDependencyTransducer::_phoneLexicon;
WFSTransducerPtr	ContextDependencyTransducer::_dict;
unsigned 		ContextDependencyTransducer::_cnt;

ContextDependencyTransducer::
ContextDependencyTransducer(unsigned contextLen,
			    const String& sil, const String& eps,
			    const String& end, const String& wb, const String& eos)
  : _contextLen(contextLen), _sil(sil), _eps(eps), _end(end), _wb(wb), _eos(eos)
{
  if (_contextLen == 0)
    throw jdimension_error("contextLen must be greater than 0");
}

void ContextDependencyTransducer::_addSelfLoops(WFSTransducerPtr& wfst, NodePtr& oldNode, const NodePtr& dictNode) const
{
  bool create = true;

  for (WFSTransducer::Node::Iterator itr(dictNode); itr.more(); itr++) {
    unsigned input = itr.edge()->input();

    String trans(wfst->inputLexicon()->symbol(input));

    if (trans.substr(0,1) != _end) continue;

    unsigned  in  = wfst->inputLexicon()->index(trans,  create);
    unsigned  out = wfst->outputLexicon()->index(trans, create);
    EdgePtr   edgePtr(new Edge(oldNode, oldNode, in, out));
    oldNode->_addEdge(edgePtr);
  }
}

// expand a given node to the end state
void ContextDependencyTransducer::
_expandToEnd(WFSTransducerPtr& wfst, StateName stateName)
{
  bool create = true;

  _expandNode(wfst, stateName, _branch);

  StateName newStateName(stateName + _sil);
  String newTransName(stateName.name(_sil));

  _expandNode(wfst, newStateName, _branch);

  for (unsigned j = 0; j <= _contextLen; j++) {

    /*
    printf("Expanding 2.5: From State : %s\n", String(stateName).c_str());
    printf("Expanding 2.5: To State   : %s\n", String(newStateName).c_str());
    printf("Expanding 2.5: Trans      : %s\n", newTransName.c_str());
    fflush(stdout);
    */

    unsigned oldIndex = wfst->stateLexicon()->index(stateName);
    NodePtr  oldNode(wfst->find(oldIndex));
    
    unsigned newIndex = wfst->stateLexicon()->index(newStateName, create);
    NodePtr  newNode(wfst->find(newIndex, create));

    // _sil is context independent
    if (stateName.phone().find(_sil) != String::npos)
      newTransName = _sil;
    else if (stateName.phone().find("GARBAGE") != String::npos)
      newTransName = "GARBAGE";
    else if (stateName.phone().find("+FILLER+") != String::npos)
      newTransName = "+FILLER+";
    else if (stateName.phone().find("+BREATH+") != String::npos)
      newTransName = "+BREATH+";

    unsigned  in  = (j == 0) ? wfst->inputLexicon()->index(_sil) : wfst->inputLexicon()->index(_eos);
    unsigned  out = wfst->outputLexicon()->index(newTransName, create);

    //    cout << "index = " << oldNode->index() << endl;

    EdgePtr   edgePtr(new Edge(oldNode, newNode, in, out));
    oldNode->_addEdge(edgePtr);

    if (j == _contextLen && wfst->hasFinal(newIndex) == false) wfst->_addFinal(newIndex);

    stateName    = newStateName;
    newTransName = newStateName.name(0);
    newStateName = newStateName + 0;
  }
}

// Second Phase: First expand nodes in 'current' with phonemes in
//               '_inputLexicon' to fill up the context. Then
// 		 expand nodes in 'current' with phonemes in
//               '_inputLexicon' to complete the automaton
void ContextDependencyTransducer::
_expandNode(WFSTransducerPtr& wfst, const StateName& stateName, const NodePtr& dictNode)
{
  bool create = true;

  unsigned  stateIndex = wfst->stateLexicon()->index(stateName, create);
  NodePtr   oldNode(wfst->find(stateIndex, create));

  // if (++_cnt % 100000 == 0) { printf(".");  fflush(stdout); }

  bool expandedToEnd = false;
  for (WFSTransducer::Node::Iterator itr(dictNode); itr.more(); itr++) {
    unsigned input  = itr.edge()->input();
    const String& trans(wfst->inputLexicon()->symbol(input));

    // handle an end node
    if (trans.substr(0, 1) == _end) {
      if (expandedToEnd == false) {
	/*
	printf("End         = %s\n", _end.c_str());
	printf("Trans       = %s\n", trans.c_str());
	printf("State Index = %d\n", stateIndex);
	printf("Expanding %s to end.\n", String(stateName).c_str());
	fflush(stdout);
	*/
	_addSelfLoops(wfst, oldNode, dictNode);
	if (_stateNameList.isPresent(stateIndex) == false)
	  _stateNameList.push(stateName, stateIndex);

	expandedToEnd = true;
      }
      continue;
    }

    StateName newStateName(stateName + trans);

    // The correct ouput label is not known until the context is full.
    String newTransName(_eps);
    if (stateName.depth() == 0) {
      newTransName = "SIL";
    } else if (stateName.depth() > _contextLen) {
      newTransName = stateName.name(trans);

      // _sil is context independent
      if (stateName.phone().find(_sil) != String::npos)
	newTransName = _sil;
      else if (stateName.phone().find("GARBAGE") != String::npos)
	newTransName = "GARBAGE";
      else if (stateName.phone().find("+FILLER+") != String::npos)
	newTransName = "+FILLER+";
      else if (stateName.phone().find("+BREATH+") != String::npos)
	newTransName = "+BREATH+";

    }

    /*
    printf("In Trans  3: %s\n",  trans.c_str());
    printf("Out Trans 3: %s\n",  newTransName.c_str());
    printf("State Name:  %s\n", String(newStateName).c_str());
    fflush(stdout);
    */

    unsigned  newIndex = wfst->stateLexicon()->index(newStateName, create);
    NodePtr   newNode(wfst->find(newIndex, create));

    unsigned  in  = wfst->inputLexicon()->index(trans);
    unsigned  out = wfst->outputLexicon()->index(newTransName, create);
    EdgePtr   edgePtr(new Edge(oldNode, newNode, in, out));
    oldNode->_addEdge(edgePtr);

    _expandNode(wfst, newStateName, itr.edge()->next());
  }
}

// this class produces a context-depenency phoneme
// transducer based on:
//   1. lexicon of context-independent, input symbols 'inlex'
//   2. context dependency length 'contextLen'
//   3. word start symbol 'beg'
//   4. word end symbol 'end'
//   5. no. of word end symbols
//
WFSTransducerPtr ContextDependencyTransducer::build(LexiconPtr& inputLexicon, WFSTransducerPtr& dict, const String& name)
{
  _phoneLexicon = inputLexicon;  _dict = dict;
  LexiconPtr outputLexicon(new Lexicon("Output Lexicon"));
  LexiconPtr stateLexicon(new Lexicon("State Lexicon"));

  bool create = true;
  outputLexicon->index(_eps, create);
  WFSTransducerPtr wfst(new WFSTransducer(stateLexicon, inputLexicon, outputLexicon, name));

  _cnt = 0;

  cout << "Initial node = " << endl;
  dict->initial()->write();

  StateName start(_contextLen,  _eps);  stateLexicon->index(start, create);

  const WFSTransducer::NodePtr& dictStartNode(dict->initial());
  WFSTransducer::Node::Iterator dictStartItr(dictStartNode);

  _branch = dictStartItr.edge()->next();

  // make original expansion from start node
  cout << "Expanding interword contexts ... " << endl;
  _stateNameList.clear();
  _expandNode(wfst, start, dict->initial());

  // now expand all cross-word contexts
  cout << "Expanding " << _stateNameList.size() << " crossword contexts ... " << endl;
  unsigned cnt = 0;
  StateName* stateName;
  while((stateName = _stateNameList.pop()) != NULL) {
    cout << "    " << ++cnt << ". " << stateName->name() << endl;
    _expandToEnd(wfst, *stateName);
    delete stateName;
  }

  return wfst;
}

WFSTransducerPtr 
buildContextDependencyTransducer(LexiconPtr& inputLexicon, WFSTransducerPtr& dict, unsigned contextLen, const String& sil,
				 const String& eps, const String& end, const String& wb, const String& eos)
{
  ContextDependencyTransducer contextBuilder(contextLen, sil, eps, end, wb, eos);
  return contextBuilder.build(inputLexicon, dict);
}


// ----- methods for class `HiddenMarkovModelTransducer' -----
//
void HiddenMarkovModelTransducer::
_addSelfLoops(WFSTransducerPtr& wfst, NodePtr& startNode, unsigned noEnd) const
{
  bool create = true;

  for (unsigned i = 1; i <= noEnd; i++) {
    char buffer[100];
    sprintf(buffer, "%d", i);
    String mark(_end + String(buffer));

    unsigned in  = wfst->inputLexicon()->index(mark, create);
    unsigned out = wfst->outputLexicon()->index(mark, create);

    // self-loop for initial state
    EdgePtr selfLoop(new Edge(startNode, startNode, in, out));
    startNode->_addEdgeForce(selfLoop);
  }
}

void HiddenMarkovModelTransducer::
_expandSilence(WFSTransducerPtr& wfst, const DistribTreePtr& dt,
	       NodePtr& startNode, NodePtr& finalNode, unsigned nStates) const
{
  bool create = true;
  static Weight logOneHalf(log(2.0));
  // static Weight logOneHalf(0.0);

  //unsigned inIndex  = wfst->inputLexicon()->index(_sil+"-m");
  unsigned inIndex  = wfst->inputLexicon()->index(_sil+"(|)-m");

  // Create the required number of silence states
  NodePtr fromNode(startNode);
  for (unsigned s = 0; s < nStates; s++) {
    unsigned outIndex = (s == 0) ? wfst->outputLexicon()->index(_sil) : 0;

    static char buffer[100];
    sprintf(buffer, "-%d", s);
    String stateName(_sil + buffer);

    cout << "Creating " << stateName << endl;

    unsigned stateIndex = wfst->stateLexicon()->index(stateName, create);
    NodePtr toNode(wfst->find(stateIndex, create));

    EdgePtr edge(new Edge(fromNode, toNode, inIndex, outIndex, logOneHalf));
    fromNode->_addEdgeForce(edge);
    EdgePtr selfLoop(new Edge(toNode, toNode, inIndex, 0, logOneHalf));
    toNode->_addEdgeForce(selfLoop);

    fromNode = toNode;
  }

  // Loop to final state
  EdgePtr edgeFinal(new Edge(fromNode, finalNode, 0, 0));
  fromNode->_addEdgeForce(edgeFinal);
}


void HiddenMarkovModelTransducer::
_expandPhone(WFSTransducerPtr& wfst, const String& outSymbol, const DistribTreePtr& dt,
	     NodePtr& startNode, NodePtr& finalNode) const
{
  bool create = true;
  static Weight logOneHalf(log(2.0));
  // static Weight logOneHalf(0.0);

  String begSymbol(dt->begin(outSymbol));
  String midSymbol(dt->middle(outSymbol));
  String endSymbol(dt->end(outSymbol));

  if (begSymbol == "+BREATH+(|)-b") {
    begSymbol = "+BREATH+-b";
    midSymbol = "+BREATH+-m";
    endSymbol = "+BREATH+-e";
  }

  if (begSymbol == "+FILLER+(|)-b") {
    begSymbol = "+FILLER+-b";
    midSymbol = "+FILLER+-m";
    endSymbol = "+FILLER+-e";
  }

  if (begSymbol == "GARBAGE(|)-b") {
    begSymbol = "GARBAGE-m";
    midSymbol = "GARBAGE-m";
    endSymbol = "GARBAGE-m";
  }

  printf("%40s --> %10s %10s %10s\n", outSymbol.c_str(), begSymbol.c_str(), midSymbol.c_str(), endSymbol.c_str());

  unsigned begState  = wfst->stateLexicon()->index(outSymbol + "-b", create);
  unsigned midState  = wfst->stateLexicon()->index(outSymbol + "-m", create);
  unsigned endState  = wfst->stateLexicon()->index(outSymbol + "-e", create);

  NodePtr begNode(wfst->find(begState, create));
  NodePtr midNode(wfst->find(midState, create));
  NodePtr endNode(wfst->find(endState, create));

  unsigned  begIndex = wfst->inputLexicon()->index(begSymbol);
  unsigned  midIndex = wfst->inputLexicon()->index(midSymbol);
  unsigned  endIndex = wfst->inputLexicon()->index(endSymbol);
  unsigned  outIndex = wfst->outputLexicon()->index(outSymbol);

  // Begin State
  EdgePtr edge0(new Edge(startNode, begNode, begIndex, outIndex, logOneHalf));
  startNode->_addEdgeForce(edge0);
  EdgePtr selfLoop0(new Edge(begNode, begNode, begIndex, 0, logOneHalf));
  begNode->_addEdgeForce(selfLoop0);

  // Middle State
  EdgePtr edge1(new Edge(begNode, midNode, midIndex, 0, logOneHalf));
  begNode->_addEdgeForce(edge1);
  EdgePtr selfLoop1(new Edge(midNode, midNode, midIndex, 0, logOneHalf));
  midNode->_addEdgeForce(selfLoop1);

  // End State
  EdgePtr edge2(new Edge(midNode, endNode, endIndex, 0, logOneHalf));
  midNode->_addEdgeForce(edge2);
  EdgePtr selfLoop2(new Edge(endNode, endNode, endIndex, 0, logOneHalf));
  endNode->_addEdgeForce(selfLoop2);

  // Loop to final state
  EdgePtr edge4(new Edge(endNode, finalNode, 0, 0));
  endNode->_addEdgeForce(edge4);
}

WFSTransducerPtr HiddenMarkovModelTransducer::
build(LexiconPtr& inputLexicon, LexiconPtr& outputLexicon, const DistribTreePtr& distribTree,
      unsigned noEnd, const String& name) const
{
  bool create = true;

  LexiconPtr stateLexicon(new Lexicon("State Lexicon"));

  WFSTransducerPtr wfst(new WFSTransducer(stateLexicon, inputLexicon, outputLexicon, name));

  String start("START");
  stateLexicon->index(start, create);
  
  String branch("BRANCH");
  NodePtr branchNode(wfst->find(stateLexicon->index(branch, create), create));

  String finish("FINISH");
  unsigned finalIndex = stateLexicon->index(finish, create);
  wfst->_addFinal(finalIndex);
  NodePtr finalNode(wfst->find(finalIndex));

  // add edge from 'initial()' to first branch node
  EdgePtr edge(new Edge(wfst->initial(), branchNode, /* in = */ 0, /* out= */ 0));
  wfst->initial()->_addEdgeForce(edge);

  // add required self-loops to branch node
  _addSelfLoops(wfst, branchNode, noEnd);

  // Loop from final node back to branch node
  EdgePtr edge3(new Edge(finalNode, branchNode, 0, 0));
  finalNode->_addEdgeForce(edge3);

  // expand all symbols in 'outputLexicon'
  for (Lexicon::Iterator itr(outputLexicon); itr.more(); itr++) {
    const String& outSymbol(*itr);

    // String outSymbol("+FILLER+");

    if (outSymbol == _eps || outSymbol.substr(0, 1) == _end) continue;

    try {
      if (outSymbol == _sil)
	_expandSilence(wfst, distribTree, branchNode, finalNode);
      else
	_expandPhone(wfst, outSymbol, distribTree, branchNode, finalNode);
    } catch (exception e) {
      printf("Could not expand polyphone %s. Continuing ...\n", outSymbol.c_str());
    }

    // break;
  }

  return wfst;
}

WFSTransducerPtr buildHiddenMarkovModelTransducer(LexiconPtr& inputLexicon, LexiconPtr& outputLexicon,
						  const DistribTreePtr& distribTree, unsigned noEnd,
						  const String& sil, const String& eps, const String& end)
{
  HiddenMarkovModelTransducer hmmBuilder(sil, eps, end);
  return hmmBuilder.build(inputLexicon, outputLexicon, distribTree, noEnd);
}


// ----- methods for class `WFSTCombinedHC' -----
//
WFSTCombinedHC::WFSTCombinedHC(LexiconPtr& distribLexicon, DistribTreePtr& distribTree,
			       LexiconPtr& stateLexicon, LexiconPtr& phoneLexicon,
			       unsigned endN, const String& sil, const String& eps, 
			       const String& end, const String& sos, const String& eos, const String& backoff,
			       const String& pad, bool correct, unsigned hashKeys, bool approximateMatch, bool dynamic)
  : WFSTSortedInput(stateLexicon, distribLexicon, phoneLexicon, dynamic, String("WFST Combined HC")),
    _contextLen(distribTree->contextLength()),  _silInput(sil+"-m"),
    _sil(sil), _eps(eps), _end(end), _sos(sos), _eos(eos), _backoff(backoff), _pad(pad),
    _distribLexicon(distribLexicon), _phoneLexiconBitMatrix(_extractPhoneLexicon(phoneLexicon)),
    _endN(endN), _correct(correct), _hashKeys(hashKeys), _approximateMatch(approximateMatch),
    _nodeCount(1), _finalNode(NULL)
{
  if (_hashKeys > (sizeof(Primes) / sizeof(Primes[0])) )
    throw jconsistency_error("Error: Inappropriate number of hash keys.");

  _validStateSequences.clear();
  _stateSequenceList.clear();
  _listS.clear();

  cout << "Constructing bit matrices array from the distribution tree ...";
  _leafBitmaps = distribTree->buildMatrices(_phoneLexiconBitMatrix);
  cout << "Done." << endl;

  _enumStateSequences();
  _enumSilenceStates();
  _calcBitMasks();

  // the HC transducer has only one final node
  _addFinal(_nodeCount);
  _finalNode = find(_nodeCount++, false); 

  assert(!_finalNode.isNull());

  // don't expand the final node
  _finalNode->_expanded = true;
}

void WFSTCombinedHC::_unique(unsigned count)
{
  cout << "Purging unused edge lists." << endl;
  // loop over nodes to purge "old" edges
  for (_NodeVectorIterator itr = _nodes.begin(); itr != _nodes.end(); itr++) {
    NodePtr node(Cast<NodePtr>(*itr));
    if (node.isNull()) continue;

    if (_findCount - node->_lastEdgesCall < count) continue;
    
    if ((node->_type == End) && (node->_edges().isNull() == false) && node->canPurge()) {
      node->_expanded = false;
      node->_edges() = NULL;
    }
  }


#if 0
  // loop again to purge "unique" nodes
  unsigned active = 0;
  unsigned purged = 0;
  for (_NodeVectorIterator itr = _nodes.begin(); itr != _nodes.end(); itr++) {
    NodePtr& node(Cast<NodePtr>(*itr));
    if (node.isNull()) continue;

    if (node.unique() == false) {
      if (node != whiteNode() && node != grayNode() && node != blackNode()) active++;
      continue;
    }

    if (node->color() == White)
      node = whiteNode();
    else if (node->color() == Gray)
      node = grayNode();
    else
      node = blackNode();

    purged++;
  }

  /*
  printf("State lexicon %s : Input lexicon %s : Output lexicon %s\n",
	 stateLexicon()->name().c_str(), inputLexicon()->name().c_str(), outputLexicon()->name().c_str());
  */

  printf("%15s : %8d : %d Purged Nodes : %d Active Nodes\n",
	 name().c_str(), _findCount, purged, active);  fflush(stdout);
#endif
}

String WFSTCombinedHC::_Metastate::symbols() const
{
  static char buffer[100];

  sprintf(buffer, "%d ", _outputSymbol);

  String ret(buffer);
  for (_SymbolListConstIterator sitr = _symbolList.begin(); sitr != _symbolList.end(); sitr++) {
    sprintf(buffer, "%d ", *sitr);
    ret += String(buffer);
  }

  return ret;
}

String WFSTCombinedHC::_Metastate::symbols(const LexiconPtr& phoneLexicon, const LexiconPtr& distribLexicon) const
{
  static char buffer[500];

  sprintf(buffer, "%s ", phoneLexicon->symbol(_outputSymbol).c_str());

  String ret(buffer);
  for (_SymbolListConstIterator sitr = _symbolList.begin(); sitr != _symbolList.end(); sitr++) {
    sprintf(buffer, "%s ", distribLexicon->symbol(*sitr).c_str());
    ret += String(buffer);
  }

  return ret;
}

void WFSTCombinedHC::_Metastate::
dump(const LexiconPtr& phoneLexicon, const LexiconPtr& phoneLexiconBitMatrix, const LexiconPtr& distribLexicon) const
{
  String sym(symbols(phoneLexicon, distribLexicon));
  cout << "Metastate: " << sym << endl;
  _bitMatrixList->dump(phoneLexiconBitMatrix);
}

WFSTCombinedHC::_BitMatrixListPtr WFSTCombinedHC::_reduce(const _BitMatrixListPtr& src, const _BitMatrix& mask)
{
  _BitMatrixListPtr listL(new _BitMatrixList);
  for (_BitMatrixList::ConstIterator fbm(src); fbm.more(); fbm++) {
    _BitMatrix newbm(*fbm & mask);
    bool insert = true;
    for (_BitMatrixList::Iterator ibm(listL); ibm.more(); ibm++)
      if (*ibm == newbm) { insert = false; break; }
    if (insert)
      listL->add(newbm);
  }

  return listL;
}

void WFSTCombinedHC::_enumSilenceStates(unsigned statesN)
{
  unsigned input  = _distribLexicon->index(_sil + "-m");
  unsigned output = outputLexicon()->index(_sil);
  
  static Weight logOneHalf(log(2.0));

  // alter the silence bitmatrix
  _BitMatrix silbm(_phoneLexiconBitMatrix->size(), 2*_contextLen+1);
  for (Lexicon::Iterator phonesiter(_phoneLexiconBitMatrix); phonesiter.more(); phonesiter++) {
    const String& phone(*phonesiter);

    if (phone == _sil || phone == _sos || phone == _eos || phone == _backoff) continue;

    silbm.resetAt(_phoneLexiconBitMatrix->index(phone), _contextLen);
    if (phone.find("WB_B") != string::npos)
      silbm.resetAt(_phoneLexiconBitMatrix->index(phone), _contextLen - 1);
    if (phone.find("WB_E") != string::npos)
      silbm.resetAt(_phoneLexiconBitMatrix->index(phone), _contextLen + 1);
    if (phone.find("WB_I") != string::npos) {
      silbm.resetAt(_phoneLexiconBitMatrix->index(phone), _contextLen - 1);
      silbm.resetAt(_phoneLexiconBitMatrix->index(phone), _contextLen + 1);
    }
  }
  silbm.setAt(_phoneLexiconBitMatrix->index(_pad), _contextLen + 1);

  cout << "Silence bit matrix: " << endl;
  silbm.dump(_phoneLexiconBitMatrix);

  _BitMatrixListPtr silbmlist(new _BitMatrixList(silbm));

  cout << "Silence bit matrix list: " << endl;
  silbmlist->dump(_phoneLexiconBitMatrix);

  // enumerate the list of input symbols of the silence metastate
  _SymbolList slist;
  for (unsigned stateX = 0; stateX < statesN; stateX++)
    slist.push_back(input);

  // create the silence metastate
  _MetastatePtr mstate(new _Metastate(NULL, NULL, output, slist, silbmlist));

  NodePtr beginNode(NULL);
  NodePtr node(NULL);

  // create the silence nodes in the transducer
  bool create = true;
  for (unsigned stateX = 0; stateX < statesN; stateX++) {

    NodePtr nextNode(find(_nodeCount++, create));
    nextNode->_metastate = mstate;

    if (beginNode.isNull()) {
      beginNode = nextNode;
    } else {
      EdgePtr edge(new Edge(node, nextNode, input, /* output= */ 0, logOneHalf));
      node->_addEdgeForce(edge);
      node->_expanded = true;
    }

    EdgePtr selfLoop(new Edge(nextNode, nextNode, input, /* output= */ 0, logOneHalf));
    nextNode->_addEdgeForce(selfLoop);

    // this is for compatibility with the old HC
    if (stateX == 0) {
      EdgePtr selfLoop(new Edge(nextNode, nextNode, input, output, logOneHalf));
      nextNode->_addEdgeForce(selfLoop);
    }

    node = nextNode;
  }

  mstate->_beginNode = beginNode;
  mstate->_endNode   = node;

  // the end state of the silence metastate has to be expanded
  node->_expanded = false;
  node->_type = End;

  // add a copy of the silence metastate for further expansion
  _MetastatePtr mstateCopy(new _Metastate(mstate->beginNode(),
					  mstate->endNode(),
					  mstate->outputSymbol(),
					  mstate->symbolList(),
					  mstate->bitMatrixList()));

  _BitMatrixListPtr bmlistCopy(new _BitMatrixList(*mstate->bitMatrixList()));
  _silCopy = mstateCopy;
  _silCopy->_bitMatrixList = bmlistCopy;

  _listS.insert(_MetastateSetValueType(output, mstateCopy));
  _mapS.insert(_MetastateSingleMapValueType(mstate->symbols(), mstateCopy));
  _listT.insert(_MetastateMapValueType(mstate->symbols(), mstateCopy));
}


bool WFSTCombinedHC::_checkBitMatrix(const _BitMatrix& bm, const String& phone)
{
  return bm.getAt(_phoneLexiconBitMatrix->index(phone), _contextLen);
}

bool WFSTCombinedHC::_checkBitMatrixList(const _BitMatrixListPtr& bmlist, const String& phone)
{
  for (_BitMatrixList::ConstIterator bmitr(bmlist); bmitr.more(); bmitr++)
    if (_checkBitMatrix(*bmitr, phone) == false) return false;
  return true;
}

// construct a new phone lexicon without sos, eos, end, and backoff symbols
// for indexing into the bit matrices
LexiconPtr WFSTCombinedHC::_extractPhoneLexicon(LexiconPtr& phoneLex)
{
  LexiconPtr newLex(new Lexicon("New Phone Lexicon"));
  for (Lexicon::Iterator pitr(phoneLex); pitr.more(); pitr++) {
    const String& phone(*pitr);
    if (phone.find(_end) == String::npos &&
	phone.find(_sos) == String::npos &&
	phone.find(_eos) == String::npos &&
	phone.find(_backoff) == String::npos &&
	phone.find(_eps) == String::npos) {
       newLex->index(phone, true);
     }
   }
   return newLex;
 }

 void WFSTCombinedHC::_enumStateSequences()
 {
   _firstPass();
   _secondPassEx();
 }

 void WFSTCombinedHC::_firstPass()
 {
   unsigned sequenceCount = 0;
   for (Lexicon::Iterator pitr(_phoneLexiconBitMatrix); pitr.more(); pitr++) {
     const String& phone(*pitr);

     if (phone.find(_pad) != string::npos) continue;

     String decisionPhone(DistribTree::getDecisionPhone(phone));

     for (Lexicon::Iterator distribitr(_distribLexicon); distribitr.more(); distribitr++) {
       //if ((*distribitr).find(decisionPhone + "-b") != 0) continue;
       if ((*distribitr).find(decisionPhone + "-0") != 0) continue;

       // --fix for a smaller phoneLexicon
       DistribTree::_BitmapListConstIterator bmitr = _leafBitmaps.find(*distribitr);
       if (bmitr == _leafBitmaps.end()) continue;
       // --end fix

       _BitMatrixList bmlist1(_leafBitmaps[*distribitr]);

       for (Lexicon::Iterator distribitr2(_distribLexicon); distribitr2.more(); distribitr2++) {
	 // if ((*distribitr2).find(decisionPhone + "-m") != 0) continue;
	 if ((*distribitr2).find(decisionPhone + "-1") != 0) continue;

	 // --fix for a smaller phoneLexicon
	 DistribTree::_BitmapListConstIterator bmitr = _leafBitmaps.find(*distribitr2);
	 if (bmitr == _leafBitmaps.end()) {
	   //cout << "Found no leaf for " << *distribitr2 << endl;
	   continue;
	 }
	 // --end fix

	 _BitMatrixList bmlist2(_leafBitmaps[*distribitr2]);

	 /*
	 cout << "Phone: " << phone << " Decision phone: " << decisionPhone << endl;
	 cout << "Testing subsequence: " << *distribitr << " " << *distribitr2 << endl;
	 cout << "Bit map list for " << *distribitr << endl;
	 for (_BitMatrixListIterator itr1 = bmlist1.begin(); itr1 != bmlist1.end(); itr1++)
	   itr1->dump();
	 cout << "Bit map list for "  << *distribitr2 << endl;
	 for (_BitMatrixListIterator itr1 = bmlist2.begin(); itr1 != bmlist2.end(); itr1++)
	   itr1->dump();
	 */

	 _BitMatrixList bmlist;
	 for (_BitMatrixList::Iterator itr1(bmlist1); itr1.more(); itr1++) {
	   if (_checkBitMatrix(*itr1, phone) == false) continue;
	   for (_BitMatrixList::Iterator itr2(bmlist2); itr2.more(); itr2++) {
	     if (_checkBitMatrix(*itr2, phone) == false) continue;
	     _BitMatrix tempbm(*itr1 & *itr2);

	     if (tempbm.isValid() == false) { /* cout << "1st pass: bm is not valid ... continuing" << endl; */ continue; }
	     // TODO: find out why there are so many repeating bitmaps
	     bool found = false;
	     for (_BitMatrixList::Iterator bmitr(bmlist); bmitr.more(); bmitr++)
	       if (*bmitr == tempbm) { found = true; break; }

	     if (!found) {
	       bmlist.add(tempbm);
	     }
	   }
	 }

	 if (bmlist.isValid() == false) continue;

	 /*
	 cout << "Valid subsequence: " << *distribitr << " " << *distribitr2 << " Phone: " << phone << endl;
	 cout << "Bitmap list: " << endl;
	 for (_BitMatrixListIterator bmitr = bmlist.begin(); bmitr != bmlist.end(); bmitr++)
	 bmitr->dump(_phoneLexiconBitMatrix);
	 */

	 seqrec srTmp(*distribitr, *distribitr2, bmlist);
	 if (_validStateSequences.find(phone) == _validStateSequences.end()) {
	   //cout << "Adding phone " << phone << endl;
	   ldsb ldsbTmp;
	   ldsbTmp.push_back(srTmp);
	   _validStateSequences.insert(ldsbpValueType(phone, ldsbTmp));
	 } else {
	   _validStateSequences[phone].push_back(srTmp);
	 }
	 sequenceCount++;
       }
     }
   }

   cout << "Number of phones after the first pass: "                << _validStateSequences.size() << endl;
   cout << "Number of valid state sequences after the first pass: " << sequenceCount               << endl;

   if  (_validStateSequences.empty())
     throw jconsistency_error("Exception: No valid subsequences found!");
 }

 void WFSTCombinedHC::_secondPassEx()
 {
   for (ldsbpIterator validitr = _validStateSequences.begin(); validitr != _validStateSequences.end(); ++validitr){
     const String& phone((*validitr).first);
     ldsb&         phoneSequenceList((*validitr).second);

     unsigned finalPhoneX = ((phone.find("WB") == string::npos) ? 1 : 2);
     static char buffer[100];
     sprintf(buffer, "%d", finalPhoneX);
     String mark(_end + String(buffer));
     unsigned markX = _distribLexicon->index(mark);
     String decisionPhone(DistribTree::getDecisionPhone(phone));

     unsigned validStateSequencesN = 0;
     for (ldsbIterator pseqiter = phoneSequenceList.begin(); pseqiter != phoneSequenceList.end(); pseqiter++) {
       for (Lexicon::Iterator distribitr(_distribLexicon); distribitr.more(); distribitr++) {
	 // if ((*distribitr).find(decisionPhone + "-e") != 0) continue;
	 if ((*distribitr).find(decisionPhone + "-2") != 0) continue;

	 // --fix for a smaller phoneLexicon
	 DistribTree::_BitmapListConstIterator bmitr = _leafBitmaps.find(*distribitr);
	 if (bmitr == _leafBitmaps.end()) continue;
	 // --end fix

	 _BitMatrixList bmlist2(_leafBitmaps[*distribitr]);

	 seqrec currec(*pseqiter);
	 _BitMatrixList bmlist1(currec.bmlist);
	 _BitMatrixList newbmlist;
	 for (_BitMatrixList::Iterator itr1(bmlist1); itr1.more(); itr1++) {
	   for (_BitMatrixList::Iterator itr2(bmlist2); itr2.more(); itr2++) {
	     if (_checkBitMatrix(*itr2, phone) == false) continue;
	     _BitMatrix newbm(*itr1 & *itr2);
	     if (newbm.isValid() == false) {  /* cout << "2nd pass: bm is not valid ... continuing" << endl;*/  continue; }
	     bool found = false;
	     for (_BitMatrixList::Iterator bmitr(newbmlist); bmitr.more(); bmitr++) {
	       if (*bmitr == newbm) { found = true; break; }
	     }
	     if (found == false) {
	       newbm.resetColumn(_contextLen);
	       newbm.setAt(_phoneLexiconBitMatrix->index(phone), _contextLen);
	       newbmlist.add(newbm);
	     }
	   }
	 }

	 if (newbmlist.isValid() == false) continue;

	 validStateSequencesN++;
	 currec.bmlist = newbmlist;
	 currec.seq.push_back(*distribitr);

	 /*
	 cout << "Valid phone " << phone << " : Valid sequence:";
	 for (dsIterator titer = currec.seq.begin(); titer != currec.seq.end(); titer++)
	   cout << " " << *titer;
	 cout << "Bitmap list: " << endl;
	 for (_BitMatrixListIterator itr = currec.bmlist.begin(); itr != currec.bmlist.end(); itr++) {
	     itr->dump(_phoneLexiconBitMatrix);
	 }
	 */

	 dsIterator dsiter = currec.seq.begin();

	 _SymbolList slist; 

	 // write the state transition in the transducer
	 static Weight logOneHalf(log(2.0));

	 dsIterator itr = currec.seq.begin();

	 // connect begin state
	 unsigned begInput   = _distribLexicon->index(*itr);
	 unsigned begOutput  = outputLexicon()->index(phone);

	 slist.push_back(begInput);
	 itr++;

	 unsigned midInput = _distribLexicon->index(*itr); 
	 slist.push_back(midInput);

	 itr++;
	 unsigned endInput = _distribLexicon->index(*itr);
	 slist.push_back(endInput);

	 _BitMatrixListPtr newlist(new _BitMatrixList(newbmlist));

	 _MetastatePtr mstate(new _Metastate(NULL, NULL, begOutput, slist, newlist));
	 _listS.insert(_MetastateSetValueType(begOutput, mstate));
	 _mapS.insert(_MetastateSingleMapValueType(mstate->symbols(), mstate));	
       }
     }
     cout << "Phone " << phone << " has " << validStateSequencesN << " valid state sequences" << endl;
   }

   cout << "Number of valid state sequences after the second pass: " << _listS.size() << endl;

   if (_listS.empty())
     throw jconsistency_error("Exception: No valid metastates!");
 }

set<unsigned> WFSTCombinedHC::_findCenterPhones(const _BitMatrixList& bmatrixlist)
{
  set<unsigned> phoneSet;
  for (_BitMatrixList::ConstIterator fbm(bmatrixlist); fbm.more(); fbm++) {
    const _BitMatrix frombm(*fbm);    

    for (unsigned rowX = 0; rowX < frombm.rowsN(); rowX++)
      if (frombm.getAt(rowX, _contextLen))
	phoneSet.insert(rowX);
  }
  return phoneSet;
}

 void WFSTCombinedHC::_calcBitMasks()
 {
   cout << "Calclating bit masks and possible connections for " << _listS.size() << " metastates." << endl;
   unsigned s1X = 0;
   for (_MetastateSetIterator sitr = _listS.begin(); sitr != _listS.end(); sitr++) {
     _MetastatePtr& s1((*sitr).second);

     const _BitMatrixListPtr src(s1->bitMatrixList());

     _BitMatrixList frombml(*src);  frombml >>= 1;

     // cout << "Source bit matrix list:" << endl;
     // (*src).dump(_phoneLexiconBitMatrix);

     // cout << "From bit matrix list:" << endl;
     // frombml.dump(_phoneLexiconBitMatrix);

     _BitMatrix allZero(_phoneLexiconBitMatrix->size(), 2*_contextLen+1);
     for (unsigned colX = 0; colX < allZero.columnsN(); colX++)
       allZero.resetColumn(colX);

     set<unsigned> phoneSet(_findCenterPhones(frombml));

     unsigned s2count = 0;
     for (set<unsigned>::iterator pitr = phoneSet.begin(); pitr != phoneSet.end(); pitr++) {

       // there is no metastate for the PAD phone, but if the connection to it is allowed
       // by the metastate's bit matrix list, it should be allowed by the corresponding
       // bit mask as well.
       if (*pitr == _phoneLexiconBitMatrix->index(_pad)) {
	 allZero.setAt(_phoneLexiconBitMatrix->index(_pad), _contextLen);
	 continue;
       }
       unsigned int phoneIndex = outputLexicon()->index(_phoneLexiconBitMatrix->symbol(*pitr));
       pair<_MetastateSetIterator, _MetastateSetIterator> range = _listS.equal_range(phoneIndex);

       for (_MetastateSetIterator ritr = range.first; ritr != range.second; ritr++) {

	 _MetastatePtr& s2((*ritr).second);
	 const _BitMatrixListPtr& tobmlistP(s2->bitMatrixList());
	 const _BitMatrixList& tobmlist(*tobmlistP);

	 // cout << "To bit matrix list:" << endl;
	 // frombml.dump(_phoneLexiconBitMatrix);

	 if (connect(frombml, tobmlist) == false) continue;

	 allZero |= tobmlist;
	 s1->metastateList().push_back(s2);
	 s2count++;
       }
     }
     _BitMatrix shiftbm(allZero << 1);
     shiftbm.resetColumn(/* column= */ 0);

     // cout << "Bit mask:" << endl;
     // shiftbm.dump(_phoneLexiconBitMatrix);

     // if (shiftbm.isValid() == false) {
     //   cout << "Bit matrix:" << endl;
     //   shiftbm.dump(_phoneLexiconBitMatrix);
     //   throw j_error("Bit mask is not valid.");
     // }

     s1->setBitMask(shiftbm);
     // set the bit mask of the silence metastate copy
     if (outputLexicon()->symbol(s1->outputSymbol()) == _sil) {
       _silCopy->_metastateList = s1->_metastateList;
       _silCopy->_mask = s1->_mask;
     }
     cout << "Metastate " << s1X << " has " << s2count << " possible connections." << endl;
     s1X++;
   }
}

WFSTCombinedHC::NodePtr& WFSTCombinedHC::initial(int idx)
{
  if (_initial.isNull()) 
    _initial = _newNode( (idx >= 0) ? idx : _nodeCount++, Unknown);
  return Cast<NodePtr>(_initial);
}

const WFSTCombinedHC::EdgePtr& WFSTCombinedHC::edges(WFSAcceptor::NodePtr& nd)
{
  static unsigned cnt = 0;

  if (++cnt % 2000 == 0)
    cout << "Nodes: " << _nodeCount << "; listT size: " << _listT.size() << endl;

  NodePtr node(Cast<NodePtr>(nd));

  bool create = true;
  static Weight logOneThird(log(3.0));

  node->_lastEdgesCall = _findCount;

  // if the adjaceny list is already expanded, simply return it
  if (node->_expanded)
    return node->_edges();

  // expand adjacency list of this node
  _expandNode(node);

  if (_approximateMatch) {
    _BitMatrixListPtr& bmlist(node->_metastate->_bitMatrixList);
    node->_metastate->_hash = _hash(bmlist);

    bmlist->clear();
    node->_metastate->_bitMatrixList = NULL;
  }

  return node->_edges();
}

void WFSTCombinedHC::_expandNode(WFSTCombinedHC::NodePtr& node)
{
  if (node->_type != End)
    throw jconsistency_error("Error: requested expansion of a non-end node.");

  _MetastatePtr q(node->_metastate);
  const _BitMatrixListPtr& fbmlist(q->bitMatrixList());
  _BitMatrixList frombmlist(*fbmlist >> 1);

  // cout << "Metastate q:" << endl;
  // q->dump(outputLexicon(), _phoneLexiconBitMatrix, _distribLexicon);

  // cout << "From bit matrix list:" << endl;
  // frombmlist.dump(_phoneLexiconBitMatrix);

  static Weight logOneHalf(log(2.0));

  // Add self loops, which were deleted by the memory recovery mechanism
  if (outputLexicon()->symbol(q->outputSymbol()) != _sil)
    _addSelfLoops(node);

  EdgePtr selfLoop(new Edge(node, node, q->endSymbol(), 0, logOneHalf));
  node->_addEdgeForce(selfLoop);
 
  assert(fbmlist.isNull() == false);
  if ((outputLexicon()->symbol(node->_metastate->outputSymbol()) == _sil))
    _connectToFinal(node);

  _MetastatePtr& fromS((*(_mapS.find(q->symbols()))).second);
  bool found = false;

  for (_MetastateListConstIterator slitr = fromS->metastateList().begin(); slitr != fromS->metastateList().end(); slitr++) {
    const _MetastatePtr& s(*slitr);
    const _BitMatrixListPtr& tobmlist(s->bitMatrixList());

    // s->dump(outputLexicon(), _phoneLexiconBitMatrix, _distribLexicon);

    _BitMatrixList bmatrixlist(frombmlist & *tobmlist);
    _BitMatrixListPtr listL(new _BitMatrixList(bmatrixlist));

    // cout << "ListL bit matrix list:" << endl;
    // listL->dump(_phoneLexiconBitMatrix);

    if (listL->isValid() == false) continue;
    
    // cout << "Bit mask:" << endl;
    // s->bitMask().dump(_phoneLexiconBitMatrix);

    _BitMatrixListPtr listLprime(_reduce(listL, s->bitMask()));

    // cout << "Lprime bit matrix:" << endl;
    // listLprime->dump(_phoneLexiconBitMatrix);  fflush(stdout);

    _MetastatePtr sPrime(_findMetastate(s, listLprime));

    unsigned markX = _distribLexicon->index(outputLexicon()->symbol(s->outputSymbol()), /* create= */ true);
    NodePtr sPrimeBeginNode(sPrime->beginNode());
    NodePtr qEndNode(node);
    EdgePtr edge(new Edge(qEndNode, sPrimeBeginNode, markX, sPrime->outputSymbol(), logOneHalf));
    qEndNode->_addEdgeForce(edge);
  }
  
  node->_expanded = true;
}

void WFSTCombinedHC::_addSelfLoops(NodePtr& startNode)
{
  bool create = true;
  Weight logOneHalf(log(2.0));
  for (unsigned endX = 0; endX <= _endN; endX++) {
    char buffer[100];
    String mark;
    if (endX == 0) {
      mark = _backoff;
    } else {
      sprintf(buffer, "%d", endX);
      mark = _end + String(buffer);
    }

    unsigned in  = inputLexicon()->index(mark, create);
    unsigned out = outputLexicon()->index(mark, create);

    // self-loop for initial state
    EdgePtr selfLoop(new Edge(startNode, startNode, in, out, logOneHalf));
    startNode->_addEdgeForce(selfLoop);
  }
}

void WFSTCombinedHC::_connectToFinal(NodePtr& node)
{
  // check if the "PAD" phone is allowed in the +1 context
  bool connectsToEnd = true;
  static Weight logOneHalf(log(2.0));

  for (_BitMatrixList::Iterator silbmitr(node->_metastate->bitMatrixList()); silbmitr.more(); silbmitr++) {
    if ((*silbmitr).getAt(_phoneLexiconBitMatrix->index(_pad), _contextLen + 1) == false) {
      connectsToEnd = false;
      break;
    }
  }
    
  // connect the silence metastate to the final state of the transducer
  if (connectsToEnd) {
    unsigned output = outputLexicon()->index(_eos);
    EdgePtr edgeToFinal(new Edge(node, _finalNode, /* input= */ 0, output, logOneHalf));
    node->_addEdgeForce(edgeToFinal);
  }
}

vector<unsigned> WFSTCombinedHC::_calcRunLengths(const _BitMatrixListPtr& listL)
{
  vector<unsigned> rlen;

  for (_BitMatrixList::ConstIterator itr(listL); itr.more(); itr++) {
    const _BitMatrix& bmatrix(*itr);
    for (unsigned colX = 0; colX < bmatrix.columnsN(); colX++) {
      unsigned run = 0;
      for (unsigned rowX = 0; rowX < bmatrix.rowsN(); rowX++) {
	run++;
	if (bmatrix.getAt(rowX, colX)) {
	  rlen.push_back(run);  run = 0;
	}
      }
    }
  }

  //sort(rlen.begin(), rlen.end());
  return rlen;
}

vector<unsigned> WFSTCombinedHC::_hash(const _BitMatrixListPtr& listL)
{
  vector<unsigned> runLengths(_calcRunLengths(listL));

  vector<unsigned> hValues(_hashKeys, 0);
  for (unsigned j = 0; j<_hashKeys; j++)
    for (unsigned i = 0; i < runLengths.size(); i++) 
      hValues[j] = Primes[j] * hValues[j] + runLengths[i];
  return hValues;
}

WFSTCombinedHC::_MetastatePtr WFSTCombinedHC::
_findMetastate(const _MetastatePtr& s, const _BitMatrixListPtr& listL)
{
  ++_findCount;

  static Weight logOneHalf(log(2.0));

  _MetastatePtr sPrime(NULL);
  // look for an "exact match"
  pair<_MetastateMapIterator, _MetastateMapIterator> range = _listT.equal_range(s->symbols());
  for (_MetastateMapIterator rangeitr = range.first; rangeitr != range.second; ++rangeitr) {
    _MetastatePtr& metaState((*rangeitr).second);
    _BitMatrixListPtr& bmlist(metaState->bitMatrixList());

    if (_correct == false || (bmlist.isNull() && metaState->hash() == _hash(listL))) {
      if (metaState->endNode()->_expanded == false) bmlist = listL;
      return metaState;
    }
    if (bmlist.isNull() == false && *listL == *bmlist) return metaState; // Schuster hack 2 - _correct = true: correct HC, _correct = false: S&H HC
  }

  // match not found, create a new meta-state
  sPrime = _createMetastate(s, listL);
  _listT.insert(_MetastateMapValueType(sPrime->symbols(), sPrime));

  return sPrime;
}

WFSTCombinedHC::_MetastatePtr WFSTCombinedHC::_createMetastate(const _MetastatePtr& s, const _BitMatrixListPtr& listL)
{
  bool create = true;
  static Weight logOneHalf(log(2.0));

  NodePtr determinizeNode(find(_nodeCount++, create));
  NodePtr fromNode(determinizeNode);
  for (_SymbolListConstIterator outputitr = s->symbolList().begin(); outputitr != s->symbolList().end(); outputitr++) {
    unsigned symbol = *outputitr;

    NodePtr toNode(find(_nodeCount++, true));
    WFSTSortedInput::EdgePtr edge;
    edge = new Edge(fromNode, toNode, symbol, 0, logOneHalf);
    fromNode->_addEdgeForce(edge);

    EdgePtr selfLoop(new Edge(toNode, toNode, symbol, 0, logOneHalf));
    toNode->_addEdgeForce(selfLoop);

    fromNode->_expanded = true;
    fromNode->_type = Unknown;
    fromNode->_metastate = NULL;

    fromNode = toNode;
  }

  // add word end markers
  if (outputLexicon()->symbol(s->outputSymbol()) != _sil)
    _addSelfLoops(fromNode);
    
  _MetastatePtr sPrime(new _Metastate(determinizeNode, fromNode, s->outputSymbol(), s->symbolList(), listL));

  // the end node should be expanded
  fromNode->_expanded = false;
  fromNode->_type = End;
  fromNode->_metastate = sPrime;

  return sPrime;
}

WFSAcceptor::Node* WFSTCombinedHC::
_newNode(const unsigned state, const NodeType& type, const _MetastatePtr& metastate)
{
  return new Node(state, type, metastate);
}

WFSAcceptor::Edge* WFSTCombinedHC::
_newEdge(NodePtr& from, NodePtr& to, unsigned input, unsigned output, Weight cost)
{
  return new Edge(from, to, input, output, cost);
}


// ----- methods for class `WFSTCombinedHC::Node' -----
//
MemoryManager<WFSTCombinedHC::Node>& WFSTCombinedHC::Node::memoryManager()
{
  static MemoryManager<Node> _MemoryManager("WFSTCombinedHC::Node");
  return _MemoryManager;
}


// ----- methods for class `WFSTCombinedHC::Edge' -----
//
MemoryManager<WFSTCombinedHC::Edge>& WFSTCombinedHC::Edge::memoryManager()
{
  static MemoryManager<Edge> _MemoryManager("WFSTCombinedHC::Edge");
  return _MemoryManager;
}

WFSTDeterminizationPtr buildDeterminizedHC(LexiconPtr& distribLexicon, DistribTreePtr& distribTree, LexiconPtr& phoneLexicon,
					   unsigned endN, const String& sil, const String& eps, const String& end,
					   const String& sos, const String& eos, const String& backoff, const String& pad, bool correct,
					   unsigned hashKeys, bool approximateMatch, bool dynamic, unsigned count)
{
  LexiconPtr stateLexicon(new Lexicon("HC State Lexicon"));
  WFSTCombinedHCPtr combinedHC(new WFSTCombinedHC(distribLexicon, distribTree, stateLexicon, phoneLexicon, endN, sil,
						  eps, end, sos, eos, backoff, pad, correct, hashKeys, approximateMatch, dynamic));
  return determinize(combinedHC, /* semiring */ "Tropical", count);
}

WFSTransducerPtr buildHC(LexiconPtr& distribLexicon, DistribTreePtr& distribTree, LexiconPtr& phoneLexicon,
			 unsigned endN, const String& sil, const String& eps, const String& end,
			 const String& sos, const String& eos, const String& backoff, const String& pad, bool correct,
			 unsigned hashKeys, bool approximateMatch, bool dynamic)
{
  LexiconPtr stateLexicon(new Lexicon("HC State Lexicon"));
  WFSTCombinedHCPtr HC(new WFSTCombinedHC(distribLexicon, distribTree, stateLexicon, phoneLexicon, endN, sil, eps, end,
					  sos, eos, backoff, pad, correct, hashKeys, approximateMatch, dynamic));

  breadthFirstSearch(HC);
  return HC;
}

// ----- methods for class `WFSTAddSelfLoops' -----
//

WFSTAddSelfLoops::WFSTAddSelfLoops(WFSTSortedInputPtr& A, const String& end, unsigned endN, bool dynamic, const String& name)
  : WFSTSortedInput(A->stateLexicon(), A->inputLexicon(), A->outputLexicon(), dynamic, name), _end(end), _endN(endN), _A(A) { }

WFSTAddSelfLoops::NodePtr& WFSTAddSelfLoops::initial(int idx)
{
  if (_initial.isNull()) _initial = new Node(_A->initial());

  return Cast<NodePtr>(_initial);
}

WFSAcceptor::Node* WFSTAddSelfLoops::_newNode(unsigned state)
{
  return new Node(state);
}

WFSAcceptor::Edge* WFSTAddSelfLoops::
_newEdge(NodePtr& from, NodePtr& to, unsigned input, unsigned output, Weight cost)
{
  return new Edge(from, to, input, output, cost);
}

WFSTAddSelfLoops::NodePtr
WFSTAddSelfLoops::find(const WFSTSortedInput::NodePtr& nd, bool create)
{
  unsigned state = nd->index();

  if (initial()->index() == state)
    return initial();

  _NodeMapIterator itr = _final.find(state);

  if (itr != _final.end()) {
    assert((*itr).second->isFinal() == true);
    return Cast<NodePtr>((*itr).second);
  }

  if (state < _nodes.size() && _nodes[state].isNull() == false)
    return Cast<NodePtr>(_nodes[state]);

  if (create == false)
    throw jkey_error("No state %u exists.", state);

  NodePtr newNode(new Node(nd));
  if (nd->isFinal()) {
    newNode->_setCost(nd->cost());
    _final.insert(WFSAcceptor::_ValueType(state, newNode));
    itr = _final.find(state);
    return Cast<NodePtr>((*itr).second);
  } else {
    if (state >= _nodes.size()) _resize(state);
    _nodes[state] = newNode;
    return Cast<NodePtr>(_nodes[state]);
  }
}

void WFSTAddSelfLoops::_addSelfLoops(NodePtr& startNode)
{
  bool create = true;
  Weight logOneHalf(log(2.0));
  for (unsigned endX = 1; endX <= _endN; endX++) {
    char buffer[100];
    sprintf(buffer, "%d", endX);
    String mark(_end + String(buffer));

    unsigned in  = inputLexicon()->index(mark, create);
    unsigned out = outputLexicon()->index(mark, create);

    EdgePtr selfLoop(new Edge(startNode, startNode, in, out, logOneHalf));
    startNode->_addEdgeForce(selfLoop);
  }
}

const WFSTAddSelfLoops::EdgePtr& WFSTAddSelfLoops::edges(WFSAcceptor::NodePtr& nd)
{
  bool create = true;

  NodePtr node(Cast<NodePtr>(nd));

  /*
  printf("\nNode %d\n", node->index());
  if (node->_edges().isNull())
    printf("node->_edges() is Null\n");
  if (node->_nodeA.isNull())
    printf("node->_nodeA is Null\n");
  fflush(stdout);
  */

  // are edges already expanded?
  if (node->_edges().isNull() == false || node->_nodeA.isNull())
    return node->_edges();

  for (WFSTSortedInput::Node::Iterator itr(_A, Cast<WFSTSortedInput::NodePtr>(node->_nodeA)); itr.more(); itr++) {
    WFSTSortedInput::EdgePtr& edge(itr.edge());
    unsigned input  = edge->input();
    unsigned output = edge->output();
    Weight   c      = edge->cost();

    NodePtr nextNode(find(edge->next(), create)); 

    EdgePtr newEdge(new Edge(node, nextNode, input, output, c));
    node->_addEdgeForce(newEdge);
    
    EdgePtr selfLoop(new Edge(node, node, input, 0, c));
    node->_addEdgeForce(newEdge);

    _addSelfLoops(node);

  }

  return node->_edges();
}


// ----- methods for class `WFSTAddSelfLoops::Edge' -----
//
MemoryManager<WFSTAddSelfLoops::Edge>&  WFSTAddSelfLoops::Edge::memoryManager()
{
  static MemoryManager<Edge> _MemoryManager("WFSTAddSelfLoops::Edge");
  return _MemoryManager;
}


// ----- methods for class `WFSTAddSelfLoops::Node' -----
//
WFSTAddSelfLoops::Node::Node(unsigned idx, Color col, Weight cost)
  : WFSTSortedInput::Node(idx, col, cost) { }

WFSTAddSelfLoops::Node::Node(const WFSTSortedInput::NodePtr& nodeA, Color col, Weight cost)
  : WFSTSortedInput::Node(nodeA, col, cost)
{
  /*
  printf("Allocating node %d\n", nodeA->index());
  fflush(stdout);
  */
}

MemoryManager<WFSTAddSelfLoops::Node>&  WFSTAddSelfLoops::Node::memoryManager()
{
  static MemoryManager<Node> _MemoryManager("WFSTAddSelfLoops::Node");
  return _MemoryManager;
}



// ----- methods for class `MinimizeFSA' -----
//
MinimizeFSA::MinimizeFSA(const WFSAcceptorPtr& fsa)
  : _fsa(fsa), _reverse(new WFSAcceptor(_fsa->stateLexicon(), _fsa->inputLexicon()))
{
  _reverse->reverse(_fsa);
}

WFSAcceptorPtr MinimizeFSA::minimize()
{
  _initialize();
  _partition();

  return _connect();
}

void MinimizeFSA::_printBlocks()
{
  for (_BlockListIterator itr = _blocklist.begin(); itr != _blocklist.end(); itr++) {
    unsigned blockX((*itr).first);

    cout << "Block " << blockX << ":";
    const _Block& block((*itr).second);
    for (_BlockConstIterator bitr = block.begin(); bitr != block.end(); bitr++) {
      unsigned stateX(*bitr);
      cout << " " << stateX;
    }
    cout << endl;
  }
}

void MinimizeFSA::_initialize()
{
  _Block block0, block1;

  // initial state
  block0.insert(_fsa->initial()->index());
  _inblock.insert(_InBlockValueType(_fsa->initial()->index(), /* blockX= */ 0));

  // emitting states
  for (WFSAcceptor::_ConstNodeVectorIterator itr = _fsa->_nodes.begin(); itr != _fsa->_nodes.end(); itr++) {
    NodePtr node(*itr);
    if (node.isNull()) continue;
    block0.insert(node->index());
    _inblock.insert(_InBlockValueType(node->index(), /* blockX= */ 0));
  }

  // final states
  for (WFSAcceptor::_ConstNodeMapIterator itr = _fsa->_final.begin(); itr != _fsa->_final.end(); itr++) {
    NodePtr node((*itr).second);
    block1.insert(node->index());
    _inblock.insert(_InBlockValueType(node->index(), /* blockX= */ 1));
  }

  _blocklist.insert(_BlockListValueType(/* blockX= */ 0, block0));  _updateWaiting(/* blockX= */ 0, block0);
  _blocklist.insert(_BlockListValueType(/* blockX= */ 1, block1));  _updateWaiting(/* blockX= */ 1, block1);
}

void MinimizeFSA::_partition()
{
  unsigned cnt = 0;
  while (_waiting.more()) {
    if (++cnt % 10000 == 0) { printf("Still %d waiting\n", _waiting.still()); fflush(stdout); }

    _WaitKey wkey(_waiting.pop());
    unsigned symbolX(wkey.symbol());

    _Block inverse(_inverse(wkey));
    _Block jlist(_jlist(inverse));

    for (_BlockIterator itr = jlist.begin(); itr != jlist.end(); itr++) {
      unsigned j(*itr);

      _Block& blockj(_blocklist[j]);
      _Block blockq;
      unsigned q(_blocklist.size());

      for (_BlockIterator iitr = inverse.begin(); iitr != inverse.end(); iitr++) {
	unsigned stateX(*iitr);
	if (_inblock[stateX] == j) {
	  blockj.erase(stateX);
	  blockq.insert(stateX);
	  _inblock[stateX] = q;

	  assert(blockj.find(stateX) == blockj.end());
	}
      }
      _blocklist.insert(_BlockListValueType(q, blockq));
      _updateWaiting(j, blockj, q, blockq);
    }
  }
}

MinimizeFSA::_Block MinimizeFSA::_inverse(const _WaitKey& key)
{
  unsigned i(key.block());
  unsigned symbolX(key.symbol());
  _Block& blocki(_blocklist[i]);

  _Block inv;
  for (_BlockIterator itr = blocki.begin(); itr != blocki.end(); itr++) {
    unsigned stateX(*itr);
    NodePtr node(_reverse->find(stateX));

    for (NodeIterator nitr(node); nitr.more(); nitr++) {
      EdgePtr edge(nitr.edge());
      if (edge->input() == symbolX)
	inv.insert(edge->next()->index());
    }
  }

  return inv;
}

MinimizeFSA::_Block MinimizeFSA::_jlist(const _Block& inverse)
{
  _InBlock jblk;
  for (_BlockIterator itr = inverse.begin(); itr != inverse.end(); itr++) {
    unsigned j(_inblock[*itr]);
    unsigned jsize = _blocklist[j].size();

    _InBlockIterator jitr = jblk.find(j);
    if (jitr == jblk.end()) {
      jblk.insert(_InBlockValueType(j, /* count= */ 0));
    }

    jblk[j]++;
    if (jblk[j] == jsize)
      jblk.erase(j);
  }

  _Block blk;
  for (_InBlockIterator itr = jblk.begin(); itr != jblk.end(); itr++)
    blk.insert((*itr).first);

  return blk;
}

void MinimizeFSA::_updateWaiting(unsigned blockX, const _Block& block)
{
  _Block syms;
  for (_BlockConstIterator itr = block.begin(); itr != block.end(); itr++) {
    unsigned stateX(*itr);
    const NodePtr& node(_reverse->find(stateX));

    for (NodeIterator nitr(node); nitr.more(); nitr++) {
      EdgePtr edge(nitr.edge());
      syms.insert(edge->input());
    }
  }

  for (_BlockIterator itr = syms.begin(); itr != syms.end(); itr++) {
    unsigned symbolX(*itr);
    _waiting.push(_WaitKey(blockX, symbolX));
  }
}

void MinimizeFSA::_updateWaiting(unsigned j, const _Block& blockj, unsigned q, const _Block& blockq)
{
  // find symbols in block 'j'
  _Block jsyms;
  for (_BlockConstIterator itr = blockj.begin(); itr != blockj.end(); itr++) {
    unsigned stateX(*itr);
    const NodePtr& node(_reverse->find(stateX));

    for (NodeIterator nitr(node); nitr.more(); nitr++) {
      EdgePtr edge(nitr.edge());
      jsyms.insert(edge->input());
    }
  }

  // find symbols in block 'q'
  _Block qsyms;
  for (_BlockConstIterator itr = blockq.begin(); itr != blockq.end(); itr++) {
    unsigned stateX(*itr);
    const NodePtr& node(_reverse->find(stateX));

    for (NodeIterator nitr(node); nitr.more(); nitr++) {
      EdgePtr edge(nitr.edge());
      qsyms.insert(edge->input());
    }
  }

  // insert elements from block 'j'
  for (_BlockConstIterator itr = jsyms.begin(); itr != jsyms.end(); itr++) {
    unsigned jsymbol(*itr);
    _WaitKey jkey(j, jsymbol);
    if (_waiting.contains(jkey) && qsyms.find(jsymbol) != qsyms.end()) {
      _waiting.push(_WaitKey(q, jsymbol));
      qsyms.erase(jsymbol);
    }
  }

  // here we always place 'q' on waiting list
  for (_BlockConstIterator itr = qsyms.begin(); itr != qsyms.end(); itr++) {
    unsigned qsymbol(*itr);
    _waiting.push(_WaitKey(q, qsymbol));
  }
}

WFSAcceptorPtr MinimizeFSA::_connect()
{
  bool create = true;

  WFSAcceptorPtr fsa(new WFSAcceptor(_fsa->stateLexicon(), _fsa->inputLexicon()));
  fsa->find(_inblock[_fsa->initial()->index()]);

  for (_BlockListIterator itr = _blocklist.begin(); itr != _blocklist.end(); itr++) {
    unsigned fromBlockX((*itr).first);
    NodePtr fromBlock(fsa->find(fromBlockX, create));

    const _Block& block((*itr).second);
    for (_BlockConstIterator bitr = block.begin(); bitr != block.end(); bitr++) {
      unsigned fromStateX(*bitr);

      const NodePtr& fromState(_fsa->find(fromStateX));
      for (NodeIterator sitr(fromState); sitr.more(); sitr++) {
	EdgePtr& edge(sitr.edge());
	unsigned toStateX(edge->next()->index());
	NodePtr  toState(_fsa->find(toStateX));

	unsigned symbolX(edge->input());
	unsigned toBlockX(_inblock[toStateX]);
	NodePtr  toBlock(fsa->find(toBlockX, create));
	EdgePtr  blockEdge(new Edge(fromBlock, toBlock, symbolX));

	fromBlock->_addEdge(blockEdge);
	if (toState->isFinal() && toBlock->isFinal() == false)
	  fsa->_addFinal(toBlockX);
      }
    }
  }

  return fsa;
}

WFSAcceptorPtr minimizeFSA(const WFSAcceptorPtr& fsa)
{
  MinimizeFSA minimizer(fsa);

  return minimizer.minimize();
}


// ----- methods for class `EncodeWFST' -----
//
WFSAcceptorPtr EncodeWFST::encode()
{
  LexiconPtr stateLexicon(new Lexicon("State Lexicon"));
  LexiconPtr symbolLexicon(new Lexicon("Symbol Lexicon"));
  WFSAcceptorPtr fsa(new WFSAcceptor(stateLexicon, symbolLexicon));

  unsigned totalNodes = _maxNodeIndex() + 1;

  // encode initial node
  _encode(fsa, _wfst->initial(), totalNodes);

  // encode intermediate nodes
  for (WFSTransducer::_NodeVector::iterator itr = _wfst->_allNodes().begin(); itr != _wfst->_allNodes().end(); itr++) {
    WFSTransducer::NodePtr& node(*itr);
    if (node.isNull()) continue;
    _encode(fsa, node, totalNodes);
  }

  // encode final nodes
  for (WFSTransducer::_NodeMap::iterator itr = _wfst->_finis().begin(); itr != _wfst->_finis().end(); itr++)
    _encode(fsa, (*itr).second, totalNodes);

  return fsa;
}

unsigned EncodeWFST::_maxNodeIndex()
{
  unsigned maxIndex = _wfst->initial()->index();

  for (WFSTransducer::_NodeVector::iterator itr = _wfst->_allNodes().begin(); itr != _wfst->_allNodes().end(); itr++) {
    WFSTransducer::NodePtr& node(*itr);
    if (node.isNull()) continue;
    if (node->index() > maxIndex)
      maxIndex = node->index();
  }

  for (WFSTransducer::_NodeMap::iterator itr = _wfst->_finis().begin(); itr != _wfst->_finis().end(); itr++)
    if ((*itr).second->index() > maxIndex)
      maxIndex = (*itr).second->index();

  return maxIndex;
}

void EncodeWFST::_encode(WFSAcceptorPtr& fsa, WFSTransducer::NodePtr& node, unsigned& totalNodes)
{
  bool create = true;

  WFSAcceptor::NodePtr from(fsa->find(node->index(), create));
  for (WFSTransducer::Node::Iterator itr(_wfst, node); itr.more(); itr++) {
    WFSTransducer::EdgePtr edge(itr.edge());
    sprintf(_buffer, "%s %-0.4f %s",
	    _wfst->inputLexicon()->symbol(edge->input()).c_str(),
	    float(edge->cost()),
	    _wfst->outputLexicon()->symbol(edge->output()).c_str());

    String symbol(_buffer);
    unsigned symbolX(fsa->inputLexicon()->index(symbol, create));

    unsigned toX(edge->next()->index());
    WFSAcceptor::NodePtr to(fsa->find(toX, create));
    WFSAcceptor::EdgePtr newEdge(new WFSAcceptor::Edge(from, to, symbolX));
    from->_addEdgeForce(newEdge);

    if (edge->next()->isFinal()) {
      if (edge->next()->cost() == ZeroWeight) {
	if (to->isFinal() == false)
	  fsa->_addFinal(toX);
      } else {
	sprintf(_buffer, "%s %-0.4f %s",
		_wfst->inputLexicon()->symbol(0).c_str(),
		float(edge->next()->cost()),
		_wfst->outputLexicon()->symbol(0).c_str());

	symbol  = _buffer;
	symbolX = fsa->inputLexicon()->index(symbol, create);

	WFSAcceptor::NodePtr finalNode(fsa->find(totalNodes, create));
	WFSAcceptor::EdgePtr finalEdge(new WFSAcceptor::Edge(to, finalNode, symbolX));
	to->_addEdgeForce(finalEdge);
	fsa->_addFinal(totalNodes);
	totalNodes++;
      }
    }
  }
}

WFSAcceptorPtr encodeWFST(WFSTransducerPtr& wfst)
{
  EncodeWFST encoder(wfst);
  return encoder.encode();
}


// ----- methods for class `DecodeFSA' -----
//
WFSTSortedInputPtr DecodeFSA::decode()
{
  LexiconPtr stateLexicon(new Lexicon("State Lexicon"));
  WFSTSortedInputPtr wfst(new WFSTSortedInput(stateLexicon, _inputLexicon, _outputLexicon));

  // break symbols into input, output and weight
  for (Lexicon::Iterator itr(_fsa->inputLexicon()); itr.more(); itr++) {
    String symbol(itr.name());
    unsigned symbolX = _fsa->inputLexicon()->index(symbol);

    String input, output;
    float weight;
    _crackSymbol(symbol, input, output, weight);

    unsigned inputX  = _inputLexicon->index(input);
    unsigned outputX = _outputLexicon->index(output);
    Weight cost(weight);

    _SymbolKey skey(inputX, outputX, cost);
    _symbolMap.insert(_SymbolMapValueType(symbolX, skey));
  }

  // decode initial node
  _decode(wfst, _fsa->initial());

  // decode intermediate nodes
  for (WFSAcceptor::_NodeVectorIterator itr = _fsa->_allNodes().begin(); itr != _fsa->_allNodes().end(); itr++) {
    WFSAcceptor::NodePtr& node(*itr);
    if (node.isNull()) continue;
    _decode(wfst, node);
  }

  // decode final nodes
  for (WFSAcceptor::_NodeMapIterator itr = _fsa->_finis().begin(); itr != _fsa->_finis().end(); itr++)
    _decode(wfst, (*itr).second);

  return wfst;
}

void DecodeFSA::_decode(WFSTransducerPtr& wfst, WFSAcceptor::NodePtr& node)
{
  bool create = true;

  WFSTransducer::NodePtr from(wfst->find(node->index(), create));
  for (WFSAcceptor::Node::Iterator itr(node); itr.more(); itr++) {
    WFSAcceptor::EdgePtr edge(itr.edge());

    _SymbolMapIterator sitr = _symbolMap.find(edge->input());
    const _SymbolKey& skey((*sitr).second);

    unsigned toX(edge->next()->index());

    WFSTransducer::NodePtr to(wfst->find(toX, create));
    WFSTransducer::EdgePtr newEdge(new WFSTransducer::Edge(from, to, skey.input(), skey.output(), skey.cost()));
    from->_addEdgeForce(newEdge);

    if (edge->next()->isFinal() && to->isFinal() == false)
      wfst->_addFinal(toX);
  }
}

void DecodeFSA::_crackSymbol(const String& symbol, String& input, String& output, float& weight)
{
  static char inputSymbol[100], outputSymbol[100];
  sscanf(symbol, "%s %f %s", inputSymbol, &weight, outputSymbol);

  input = inputSymbol;  output = outputSymbol;
}

WFSTSortedInputPtr decodeFSA(LexiconPtr& inputLexicon, LexiconPtr& outputLexicon, WFSAcceptorPtr& fsa)
{
  DecodeFSA decoder(inputLexicon, outputLexicon, fsa);
  return decoder.decode();
}


// ----- methods for class `PurgeWFST' -----
//
WFSTSortedInputPtr PurgeWFST::purge(const WFSTransducerPtr& wfst)
{
  WFSTransducerPtr backward(reverse(wfst));

  breadthFirstSearch(backward);

  _whichNodes(backward);

  LexiconPtr statelex(new Lexicon("Purged State Lexicon"));
  WFSTSortedInputPtr forward(new WFSTSortedInput(statelex, wfst->inputLexicon(), wfst->outputLexicon(), wfst->name()));

  _connect(wfst->initial(), forward);

  for (_NodeVectorConstIterator itr = wfst->_allNodes().begin(); itr != wfst->_allNodes().end(); itr++) {
    const _NodePtr& node(*itr);
    if (node.isNull()) continue;
    _connect(node, forward);
  }

  for (_NodeMapConstIterator itr = wfst->_finis().begin(); itr != wfst->_finis().end(); itr++) {
    const _NodePtr& node((*itr).second);
    _connect(node, forward);
  }

  return forward;
}

void PurgeWFST::_whichNodes(const WFSTransducerPtr& backward)
{
  _keep.clear();

  if (backward->initial()->color() == WFSAcceptor::Black)
    _keep.insert(backward->initial()->index());

  for (_NodeVectorConstIterator itr = backward->_allNodes().begin(); itr != backward->_allNodes().end(); itr++) {
    const _NodePtr& node(*itr);
    if (node.isNull()) continue;
    if (node->color() == WFSAcceptor::Black)
      _keep.insert(node->index());
  }

  for (_NodeMapConstIterator itr = backward->_finis().begin(); itr != backward->_finis().end(); itr++) {
    const _NodePtr& node((*itr).second);
    if (node->color() == WFSAcceptor::Black)
      _keep.insert(node->index());
  }
}

void PurgeWFST::_connect(const _NodePtr& node, WFSTSortedInputPtr& forward)
{
  if (_keep.find(node->index()) == _keep.end()) return;

  bool create = true;
  _NodePtr fromNode(forward->find(node->index(), create));
  for (WFSTransducer::Node::Iterator itr(node); itr.more(); itr++) {
    unsigned toIndex = itr.edge()->next()->index();

    if (_keep.find(toIndex) == _keep.end()) continue;

    _NodePtr toNode(forward->find(toIndex, create));
    _EdgePtr edge(new _Edge(fromNode, toNode, itr.edge()->input(), itr.edge()->output(), itr.edge()->cost()));
    fromNode->_addEdgeForce(edge);

    if (itr.edge()->next()->isFinal() && toNode->isFinal() == false)
      forward->_addFinal(toIndex, itr.edge()->next()->cost());
  }
}

WFSTSortedInputPtr purgeWFST(const WFSTransducerPtr& wfst)
{
  PurgeWFST purge;
  return purge.purge(wfst);
}
