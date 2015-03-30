//
//                                Millennium
//                    Distant Speech Recognition System
//                                  (dsr)
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


#include <algorithm>
#include "decoder/wfstFlyWeight.h"
#include "common/mach_ind_io.h"


// ----- methods for class `WFSTFlyWeight' -----
//
const int WFSTFlyWeight::EndMarker = 2147483647; 	// i.e., INT_MAX on a 32-bit machine

WFSTFlyWeight::WFSTFlyWeight(LexiconPtr& statelex, LexiconPtr& inlex, LexiconPtr& outlex, const String& name)
  : _name(name), _stateLexicon(statelex), _inputLexicon(inlex), _outputLexicon(outlex),
    _initial(NULL) { }

WFSTFlyWeight::~WFSTFlyWeight() { _clear(); }

// Note: Must explicitly clear nodes because all links hold smart
//	 pointers to their 'from' nodes.
void WFSTFlyWeight::_clear()
{
  // printf("Clearing Fly Weight WFST\n");  fflush(stdout);

  delete _initial;  _initial = NULL;

  for (_NodeMapIterator itr = _nodes.begin(); itr != _nodes.end(); itr++)
    delete (*itr).second;
  _nodes.erase(_nodes.begin(), _nodes.end());

  for (_NodeMapIterator itr = _final.begin(); itr != _final.end(); itr++)
    delete (*itr).second;
  _final.erase(_final.begin(), _final.end());
}

bool WFSTFlyWeight::hasFinal(unsigned state)
{
  _NodeMapIterator itr = _final.find(state);
  return itr != _final.end();
}

WFSTFlyWeight::Node* WFSTFlyWeight::_newNode(unsigned state)
{
  return new Node(state);
}

WFSTFlyWeight::Edge* WFSTFlyWeight::
_newEdge(Node* from, Node* to, unsigned input, unsigned output, float cost)
{
  return new Edge(from, to, input, output, cost);
}

void WFSTFlyWeight::_addFinal(unsigned state, float cost)
{
  if (hasFinal(state))
    throw jconsistency_error("Automaton already has final node %d.",
			     state);

  _NodeMapIterator itr = _nodes.find(state);
  if (itr == _nodes.end()) {
    Node* ptr(_newNode(state));
    _final.insert(_ValueType(state, ptr));
    ptr->_setCost(cost);
  } else {
    Node* ptr((*itr).second);
    ptr->_setCost(cost);
    _final.insert(_ValueType(state, ptr));
    _nodes.erase(itr);
  }
}

WFSTFlyWeight::Node* WFSTFlyWeight::find(unsigned state, bool create)
{
  if (initial()->index() == state)
    return initial();

  _NodeMapIterator itr = _nodes.find(state);

  if (itr != _nodes.end())
    return (*itr).second;

  itr = _final.find(state);

  if (itr != _final.end()) {
    assert((*itr).second->isFinal() == true);
    return (*itr).second;
  }

  if (create == false)
    throw jkey_error("No state %u exists.", state);

  // Fix this!
  _nodes.insert(_ValueType(state, _newNode(state)));
  itr = _nodes.find(state);

  return (*itr).second;
}

void WFSTFlyWeight::read(const String& fileName, bool binary)
{
  if (fileName == "")
    jio_error("File name is null.");

  _clear();

  printf("\nReading Fly Weight WFST from file %s\n", fileName.c_str());

  FILE* fp;
  if (binary) {
    fp = fileOpen(fileName, "rb");
    _readBinary(fp);
  } else {
    fp = fileOpen(fileName, "r");
    _readText(fp);
  }

  fileClose( fileName, fp);
}

void WFSTFlyWeight::reverse(const WFSTFlyWeightPtr& wfst)
{
  typedef _NodeMap::const_iterator ConstNodeIterator;

  _clear();
  bool create = true;

  // create super initial and final states
  Node* rinitial(initial(WFSTFlyWeight::Node::_MaximumIndex-3));
  _addFinal(wfst->initial()->index(), /* cost= */ 0.0);

  // add arcs from final (i.e., initial) node
  // printf("From final (i.e., initial) node:\n");
  Node* rfinal(find(wfst->initial()->index()));
  for (Node::Iterator eitr(wfst->initial()); eitr.more(); eitr++) {
    const Edge* edge(eitr.edge());
    const Node* node2(edge->next());
    Node* rnode2(find(node2->index(), create));
    Edge* redge(_newEdge(rnode2, rfinal, edge->input(), edge->output(), edge->cost()));
    /*
    redge->write(stateLexicon(), inputLexicon(), outputLexicon());
    */
    rnode2->_addEdgeForce(redge);
  }  
  
  // add arcs from super initial node
  // printf("From super initial node:\n");
  for (ConstNodeIterator nitr = wfst->final().begin(); nitr != wfst->final().end(); nitr++) {
    const Node* node((*nitr).second);
    Node* rnode(find(node->index(), create));
    Edge* redge(_newEdge(rinitial, rnode, 0, 0, node->cost()));
    /*
    redge->write(stateLexicon(), inputLexicon(), outputLexicon());
    */
    rinitial->_addEdgeForce(redge);
  }

  // add arcs from final nodes
  // printf("From final nodes:\n");
  for (_NodeMapIterator nitr = wfst->final().begin(); nitr != wfst->final().end(); nitr++) {
    Node* node1((*nitr).second);
    Node* rnode1(find(node1->index(), create));
    for (Node::Iterator eitr(node1); eitr.more(); eitr++) {
      const Edge* edge(eitr.edge());
      const Node* node2(edge->next());
      Node* rnode2(find(node2->index(), create));
      Edge* redge(_newEdge(rnode2, rnode1, edge->input(), edge->output(), edge->cost()));
      /*
      redge->write(stateLexicon(), inputLexicon(), outputLexicon());
      */
      rnode2->_addEdgeForce(redge);
    }
  }

  // add arcs from internal nodes
  // printf("From internal nodes:\n");
  for (_NodeMapIterator nitr = wfst->_nodes.begin(); nitr != wfst->_nodes.end(); nitr++) {
    Node* node1((*nitr).second);
    if (node1 == NULL) continue;
    Node* rnode1(find(node1->index(), create));
    for (Node::Iterator eitr(node1); eitr.more(); eitr++) {
      const Edge* edge(eitr.edge());
      const Node* node2(edge->next());
      Node* rnode2(find(node2->index(), create));
      Edge* redge(_newEdge(rnode2, rnode1, edge->input(), edge->output(), edge->cost()));
      /*
      redge->write(stateLexicon(), inputLexicon(), outputLexicon());
      */
      rnode2->_addEdgeForce(redge);
    }
  }  
}

// read a fly weight transducer in reverse order
void WFSTFlyWeight::reverseRead(const String& fileName)
{
  if (fileName == "")
    jio_error("File name is null.");

  _clear();

  // create super initial and final states
  Node* rinitial(initial(WFSTFlyWeight::Node::_MaximumIndex-3));
  bool initialFlag = false;

  printf("\nReverse Fly Weight WFST from file %s\n", fileName.c_str());

  FILE*         fp     = fileOpen(fileName, "r");
  static size_t n      = 0;
  static char*  buffer = NULL;

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

      Node* rnode(find(s1));
      Edge* redge = _newEdge(rinitial, rnode, 0, 0);
      rinitial->_addEdgeForce(redge);

    } else if (i == 2) {		// add a final state with non-zero cost

      float cost;
      sscanf(token[1], "%f", &cost);

      Node* rnode(find(s1));
      Edge* redge = _newEdge(rinitial, rnode, 0, 0, cost);
      rinitial->_addEdgeForce(redge);

    } else if (i == 4 || i == 5) {	// add an arc

      bool create = true;

      p = NULL;
      unsigned s2 = strtoul(token[1], &p, 0);
      if (p == token[1])
	s2 = _stateLexicon->index(token[1]);

      if (initialFlag == false) { _addFinal(s1, /* cost= */ 0.0);  initialFlag = true; }
      Node* from(find(s1, create));
      Node* to(find(s2, create));

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

      Edge* edgePtr = _newEdge(to, from, input, output, cost);
      to->_addEdgeForce(edgePtr);

    } else
      throw jio_error("Transducer file %s is inconsistent.", fileName.chars());
  }

  fileClose( fileName, fp);
}

void WFSTFlyWeight::_readText(FILE* fp)
{
  static size_t n = 0;
  static char*  buffer = NULL;

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

      _addFinal(s1, /* cost= */ 0.0);

    } else if (i == 2) {		// add a final state with non-zero cost

      float cost;
      sscanf(token[1], "%f", &cost);
      _addFinal(s1, cost);
      
    } else if (i == 4 || i == 5) {	// add an arc

      bool create = true;

      p = NULL;
      unsigned s2 = strtoul(token[1], &p, 0);
      if (p == token[1])
	s2 = _stateLexicon->index(token[1]);

      Node* from;
      if (_initial == NULL)
	_initial = from = _newNode(s1);
      else
	from = find(s1, create);
      Node* to = find(s2, create);

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

      Edge* edgePtr = _newEdge(from, to, input, output, cost);
      from->_addEdgeForce(edgePtr);

    } else
      throw jio_error("Transducer file is inconsistent.");
  }
}

void WFSTFlyWeight::_readBinary(FILE* fp)
{
  size_t  n;
  static char* buffer = NULL;

  int nEntries;
  while ((nEntries = read_int(fp)) != EndMarker) {

    if (nEntries == 3) {		    // read a state

      int   index = read_int(fp);
      float cost  = read_float(fp);
      int   end   = read_int(fp);

      if (end != EndMarker)
	throw jio_error("Transducer binary file is inconsistent.");

      _addFinal(index, cost);

    } else if (nEntries == 6) {		    // read an arc
      
      int   s1     = read_int(fp);
      int   s2     = read_int(fp);
      int   input  = read_int(fp);
      int   output = read_int(fp);
      float cost   = read_float(fp);
      int   end    = read_int(fp);

      if (end != EndMarker)
	throw jio_error("Transducer binary file is inconsistent.");

      bool create = true;
      Node* from;
      if (_initial == NULL)
	_initial = from = _newNode(s1);
      else
	from = find(s1, create);
      Node* to = find(s2, create);

      Edge* edgePtr = _newEdge(from, to, input, output, cost);
      from->_addEdgeForce(edgePtr);

    } else
      	throw jio_error("Transducer binary file is inconsistent.");

  }
}

void WFSTFlyWeight::write(const String& fileName, bool binary, bool useSymbols) const
{
  FILE* fp;
  if (fileName == "")
    throw jio_error("Must specify a non-null file name for writing.");

  if (binary)
    fp = fileOpen(fileName, "wb");
  else
    fp = fileOpen(fileName, "w");

  // write edges leaving from initial state
  for (Node::Iterator itr(initial()); itr.more(); itr++)
    if (useSymbols)
      itr.edge()->write(stateLexicon(), inputLexicon(), outputLexicon(), fp);
    else
      itr.edge()->write(fp, this, binary);

  // write edges leaving from intermediate states
  for (_ConstNodeMapIterator itr = _nodes.begin(); itr != _nodes.end(); itr++) {
    Node* nd((*itr).second);

    for (Node::Iterator itr(nd); itr.more(); itr++)
      if (useSymbols)
	itr.edge()->write(stateLexicon(), inputLexicon(), outputLexicon(), fp);
      else
	itr.edge()->write(fp, this, binary);
  }

  // write final states
  for (_ConstNodeMapIterator itr = _final.begin(); itr != _final.end(); itr++) {
    Node* nd((*itr).second);

    // write edges
    for (Node::Iterator itr(nd); itr.more(); itr++)
      if (useSymbols)
	itr.edge()->write(stateLexicon(), inputLexicon(), outputLexicon(), fp);
      else
	itr.edge()->write(fp, this, binary);

    // write nodes
    nd->write(fp, binary);
  }

  if (binary)
    write_int(fp, EndMarker);

  fileClose(fileName, fp);
}


// ----- methods for class `WFSTFlyWeight::Edge' -----
//
MemoryManager<WFSTFlyWeight::Edge>& WFSTFlyWeight::Edge::memoryManager()
{
  static MemoryManager<Edge> _MemoryManager("WFSTFlyWeight::Edge");
  return _MemoryManager;
}

void WFSTFlyWeight::Edge::write(FILE* fp, const WFSTFlyWeight* wfst, bool binary) const
{
  if (binary) {

    write_int(fp, 6);
    write_int(fp,   prev()->index());
    write_int(fp,   next()->index());
    write_int(fp,   input());
    write_int(fp,   output());
    write_float(fp, cost());
    write_int(fp, EndMarker);

  } else {

    fprintf(fp, "%10d  %10d  %10d  %10d",
	    prev()->index(), next()->index(), input(), output());

    if (cost() == 0.0) fprintf(fp, "\n");
    else fprintf(fp, "  %12g\n", cost());

  }
}

static const double MinimumCost = 1.0E-04;

void WFSTFlyWeight::Edge::write(const LexiconPtr& statelex, const LexiconPtr& inlex, const LexiconPtr& outlex, FILE* fp) const
{
  if (statelex.isNull() || statelex->size() == 0)
    fprintf(fp, "%10d  %10d  %10s  %20s",
	    prev()->index(), next()->index(),
	    (inlex->symbol(input())).c_str(),
	    (outlex->symbol(output())).c_str());
  else
    fprintf(fp, "%25s  %25s  %10s  %20s",
	    (statelex->symbol(prev()->index())).c_str(),
	    (statelex->symbol(next()->index())).c_str(),
	    (inlex->symbol(input())).c_str(),
	    (outlex->symbol(output())).c_str());

  if (fabs(float(cost())) < MinimumCost) fprintf(fp, "\n");
  else fprintf(fp, "  %12g\n", float(cost()));
}


// ----- methods for class `WFSTFlyWeight::Node' -----
//
const unsigned WFSTFlyWeight::Node::_MaximumIndex = 536870911;

MemoryManager<WFSTFlyWeight::Node>& WFSTFlyWeight::Node::memoryManager()
{
  static MemoryManager<Node> _MemoryManager("WFSTFlyWeight::Node");
  return _MemoryManager;
}

WFSTFlyWeight::Node::Node(unsigned idx, float cost)
  : _cost(0.0), _edgeList(NULL)
{
  _index.index = idx;
  _index.color = 0;	// i.e., 'White'
  _index.final = 0;

  if (cost > 0.0) _setCost(cost);
}

WFSTFlyWeight::Node::~Node()
{
  delete _edgeList;
}

void WFSTFlyWeight::Node::writeArcs(const LexiconPtr& statelex, const LexiconPtr& inlex, const LexiconPtr& outlex, FILE* fp) const
{
  const Edge* edge(_edges());
  while (edge != NULL) {
    edge->write(statelex, inlex, outlex, fp);
    edge = edge->_edges();
  }
}

void WFSTFlyWeight::Node::_addEdgeForce(Edge* newEdge)
{
  newEdge->_edgeList = _edgeList;
  _edgeList  = newEdge;
}

void WFSTFlyWeight::Node::write(FILE* fp, bool binary)
{
  if (binary) {

    write_int(fp, 3);
    write_int(fp, index());
    write_float(fp, cost());
    write_int(fp, EndMarker);

  } else {

    if (_cost == 0.0)
      fprintf(fp, "%10d\n", index());
    else
      fprintf(fp, "%10d  %12g\n", index(), cost());

  }
}

void WFSTFlyWeight::reportMemoryUsage()
{
  // WFSTFlyWeight::memoryManager().report();
  WFSTFlyWeight::Edge::memoryManager().report();
  WFSTFlyWeight::Node::memoryManager().report();  
}

void WFSTFlyWeight_reportMemoryUsage() { WFSTFlyWeight::reportMemoryUsage(); }


// ----- methods for class `WFSTFlyWeightSortedInput' -----
//
void WFSTFlyWeightSortedInput::hash()
{
  initial()->hash();
  for (_NodeMapIterator itr = _nodes.begin(); itr != _nodes.end(); itr++) {
    Node* nd(Cast<Node*>((*itr).second));
    nd->hash();
  }

  for (_ConstNodeMapIterator itr = _final.begin(); itr != _final.end(); itr++) {
    Node* nd(Cast<Node*>((*itr).second));
    nd->hash();
  }
}

WFSTFlyWeight::Node* WFSTFlyWeightSortedInput::_newNode(unsigned state)
{
  return new Node(state);
}

WFSTFlyWeight::Edge* WFSTFlyWeightSortedInput::
_newEdge(WFSTFlyWeight::Node* from, WFSTFlyWeight::Node* to, unsigned input, unsigned output, float cost)
{
  return new Edge(Cast<Node*>(from), Cast<Node*>(to), input, output, cost);
}


// ----- methods for class `WFSTFlyWeightSortedInput::Edge' -----
//
WFSTFlyWeightSortedInput::Edge::
Edge(Node* prev, Node* next, unsigned input, unsigned output, float cost)
  : WFSTFlyWeight::Edge(prev, next, input, output, cost), _chain(NULL) { }

MemoryManager<WFSTFlyWeightSortedInput::Edge>& WFSTFlyWeightSortedInput::Edge::memoryManager()
{
  static MemoryManager<Edge> _MemoryManager("WFSTFlyWeightSortedInput::Edge");
  return _MemoryManager;
}


// ----- methods for class `WFSTFlyWeightSortedInput::Node' -----
//
bool WFSTFlyWeightSortedInput::Node::Verbose = false;

MemoryManager<WFSTFlyWeightSortedInput::Node>& WFSTFlyWeightSortedInput::Node::memoryManager()
{
  static MemoryManager<Node> _MemoryManager("WFSTFlyWeightSortedInput::Node");
  return _MemoryManager;
}

void WFSTFlyWeightSortedInput::Node::_addEdgeForce(WFSTFlyWeight::Edge* newEdge)
{
  WFSTFlyWeight::Edge* ptr    = _edges();
  WFSTFlyWeight::Edge* oldPtr = ptr;
  while (ptr != NULL && ptr->input() < newEdge->input()) {
    oldPtr = ptr;
    ptr    = ptr->_edges();
  }
  while (ptr != NULL && ptr->input() == newEdge->input() && ptr->output() < newEdge->output()) {
    oldPtr = ptr;
    ptr    = ptr->_edges();
  }

  if (ptr == oldPtr) {
    newEdge->_edgeList = _edgeList;
    _edgeList  = newEdge;

  } else {

    newEdge->_edgeList = ptr;
    oldPtr->_edgeList  = newEdge;

  }
}


// ----- methods for class `WFSTFlyWeightSortedOutput' -----
//
WFSTFlyWeight::Node* WFSTFlyWeightSortedOutput::_newNode(unsigned state)
{
  return new Node(state);
}

WFSTFlyWeight::Edge* WFSTFlyWeightSortedOutput::
_newEdge(WFSTFlyWeight::Node* from, WFSTFlyWeight::Node* to, unsigned input, unsigned output, float cost)
{
  return new Edge(Cast<Node*>(from), Cast<Node*>(to), input, output, cost);
}

void WFSTFlyWeightSortedOutput::setComingSymbols()
{
  Node* init(initial());

  FirstInFirstOut<Node*> stateQueue;  stateQueue.clear();  stateQueue.push(initial());
  set<unsigned> discovered;

  while (stateQueue.more()) {
    Node* node(stateQueue.pop());
    node->findComing();

    for (Node::Iterator itr(node); itr.more(); itr++) {
      Edge* edge(itr.edge());
      if (edge->prev() == edge->next()) continue;	// skip self-loops
      if (discovered.find(edge->next()->index()) == discovered.end()) {
	stateQueue.push(edge->next()); discovered.insert(edge->next()->index());
      }
    }
  }
}


// ----- methods for class `WFSTFlyWeightSortedOutput::Edge' -----
//
WFSTFlyWeightSortedOutput::Edge::
Edge(Node* prev, Node* next, unsigned input, unsigned output, float cost)
  : WFSTFlyWeight::Edge(prev, next, input, output, cost) { }

MemoryManager<WFSTFlyWeightSortedOutput::Edge>& WFSTFlyWeightSortedOutput::Edge::memoryManager()
{
  static MemoryManager<Edge> _MemoryManager("WFSTFlyWeightSortedOutput::Edge");
  return _MemoryManager;
}

list<unsigned> WFSTFlyWeightSortedOutput::Edge::findComing() const
{
  list<unsigned> symbols;
  if (output() == 0) {
    set<unsigned> coming;
    for (Node::Iterator itr(next()); itr.more(); itr++) {
      const Edge* edge(itr.edge());
      if (edge->prev() == edge->next()) continue;	// skip self-loops

      list<unsigned> fromEdge(edge->findComing());
      for (list<unsigned>::iterator itr = fromEdge.begin(); itr != fromEdge.end(); itr++)
	coming.insert(*itr);
    }

    for (set<unsigned>::iterator itr = coming.begin(); itr != coming.end(); itr++)
      symbols.push_back(*itr);

  } else {
    symbols.push_back(output());
  }

  return symbols;
}


// ----- methods for class `WFSTFlyWeightSortedOutput::Node' -----
//
MemoryManager<WFSTFlyWeightSortedOutput::Node>& WFSTFlyWeightSortedOutput::Node::memoryManager()
{
  static MemoryManager<Node> _MemoryManager("WFSTFlyWeightSortedOutput::Node");
  return _MemoryManager;
}

WFSTFlyWeightSortedOutput::Node::Node(unsigned idx, float cost)
  : WFSTFlyWeight::Node(idx, cost), _comingSymbols(NULL) { }

WFSTFlyWeightSortedOutput::Node::~Node()
{
  delete[] _comingSymbols;
}

void WFSTFlyWeightSortedOutput::Node::_addEdgeForce(WFSTFlyWeight::Edge* newEdge)
{
  WFSTFlyWeight::Edge* ptr    = _edges();
  WFSTFlyWeight::Edge* oldPtr = ptr;
  while (ptr != NULL && ptr->output() < newEdge->output()) {
    oldPtr = ptr;
    ptr    = ptr->_edges();
  }
  while (ptr != NULL && ptr->output() == newEdge->output() && ptr->input() < newEdge->input()) {
    oldPtr = ptr;
    ptr    = ptr->_edges();
  }

  if (ptr == oldPtr) {

    newEdge->_edgeList = _edgeList;
    _edgeList  = newEdge;

  } else {

    newEdge->_edgeList = ptr;
    oldPtr->_edgeList  = newEdge;

 }
}

void WFSTFlyWeightSortedOutput::Node::findComing()
{
  if (_comingSymbols != NULL) return;

  set<unsigned> symbols;
  for (Iterator itr(this); itr.more(); itr++) {
    const Edge* edge(itr.edge());
    if (edge->prev() == edge->next()) continue;		// skip self-loops

    list<unsigned> sym(edge->findComing());
    for (list<unsigned>::iterator itr = sym.begin(); itr != sym.end(); itr++)
      symbols.insert(*itr);
  }

  unsigned symX  = 0;
  unsigned nsym  = symbols.size();
  vector<unsigned> symvec(nsym);
  for (set<unsigned>::iterator itr = symbols.begin(); itr != symbols.end(); itr++)
    symvec[symX++] = *itr;
  sort(symvec.begin(), symvec.end());

  symX = 0;
  _comingSymbols = new unsigned[nsym+1];
  for (vector<unsigned>::iterator itr = symvec.begin(); itr != symvec.end(); itr++)
    _comingSymbols[symX++] = *itr;
  _comingSymbols[nsym] = UINT_MAX;

  /*
  printf("Coming Symbols:\n");
  for (unsigned i = 0; i < nsym; i++)
    printf("  %4d\n", _comingSymbols[i]);
  printf("\n");
  */
}


// ----- methods for class `WFSTFlyWeightSortedInput::Node::_EdgeMap' -----
//
const float    WFSTFlyWeightSortedInput::Node::MaxHashDepth = 1.2;
const unsigned WFSTFlyWeightSortedInput::Node::PrimeN       = 36;
const unsigned WFSTFlyWeightSortedInput::Node::Primes[]     = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151};

WFSTFlyWeightSortedInput::Node::_EdgeMap::_EdgeMap()
  : _bins(NULL) { }

WFSTFlyWeightSortedInput::Node::_EdgeMap::~_EdgeMap()
{
  delete[] _bins;
}

float WFSTFlyWeightSortedInput::Node::_EdgeMap::hash(Edge* edgeList)
{
  unsigned edgeN = 0;
  WFSTFlyWeightSortedInput::Edge* edge = edgeList;
  while (edge != NULL) {
    edgeN++;
    edge = edge->_edges();
  }

  unsigned primeX;
  for (primeX = 0; primeX < PrimeN; primeX++)
    if (Primes[primeX] > edgeN) break;

  float hashDepth;
  do {
    if (primeX == PrimeN)
      throw jkey_error("No prime number with index %d", primeX);

    delete[] _bins;
    _binsN = Primes[primeX++];
    _bins  = new Edge*[_binsN];
    for (unsigned binX = 0; binX < _binsN; binX++)
      _bins[binX] = NULL;

    // printf("Hashing to %d bins.\n", _binsN);

    edge   = edgeList;
    while (edge != NULL) {
      unsigned hashKey = _hash(edge->input());
      edge->_chain     = _bins[hashKey];
      _bins[hashKey]   = edge;
      edge = edge->_edges();
    }
    hashDepth = averageHashDepth();
  } while (hashDepth > MaxHashDepth);

  return hashDepth;
}

float WFSTFlyWeightSortedInput::Node::_EdgeMap::averageHashDepth() const
{
  unsigned filledBinsN = 0;
  unsigned keysN       = 0;
  for (unsigned binX = 0; binX < _binsN; binX++) {
    if (_bins[binX] != NULL) {
      filledBinsN++;
      WFSTFlyWeightSortedInput::Edge* edge = _bins[binX];
      do {
	keysN++;
	edge = edge->_chain;
      } while (edge != NULL);
    }
  }

  return float(keysN) / float(filledBinsN);
}


// ----- methods for class `WFSTFlyWeightSortedInput::Node' -----
//
void WFSTFlyWeightSortedInput::Node::hash()
{
  float hashDepth = _edgeMap.hash(_edges());
  if (Verbose)
    printf("Node %d has average hash depth of %6.4f\n", index(), hashDepth);
}


// ----- methods for class `Fence' -----
//
MemoryManager<Fence::FenceIndex>&
Fence::FenceIndex::memoryManager()
{
  static MemoryManager<FenceIndex> _MemoryManager("Fence::FenceIndex");
  return _MemoryManager;
}

Fence::Fence(unsigned bucketN, const String& fileName, float rehashFactor)
  : _rehashFactor(rehashFactor), _bucketN(bucketN), _bucketN2(unsigned(bucketN * _rehashFactor)),
    _indexN(0), _fence(new FenceIndex*[_bucketN]), _list(NULL) { }

Fence::~Fence()
{
  clear();
  delete[] _fence;
}

void Fence::clear()
{
  FenceIndex* current = _list;
  while (current != NULL) {
    FenceIndex* next = current->_next;
    delete current;
    current = next;
  }

  _list = NULL;  _indexN = 0;
  for (unsigned i = 0; i < _bucketN; i++)
    _fence[i] = NULL;
}

void Fence::_rehash()
{
  delete[] _fence;  _bucketN *= 2;  _bucketN2 = unsigned (_bucketN * _rehashFactor);

  printf("Rehashing 'Fence' to %d buckets.\n", _bucketN);  fflush(stdout);
  printf("_indexN = %d : _bucketN = %d : _bucketN2 = %d\n", _indexN, _bucketN, _bucketN2);  fflush(stdout);

  _fence = new FenceIndex*[_bucketN];
  for (unsigned i = 0; i < _bucketN; i++)
    _fence[i] = NULL;

  FenceIndex* next = _list;
  while (next != NULL) {
    
    unsigned bucketX = _hash(next->_indexA, next->_indexB);
    next->_chain     = _fence[bucketX];
    _fence[bucketX]  = next;
    
    next = next->_next;
  }
}

bool Fence::insert(unsigned indexA, unsigned indexB)
{
  if (_indexN > _bucketN2) _rehash();

  unsigned    bucketX   = _hash(indexA, indexB);
  FenceIndex* holder    = _fence[bucketX];
  while (holder != NULL) {
    if (holder->_indexA == indexA && holder->_indexB == indexB) return false;
    holder = holder->_chain;
  }

  FenceIndex* newHolder = new FenceIndex(indexA, indexB, _list, _fence[bucketX]);
  _list                 = newHolder;
  _fence[bucketX]       = newHolder;

  _indexN++;
  return true;
}

bool Fence::present(unsigned indexA, unsigned indexB) const
{
  FenceIndex* holder = _fence[_hash(indexA, indexB)];
  while (holder != NULL) {
    if (holder->_indexA == indexA && holder->_indexB == indexB)
      return true;
    holder = holder->_chain;
  }
  return false;
}

void Fence::read(const String& fileName)
{
  clear();

  FILE* fp = fileOpen(fileName, "rb");
  unsigned fenceN = read_int(fp);
  for (unsigned i = 0; i < fenceN; i++) {
    unsigned indexA = read_int(fp);
    unsigned indexB = read_int(fp);
    insert(indexA, indexB);
  }
  fileClose(fileName, fp);
}

void Fence::write(const String& fileName) const
{
  FILE* fp = fileOpen(fileName, "wb");
  write_int(fp, size());
  for (FenceIndex* next = _list; next != NULL; next = next->_next) {
    write_int(fp, next->_indexA); write_int(fp, next->_indexB);
  }
  fileClose(fileName, fp);
}


// ----- methods for class `FenceFinder::_State::_Iterator' -----
//
void FenceFinder::_State::_Iterator::operator++(int)
{
  bool matchFound = false;
  while (_edgeA != NULL && matchFound == false) {

    unsigned wordX = _edgeA->output();

    if (wordX == 0) {
      _nodeA = _edgeA->next();
      matchFound = true;
    } else {
      const WFSTFlyWeight::Edge* testB = (_edgeB == NULL) ? _nodeB->_edges() : _edgeB;
      while (testB != NULL && testB->input() < wordX) {
	if (testB->_edges() != NULL && testB->input() == testB->_edges()->input())
	  throw jindex_error("Two edges in B transducer with symbol %d", testB->input());

	_edgeB = testB;  testB = testB->_edges();
      }

      if (testB != NULL && testB->input() == wordX) {
	_edgeB = testB;
	_nodeA = _edgeA->next();
	_nodeB = _edgeB->next();
	matchFound = true;
      }
    }
    _edgeA = _edgeA->_edges();
  }

  if (matchFound == false) { _nodeA = _nodeB = NULL; return; }
}


// ----- methods for class `FenceFinder::_State::_NonDeterministicIterator' -----
//
void FenceFinder::_State::_NonDeterministicIterator::operator++(int)
{
  if (_edgeA == NULL) { _nodeA = _nodeB = NULL; return; }
  if (_edgeA->output() == 0) {		// process epsilons on 'edgeA'
    _nodeA = _edgeA->next();
    _edgeA = _edgeA->_edges();
    return;
  }

  if (_edgeB == NULL)
    _edgeB = _nodeB->_edges();
  if (_edgeB == NULL) { _nodeA = _nodeB = NULL; return; }

  bool matchFound = false;		// process non-epsilons on 'edgeA'
  while (_edgeA != NULL && matchFound == false) {
    unsigned wordX = _edgeA->output();
    if (_edgeB->input() == wordX) {
      matchFound = true;
      _nodeA     = _edgeA->next();
      _nodeB     = _edgeB->next();
      if (_edgeB->_edges() != NULL && _edgeB->_edges()->input() == wordX) {
	_edgeB = _edgeB->_edges();
      } else {
	_edgeA = _edgeA->_edges();
	if (_edgeA != NULL && _edgeA->output() == wordX)
	  _edgeB = _firstMatchEdgeB;
      }
    } else {
      const WFSTFlyWeight::Edge* testB = _edgeB;
      do { _edgeB = testB;  testB = testB->_edges(); } while (testB != NULL && testB->input() < wordX);
      if (testB == NULL) break;
      if (testB->input() == wordX) {	// match found
	matchFound       = true;
	_nodeA           = _edgeA->next();
	_nodeB           = testB->next();
	_firstMatchEdgeB = testB;
	if (testB->_edges() != NULL) {	// attempt to advance 'edgeB' pointer
	  _edgeB = testB->_edges();
	} else {			// else advance 'edgeA' pointer
	  _edgeA = _edgeA->_edges();
	}
      } else {				// no match for 'wordX': advance 'edgeA' pointer
	_firstMatchEdgeB = _edgeB;
	_edgeA = _edgeA->_edges();
      }
    }
  }
  if (matchFound == false) { _nodeA = _nodeB = NULL; return; }
}


// ----- methods for class `FenceFinder' -----
//
FenceFinder::FenceFinder(const WFSTFlyWeightSortedOutputPtr& wfstA, const WFSTFlyWeightSortedInputPtr& wfstB, unsigned bucketN)
  : _wfstA(wfstA), _wfstB(wfstB), _discoveredNodes(bucketN), _accessibleNodes(bucketN), _coaccessibleNodes(bucketN)
{
  LexiconPtr stateLexiconA(new Lexicon("State Lexicon A"));
  const String nameA(String("WFST A Reversed"));
  _wfstAReverse = new WFSTFlyWeightSortedOutput(stateLexiconA, wfstA->inputLexicon(), wfstA->outputLexicon(), nameA);
  _wfstAReverse->reverse(_wfstA);

  LexiconPtr stateLexiconB(new Lexicon("State Lexicon B"));
  const String nameB(String("WFST B Reversed"));
  _wfstBReverse = new WFSTFlyWeightSortedInput(stateLexiconB, wfstB->inputLexicon(), wfstB->outputLexicon(), nameB);
  _wfstBReverse->reverse(_wfstB);
}

MemoryManager<FenceFinder::_State>&
FenceFinder::_State::memoryManager()
{
  static MemoryManager<_State> _MemoryManager("Fence::_State");
  return _MemoryManager;
}

FencePtr FenceFinder::fence()
{
  _findAccessibleNodes();
  _findCoaccessibleNodes();
  return _findFence();
}

void FenceFinder::_findAccessibleNodes()
{
  unsigned initialA = _wfstA->initial()->index();
  unsigned initialB = _wfstB->initial()->index();

  _State initialState(_wfstA->initial(), _wfstB->initial());

  _endNodes.clear();
  _accessibleNodes.clear();	_accessibleNodes.insert(initialA, initialB);
  _stateQueue.clear();		_stateQueue.push(initialState);

  unsigned pops = 0;
  while (_stateQueue.more()) {
    const _State state(_stateQueue.pop());

    if (state.nodeA()->isFinal() && state.nodeB()->isFinal()) {
      printf("Node (%d x %d) is final.\n", state.nodeA()->index(), state.nodeB()->index());
      _endNodes.insert(state.nodeA()->index(), state.nodeB()->index());
    }

    _State::_Iterator eitr(state);
    while (eitr.more()) {
      unsigned indexA = eitr.nodeA()->index();
      unsigned indexB = eitr.nodeB()->index();
      if (_accessibleNodes.insert(indexA, indexB))
	_stateQueue.push(_State(eitr.nodeA(), eitr.nodeB()));
      eitr++;
    }
    if (++pops % 100000 == 0) {
      printf("%d end nodes discovered : %d accessible nodes discovered : %d nodes to pop\n",
	     _endNodes.size(), _accessibleNodes.size(), _stateQueue.elementN());  fflush(stdout); }
  }
  printf("There are %d accessible nodes.\n", _accessibleNodes.size());
  printf("There are %d end nodes.\n", _endNodes.size());
  fflush(stdout);
}

void FenceFinder::_findCoaccessibleNodes()
{
  printf("Finding coaccessible nodes ...\n");  fflush(stdout);

  _coaccessibleNodes.clear();  _stateQueue.clear();
  for (Fence::Iterator itr(_endNodes); itr.more(); itr++) {
    unsigned indexA = itr.index().indexA();
    unsigned indexB = itr.index().indexB();
    _coaccessibleNodes.insert(indexA, indexB);

    _State initialState(_wfstAReverse->find(indexA), _wfstBReverse->find(indexB));
    _stateQueue.push(initialState);
  }

  unsigned pops = 0;
  while (_stateQueue.more()) {
    const _State state(_stateQueue.pop());

    _State::_NonDeterministicIterator eitr(state);
    while (eitr.more()) {
      unsigned indexA = eitr.nodeA()->index();
      unsigned indexB = eitr.nodeB()->index();

      if (_accessibleNodes.present(indexA, indexB) && _coaccessibleNodes.insert(indexA, indexB))
	_stateQueue.push(_State(eitr.nodeA(), eitr.nodeB()));
      eitr++;
    }
    if (++pops % 100000 == 0) {
      printf("%d coaccessible nodes discovered : %d nodes to pop\n",
	     _coaccessibleNodes.size(), _stateQueue.elementN());  fflush(stdout); }
  }  
  printf("There are %d coaccessible nodes.\n", _coaccessibleNodes.size());  fflush(stdout);
}

FencePtr FenceFinder::_findFence()
{
  FencePtr fen(new Fence());
  printf("Finding fence nodes ...\n");  fflush(stdout);

  _State initialState(_wfstA->initial(), _wfstB->initial());
  _discoveredNodes.clear();  _discoveredNodes.insert(initialState);
  _stateQueue.clear();  _stateQueue.push(initialState);

  unsigned pops = 0;
  while (_stateQueue.more()) {
    const _State state(_stateQueue.pop());

    _State::_Iterator eitr(state);
    while (eitr.more()) {

      // exclude self-loops
      if (eitr.nodeA() == state.nodeA() && eitr.nodeB() == state.nodeB()) { eitr++;  continue; }

      unsigned indexA = eitr.nodeA()->index();
      unsigned indexB = eitr.nodeB()->index();

      if (_coaccessibleNodes.present(indexA, indexB)) {
	_State nextState(eitr.nodeA(), eitr.nodeB());
	if (_discoveredNodes.insert(nextState))
	  _stateQueue.push(nextState);
      } else {
	// printf("Node (%d x %d) is on the fence.\n", indexA, indexB);
	fen->insert(indexA, indexB);
      }
      eitr++;
    }
    if (++pops % 100000 == 0) {
      printf("%d fence nodes discovered : %d nodes to pop\n",
	     fen->size(), _stateQueue.elementN());  fflush(stdout); }
  }
  printf("There are %d fence nodes.\n", fen->size());  fflush(stdout);

  return fen;
}


// ----- methods for class `FenceFinder::_StateSet' -----
//
FenceFinder::_StateSet::_StateSet(unsigned bucketN)
  : _bucketN(bucketN), _bucketN2(bucketN * 2), _indexN(0), _fence(new _State*[_bucketN]), _list(NULL) { }

FenceFinder::_StateSet::~_StateSet()
{
  clear();
  delete[] _fence;
}

void FenceFinder::_StateSet::clear()
{
  _State* current = _list;
  while (current != NULL) {
    _State* next = current->_next;
    delete current;
    current = next;
  }

  _list = NULL;  _indexN = 0;
  for (unsigned i = 0; i < _bucketN; i++)
    _fence[i] = NULL;
}

void FenceFinder::_StateSet::_rehash()
{
  delete[] _fence;  _bucketN *= 2;  _bucketN2 = _bucketN * 2;

  printf("Rehashing 'FenceFinder::_StateSet' to %d buckets.\n", _bucketN);  fflush(stdout);

  _fence = new _State*[_bucketN];
  for (unsigned i = 0; i < _bucketN; i++)
    _fence[i] = NULL;

  _State* next = _list;
  while (next != NULL) {
    
    unsigned bucketX = _hash(*next);
    next->_chain     = _fence[bucketX];
    _fence[bucketX]  = next;
    
    next = next->_next;
  }
}

bool FenceFinder::_StateSet::insert(const _State& state)
{
  if (_indexN > _bucketN2) _rehash();

  unsigned bucketX   = _hash(state);
  _State*  holder    = _fence[bucketX];
  while (holder != NULL) {
    if (*holder == state) return false;
    holder = holder->_chain;
  }

  _State*  newHolder = new _State(state, _list, _fence[bucketX]);
  _list              = newHolder;
  _fence[bucketX]    = newHolder;

  _indexN++;
  return true;
}

bool FenceFinder::_StateSet::present(const _State& state) const
{
  _State* holder = _fence[_hash(state)];
  while (holder != NULL) {
    if (*holder == state)
      return true;
    holder = holder->_chain;
  }
  return false;
}

FencePtr findFence(const WFSTFlyWeightSortedOutputPtr& wfstA, const WFSTFlyWeightSortedInputPtr& wfstB, unsigned bucketN)
{
  FenceFinder finder(wfstA, wfstB, bucketN);
  return finder.fence();
}
