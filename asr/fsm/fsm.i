//                              -*- C++ -*-
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


%module fsm

#ifdef AUTODOC
%section "FSM"
#endif

/* operator overloading */
%rename(__str__) *::puts();

%include typedefs.i
%include jexception.i

typedef unsigned short UnShrt;

%{
#include "fsm.h"
%}

#ifdef AUTODOC
%subsection "FSM", before
#endif


// ----- definition for class `WFSAcceptor' -----
// 
%ignore WFSAcceptor;
class WFSAcceptor {
 public:
  WFSAcceptor(LexiconPtr& inlex);
  WFSAcceptor(LexiconPtr& statelex, LexiconPtr& inlex, const String& name);
  ~WFSAcceptor();

  // initial state
  WFSAcceptorNodePtr initial();

  // input lexicon
  LexiconPtr inputLexicon();

  // state lexicon
  LexiconPtr stateLexicon();

  // read a WFSAcceptor
  void read(const String fileName, bool noSelfLoops = false);

  // write a WFSAcceptor
  void write(const String fileName = "", bool useSymbols = false);

  // clear a WFSAcceptor
  void clear();

  // returns true if acceptor has a final state
  bool hasFinalState();

  // search for epsilon cycles
  bool epsilonCycle(const WFSAcceptorNodePtr node);

  // print network dimensions
  void printStats() const;
};

class WFSAcceptorPtr {
 public:
  %extend {
    WFSAcceptorPtr(LexiconPtr& inlex) {
      return new WFSAcceptorPtr(new WFSAcceptor(inlex));
    }
    WFSAcceptorPtr(LexiconPtr& statelex, LexiconPtr& inlex, const String& name = "WFSAcceptor") {
      return new WFSAcceptorPtr(new WFSAcceptor(statelex, inlex, name));
    }
  }
  WFSAcceptor* operator->();
};


// ----- definition for class `WFSTransducer' -----
// 
%ignore WFSTransducer;
class WFSTransducer : public WFSAcceptor {
 public:
  WFSTransducer(WFSAcceptorPtr& wfsa, const String& name = "WFSTransducer");
  WFSTransducer(LexiconPtr statelex, LexiconPtr inlex, LexiconPtr outlex, const String& name = "WFSTransducer");
  ~WFSTransducer();

  // initial state
  WFSTransducerNodePtr initial();

  // output lexicon
  LexiconPtr outputLexicon();

  // search for epsilon cycles
  bool epsilonCycle(const WFSTransducerNodePtr node);

  // print network dimensions
  void printStats() const;

  // purge unreferenced nodes
  void purgeUnique(unsigned count = 10000);

  // reverse a given transducer
  void reverse(const WFSTransducerPtr& wfst);

  // replace one symbol with another on all arcs
  void replaceSymbol(const String& fromSym, const String& toSym, bool input = true);

  // initialize transducer from a word string
  void fromWords(const String& words, const String& end = "</s>", const String& filler = "", bool clear = true, float logprob = 0.0,
		 const String& garbageWord = "", float garbageBeginPenalty = 0.0, float garbageEndPenalty = 0.0);

  // return string associated with shortest path
  String bestString();

  // read a transducer in reverse order
  virtual void reverseRead(const String& fileName, bool noSelfLoops = false);
};

class WFSTransducerPtr : public WFSAcceptorPtr {
 public:
  %extend {
    WFSTransducerPtr(WFSAcceptorPtr& wfsa, const String& name = "WFSTransducer") {
      return new WFSTransducerPtr(new WFSTransducer(wfsa, name));
    }

    WFSTransducerPtr(LexiconPtr statelex, LexiconPtr inlex, LexiconPtr outlex, const String& name = "Transducer") {
      return new WFSTransducerPtr(new WFSTransducer(statelex, inlex, outlex, name));
    }
  }
  WFSTransducer* operator->();
};

WFSTransducerPtr reverse(const WFSTransducerPtr& wfst);

void pushWeights(WFSTransducerPtr& wfst);


// ----- definition for class `ConfidenceList' -----
// 
%ignore ConfidenceList;
class ConfidenceList {
 public:
  ConfidenceList(const String& nm);
  ~ConfidenceList();

  const String& name() const;

  String word(int depth) const;

  void binarize(float threshold = 0.5);

  void write(const String& fileName = "");
};

class ConfidenceListPtr {
 public:
  %extend {
    ConfidenceListPtr(const String& nm) {
      return new ConfidenceListPtr(new ConfidenceList(nm));
    }
  }
  ConfidenceList* operator->();
};


// ----- definition for class `WFSTSortedInput' -----
// 
%ignore WFSTSortedInput;
class WFSTSortedInput : public WFSTransducer {
 public:
  WFSTSortedInput(const WFSAcceptorPtr& wfsa);
  WFSTSortedInput(const WFSTransducerPtr& A, bool dynamic = false);
  WFSTSortedInput(LexiconPtr& statelex, LexiconPtr& inlex, LexiconPtr& outlex,
                  bool dynamic = false, const String& name = "WFST Sorted Input");
  ~WFSTSortedInput();

  // initial state
  // WFSTSortedInputNodePtr initial();

  // initialize transducer from confidence scores
  ConfidenceListPtr fromConfs(const String& words, const String& end = "</s>", const String& filler = "");
};

class WFSTSortedInputPtr : public WFSTransducerPtr {
 public:
  %extend {
    WFSTSortedInputPtr(const WFSAcceptorPtr& wfsa) {
      return new WFSTSortedInputPtr(new WFSTSortedInput(wfsa));
    }

    WFSTSortedInputPtr(const WFSTransducerPtr& A, bool dynamic = false) {
      return new WFSTSortedInputPtr(new WFSTSortedInput(A, dynamic));
    }

    WFSTSortedInputPtr(LexiconPtr& statelex, LexiconPtr& inlex, LexiconPtr& outlex,
		       bool dynamic = false, const String& name = "Sorted Input Transducer") {
      return new WFSTSortedInputPtr(new WFSTSortedInput(statelex, inlex, outlex, dynamic, name));
    }
  }
  WFSTSortedInput* operator->();
};


// ----- definition for class `WFSTEpsilonRemoval' -----
// 
%ignore WFSTEpsilonRemoval;
class WFSTEpsilonRemoval : public WFSTSortedInput {
 public:
  WFSTEpsilonRemoval(WFSTransducerPtr& wfst);
  ~WFSTEpsilonRemoval();
};

class WFSTEpsilonRemovalPtr : public WFSTSortedInputPtr {
 public:
  %extend {
    WFSTEpsilonRemovalPtr(WFSTransducerPtr& wfst) {
      return new WFSTEpsilonRemovalPtr(new WFSTEpsilonRemoval(wfst));
    }
  }
  WFSTEpsilonRemoval* operator->();
};

WFSTEpsilonRemovalPtr removeEpsilon(WFSTransducerPtr& wfst);


// ----- definition for class `WFSTEpsilonMinusOneRemoval' -----
// 
%ignore WFSTEpsilonMinusOneRemoval;
class WFSTEpsilonMinusOneRemoval : public WFSTEpsilonRemoval {
 public:
  WFSTEpsilonMinusOneRemoval(WFSTransducerPtr& wfst);
  ~WFSTEpsilonMinusOneRemoval();
};

class WFSTEpsilonMinusOneRemovalPtr : public WFSTEpsilonRemovalPtr {
 public:
  %extend {
    WFSTEpsilonMinusOneRemovalPtr(WFSTransducerPtr& wfst) {
      return new WFSTEpsilonMinusOneRemovalPtr(new WFSTEpsilonMinusOneRemoval(wfst));
    }
  }
  WFSTEpsilonMinusOneRemoval* operator->();
};

WFSTEpsilonMinusOneRemovalPtr removeEpsilonMinusOne(WFSTransducerPtr& wfst);

// ----- definition for class `WFSTProjection' -----
// 
%ignore WFSTProjection;
class WFSTProjection : public WFSTSortedInput {
 public:
  WFSTProjection(WFSTransducerPtr& wfst, Side side = Input);
  ~WFSTProjection();
};

class WFSTProjectionPtr : public WFSTSortedInputPtr {
 public:
  %extend {
    WFSTProjectionPtr(WFSTransducerPtr& wfst, WFSTProjection::Side side = WFSTProjection::Output) {
      return new WFSTProjectionPtr(new WFSTProjection(wfst, side));
    }
  }
  WFSTProjection* operator->();
};

WFSTProjectionPtr project(WFSTransducerPtr& wfst, const String& side = "Output");


// ----- definition for class `WFSTRemoveEndMarkers' -----
// 
%ignore WFSTRemoveEndMarkers;
class WFSTRemoveEndMarkers : public WFSTSortedInput {
 public:
  WFSTRemoveEndMarkers(WFSTSortedInputPtr& A, const String& end = "#", bool dynamic = false,
		       bool input = true, const String& name = "WFST Remove Word End Markers");

  ~WFSTRemoveEndMarkers();
};

class WFSTRemoveEndMarkersPtr : public WFSTSortedInputPtr {
 public:
  %extend {
    WFSTRemoveEndMarkersPtr(WFSTSortedInputPtr& A, const String& end = "#", bool dynamic = false,
			    bool input = true, const String& name = "WFST Remove Word End Markers")
    {
      return new WFSTRemoveEndMarkersPtr(new WFSTRemoveEndMarkers(A, end, dynamic, input, name));
    }
  }
  WFSTRemoveEndMarkers* operator->();
};

WFSTSortedInputPtr removeEndMarkers(WFSTSortedInputPtr& A, bool input = true);


// ----- definition for class `WFSTSortedOutput' -----
// 
%ignore WFSTSortedOutput;
class WFSTSortedOutput : public WFSTransducer {
 public:
  WFSTSortedOutput(const WFSAcceptorPtr& wfsa, bool convertWFSA = true);
  WFSTSortedOutput(const WFSTransducerPtr& A);
  WFSTSortedOutput(LexiconPtr& statelex, LexiconPtr& inlex, LexiconPtr& outlex, const String& name = "WFST Sorted Output");
  ~WFSTSortedOutput();

  // initial state
  // WFSTSortedOutputNodePtr initial();
};

class WFSTSortedOutputPtr : public WFSTransducerPtr {
 public:
  %extend {
    WFSTSortedOutputPtr(const WFSAcceptorPtr& wfsa, bool convertWFSA = true) {
      return new WFSTSortedOutputPtr(new WFSTSortedOutput(wfsa, convertWFSA));
    }
    WFSTSortedOutputPtr(const WFSTransducerPtr& A) {
      return new WFSTSortedOutputPtr(new WFSTSortedOutput(A));
    }
    WFSTSortedOutputPtr(LexiconPtr& statelex, LexiconPtr& inlex, LexiconPtr& outlex, const String& name = "Sorted Output Transducer") {
      return new WFSTSortedOutputPtr(new WFSTSortedOutput(statelex, inlex, outlex, name));
    }
  }
  WFSTSortedOutput* operator->();
};


// ----- definition for class `WFSTComposition' -----
// 
%ignore WFSTComposition;
class WFSTComposition : public WFSTSortedInput {
 public:
  WFSTComposition(WFSTSortedOutputPtr& A, WFSTSortedInputPtr& B, LexiconPtr& stateLex,
                  bool dynamic = false, const String& name = "WFST Composition");
  ~WFSTComposition();

  // initial state
  // WFSTCompositionNodePtr initial();
};

class WFSTCompositionPtr : public WFSTSortedInputPtr {
 public:
  %extend {
    WFSTCompositionPtr(WFSTSortedOutputPtr& A, WFSTSortedInputPtr& B, LexiconPtr& stateLex,
		       const String& semiring = "Tropical", bool dynamic = false,
                       const String& name = "WFST Composition") {
      if (semiring == "Tropical")
	return new WFSTCompositionPtr(new WFSTComp<TropicalSemiring>(A, B, stateLex, dynamic, name));
      else if (semiring == "LogProb")
	return new WFSTCompositionPtr(new WFSTComp<LogProbSemiring>(A, B, stateLex, dynamic, name));
      else
	throw jtype_error("Could not identify semiring %s", semiring.c_str());
    }
  }
  WFSTComposition* operator->();
};

void WFSTComposition_reportMemoryUsage();

WFSTSortedInputPtr compose(WFSTSortedOutputPtr& sortedA, WFSTSortedInputPtr& sortedB,
			   const String& semiring = "Tropical", const String& name = "Composition");

void breadthFirstSearch(WFSTransducerPtr& A, unsigned purgeCount = 10000);

void breadthFirstWrite(WFSTransducerPtr& A, const String& fileName, bool useSymbols, unsigned purgeCount = 10000);


// ----- definition for class `WFSTDeterminization' -----
// 
%ignore WFSTDeterminization;
class WFSTDeterminization : public WFSTSortedInput {
 public:
  WFSTDeterminization(WFSTSortedInputPtr& A, LexiconPtr& stateLex, bool dynamic = false);
  ~WFSTDeterminization();

  // initial state
  // WFSTDeterminizationNodePtr initial();
};

class WFSTDeterminizationPtr : public WFSTSortedInputPtr {
 public:
  %extend {
    WFSTDeterminizationPtr(WFSTSortedInputPtr& A, LexiconPtr& stateLex,
			   const String& semiring = "Tropical", bool dynamic = false) {
      if (semiring == "Tropical")
	return new WFSTDeterminizationPtr(new WFSTDet<TropicalSemiring>(A, stateLex, dynamic));
      else if (semiring == "LogProb")
	return new WFSTDeterminizationPtr(new WFSTDet<LogProbSemiring>(A, stateLex, dynamic));
      else
	throw jtype_error("Could not identify semiring %s", semiring.c_str());
    }
  }
  WFSTDeterminization* operator->();
};

// WFSTDeterminizationPtr determinize(WFSTransducerPtr& A, const String& semiring = "Tropical");

WFSTDeterminizationPtr determinize(WFSTSortedInputPtr& sortedA, const String& semiring = "Tropical", unsigned count = 10000);

void WFSTDeterminization_reportMemoryUsage();


// ----- definition for class `WFSTIndexed' -----
// 
%ignore WFSTIndexed;
class WFSTIndexed : public WFSTSortedOutput {
 public:
  WFSTIndexed(const WFSAcceptorPtr& wfsa, bool convertWFSA = true);

  WFSTIndexed(LexiconPtr& statelex, LexiconPtr& inlex,
	      const String& grammarFile = "", const String& name = "WFST Grammar");

  WFSTIndexed(LexiconPtr& statelex, LexiconPtr& inlex, LexiconPtr& outlex, const String& name = "WFST Grammar");

  ~WFSTIndexed();

  void setLattice(WFSTSortedInputPtr& lattice);

  // initial state
  // WFSTIndexedNodePtr initial();
};

class WFSTIndexedPtr : public WFSTSortedOutputPtr {
 public:
  %extend {
    WFSTIndexedPtr(const WFSAcceptorPtr& wfsa, bool convertWFSA = true) {
      return new WFSTIndexedPtr(new WFSTIndexed(wfsa, convertWFSA));
    }

    WFSTIndexedPtr(LexiconPtr& statelex, LexiconPtr& inlex,
		   const String& grammarFile = "", const String& name = "WFST Grammar") {
      return new WFSTIndexedPtr(new WFSTIndexed(statelex, inlex, grammarFile, name));
    }

    WFSTIndexedPtr(LexiconPtr& statelex, LexiconPtr& inlex, LexiconPtr& outlex, const String& name = "WFST Grammar") {
      return new WFSTIndexedPtr(new WFSTIndexed(statelex, inlex, outlex, name));
    }

  }
  WFSTIndexed* operator->();
};


// ----- definition for class `WFSTLexicon' -----
// 
%ignore WFSTLexicon;
class WFSTLexicon : public WFSTSortedOutput {
 public:
  WFSTLexicon(LexiconPtr& statelex, LexiconPtr& inlex, LexiconPtr& outlex,
	      const String& sil = "SIL", const String& breath = "{+BREATH+:WB}",
	      const String& sos = "", const String& eos = "</s>", const String& end = "#",
	      const String& backOff = "", double backOffWeight = 3.0,
	      unsigned maxWordN = 65535, const String& lexiconFile = "",
	      bool epsilonToBranch = false, const String& name = "WFST Lexicon");
  ~WFSTLexicon();

  // read the lexicon
  virtual void read(const String& fileName);

  // determine the edit distance between two words
  unsigned editDistance(unsigned word1, unsigned word2) const;

  // set the grammar
  void setGrammar(WFSTSortedInputPtr& grammar);

  // initial state
  // WFSTLexiconNodePtr initial();
};

class WFSTLexiconPtr : public WFSTSortedOutputPtr {
 public:
  %extend {
    WFSTLexiconPtr(LexiconPtr& statelex, LexiconPtr& inlex, LexiconPtr& outlex,
	      const String& sil = "SIL", const String& breath = "{+BREATH+:WB}",
	      const String& sos = "", const String& eos = "</s>", const String& end = "#",
	      const String& backOff = "", double backOffWeight = 3.0,
		   unsigned maxWordN = 65535, const String& lexiconFile = "", bool epsilonToBranch = false,
	      const String& name = "WFST Lexicon") {
      return new WFSTLexiconPtr(new WFSTLexicon(statelex, inlex, outlex, sil, breath, sos, eos, end, backOff,
						backOffWeight, maxWordN, lexiconFile, epsilonToBranch, name));
    }
  }
  WFSTLexicon* operator->();
};


// ----- definition for class `WFSTContextDependency' -----
// 
%ignore WFSTContextDependency;
class WFSTContextDependency : public WFSTSortedOutput {
 public:
  WFSTContextDependency(LexiconPtr& statelex, LexiconPtr& inlex, WFSTSortedInputPtr& dict,
			unsigned contextLen = 1, const String& sil = "SIL", const String& eps = "eps", const String& eos = "</s>",
			const String& end = "#", const String& wb = "WB", const String& name = "WFST Context Dependency");
  ~WFSTContextDependency();

  // initial state
  // WFSTContextDependencyNodePtr initial();
};

class WFSTContextDependencyPtr : public WFSTSortedOutputPtr {
 public:
  %extend {
    WFSTContextDependencyPtr(LexiconPtr& statelex, LexiconPtr& inlex, WFSTSortedInputPtr& dict,
			     unsigned contextLen = 1, const String& sil = "SIL", const String& eps = "eps", const String& eos = "</s>",
			     const String& end = "#", const String& wb = "WB", const String& name = "WFST Context Dependency") {
      return new WFSTContextDependencyPtr(new WFSTContextDependency(statelex, inlex, dict, contextLen, sil, eps, eos,
								    end, wb, name));
    }
  }
  WFSTContextDependency* operator->();
};


// ----- definition for class `WFSTHiddenMarkovModel' -----
// 
%ignore WFSTHiddenMarkovModel;
class WFSTHiddenMarkovModel : public WFSTSortedOutput {
 public:
  WFSTHiddenMarkovModel(LexiconPtr& statelex, LexiconPtr& inlex, LexiconPtr& outlex, const DistribTreePtr& distribTree,
			bool caching = false, const String& sil = "SIL", const String& eps = "eps", const String& end = "#",
			const String& name = "WFST hidden Markov model");
  ~WFSTHiddenMarkovModel();

  // initial state
  // WFSTHiddenMarkovModelNodePtr initial();

  void setContext(WFSTSortedInputPtr& context);
};

class WFSTHiddenMarkovModelPtr : public WFSTSortedOutputPtr {
 public:
  %extend {
    WFSTHiddenMarkovModelPtr(LexiconPtr& statelex, LexiconPtr& inlex, LexiconPtr& outlex, const DistribTreePtr& distribTree,
			     bool caching = false, const String& sil = "SIL", const String& eps = "eps", const String& end = "#",
			     const String& name = "WFST hidden Markov model") {
      return new WFSTHiddenMarkovModelPtr(new WFSTHiddenMarkovModel(statelex, inlex, outlex, distribTree, caching, sil, eps, end, name));
    }
  }
  WFSTHiddenMarkovModel* operator->();
};


// ----- definition for class `WFSTCompare' -----
// 
%ignore WFSTCompare;
class WFSTCompare : public WFSTSortedInput {
 public:
  WFSTCompare(LexiconPtr& statelex, LexiconPtr& inlex, LexiconPtr& outlex, WFSTSortedInputPtr& comp);
  ~WFSTCompare();

  // initial state
  // WFSTCompareNodePtr initial();
};

class WFSTComparePtr : public WFSTSortedInputPtr {
 public:
  %extend {
    WFSTComparePtr(LexiconPtr& statelex, LexiconPtr& inlex, LexiconPtr& outlex, WFSTSortedInputPtr& comp) {
      return new WFSTComparePtr(new WFSTCompare(statelex, inlex, outlex, comp));
    }
  }
  WFSTCompare* operator->();
};

// ----- definition for class `WFSTCombinedHC' -----
// 
%ignore WFSTCombinedHC;
class WFSTCombinedHC : public WFSTSortedInput {
public:
  WFSTCombinedHC(LexiconPtr& distribLexicon, DistribTreePtr& distribTree,
		 LexiconPtr& stateLexicon, LexiconPtr& phoneLexicon,
		 unsigned endN = 1, const String& sil = "SIL", /*const unsigned silStates = 4, */
		 const String& eps = "eps", const String& end = "#",
		 const String& eos = "</s>", bool correct = true, unsigned hashKeys = 1,
		 bool approximateMatch = false, bool dynamic = false);
  

  ~WFSTCombinedHC();
};

class WFSTCombinedHCPtr : public WFSTSortedInputPtr {
public:
  %extend {
    WFSTCombinedHCPtr(LexiconPtr& distribLexicon, DistribTreePtr& distribTree, 
		      LexiconPtr& stateLexicon, LexiconPtr& phoneLexicon,
		      unsigned endN = 1, const String& sil = "SIL", /*const unsigned silStates = 4, */
		      const String& eps = "eps", const String& end = "#",
		      const String& sos = "<s>", const String& eos = "</s>", const String& backoff = "%",
		      const String& pad = "PAD", bool correct = true, unsigned hashKeys = 1,
		      bool approximateMatch = false, bool dynamic = false) {
      
      return new WFSTCombinedHCPtr(new WFSTCombinedHC(distribLexicon, distribTree, stateLexicon, phoneLexicon, 
						      endN, sil, eps, end, sos, eos, backoff, pad, correct, hashKeys,
						      approximateMatch, dynamic));
    }
  }
  WFSTCombinedHC* operator->();
};


// build context-dependency transducer
WFSTransducerPtr
buildContextDependencyTransducer(LexiconPtr& inputLexicon, WFSTransducerPtr& dict, unsigned contextLen = 1,
                                 const String& sil = "SIL", const String& eps = "eps", const String& end = "#",
                                 const String& wb = "WB", const String& eos = "</s>");


// build a hidden Markov model transducer
WFSTransducerPtr
buildHiddenMarkovModelTransducer(LexiconPtr& inputLexicon, LexiconPtr& outputLexicon,
				 const DistribTreePtr& distribTree, unsigned noEnd = 1,
				 const String& sil = "SIL", const String& eps = "eps",
				 const String& end = "#");

// build a determinized, combined 'HC' transducer
WFSTDeterminizationPtr
buildDeterminizedHC(LexiconPtr& distribLexicon, DistribTreePtr& distribTree, LexiconPtr& phoneLexicon,
		    unsigned endN = 1, const String& sil = "SIL", const String& eps = "eps", const String& end = "#",
		    const String& sos = "<s>", const String& eos = "</s>", const String& backoff = "%",
		    const String& pad = "PAD", bool correct = true, unsigned hashKeys = 1, bool approximateMatch = false,
		    bool dynamic = false, unsigned count = 10000);

// build a combined 'HC' transducer
WFSTransducerPtr
buildHC(LexiconPtr& distribLexicon, DistribTreePtr& distribTree, LexiconPtr& phoneLexicon,
	unsigned endN = 1, const String& sil = "SIL", const String& eps = "eps", const String& end = "#",
	const String& sos = "<s>", const String& eos = "</s>", const String& backoff = "%",
	const String& pad = "PAD", bool correct = true, unsigned hashKeys = 1, bool approximateMatch = false,
	bool dynamic = false);

// minimize a weighted finite state acceptor
WFSAcceptorPtr minimizeFSA(const WFSAcceptorPtr& fsa);

// encode a weighted finite state transducer into an unweighted acceptor
WFSAcceptorPtr encodeWFST(WFSTransducerPtr& wfst);

// decode an unweighted finite state acceptor into a weighted transducer
WFSTSortedInputPtr decodeFSA(LexiconPtr& inputLexicon, LexiconPtr& outputLexicon, WFSAcceptorPtr& fsa);

// remove nodes for which there is no path to an end state
WFSTSortedInputPtr purgeWFST(const WFSTransducerPtr& wfst);
