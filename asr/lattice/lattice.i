//                              -*- C++ -*-
//
//                              Millennium
//                  Automatic Speech Recognition System
//                                 (asr)
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


%module lattice

#ifdef AUTODOC
%section "Lattice"
#endif

/* operator overloading */
%rename(__str__) *::puts();

%{
#include "fsm/fsm.h"
#include "lattice/lattice.h"
%}

%include typedefs.i
%include jexception.i

%import fsm/fsm.i

#ifdef AUTODOC
%subsection "Lattice", before
#endif

typedef double LogDouble;
LogDouble logAdd(LogDouble ap, LogDouble bp);


// ----- definition for class `LatticeNodeData' -----
//
%ignore LatticeNodeData;
class LatticeNodeData {
  friend class Lattice;
 public:
  LatticeNodeData();

  void write(FILE* fp) const;

  const TokenPtr& tok() const { return _tok; }

  void clearProbs();

  LogDouble forwardProb()  const { return _forwardProb;  }
  LogDouble backwardProb() const { return _backwardProb; }
};


// ----- definition for class `LatticeEdgeData` -----
//
%ignore LatticeEdgeData;
class LatticeEdgeData {
public:
  LatticeData(unsigned s, unsigned e, double ac, double lm);

  void write(FILE* fp) const;

  unsigned start() const;
  unsigned end()   const;
  double   ac()    const;
  double   lm()    const;
  double   gamma() const;

  void setGamma(double g) { _gamma = g; }
};


// ----- definition for class `Lattice` -----
// 
%ignore Lattice;
class Lattice : public WFSTransducer {
 public:
  // Lattice(LexiconPtr& statelex, LexiconPtr& inlex, LexiconPtr& outlex);
  Lattice(LexiconPtr statelex, LexiconPtr inlex, LexiconPtr outlex);
  Lattice(WFSTSortedInputPtr& wfst);

  virtual ~Lattice();

  // read lattice
  void read(const String& fileName, bool noSelfLoops = false, bool readData = false);

  // write lattice
  void write(const String& fileName = "", bool useSymbols = false, bool writeData = false);

  // rescore the lattice with given LM scale factor and penalty
  float rescore(double lmScale = 30.0, double lmPenalty = 0.0, double silPenalty = 0.0, const String& silSymbol = "SIL-m");

  // write best hypo to disk in CTM format
  void writeCTM(const String& conv, const String& channel, const String& spk, const String& utt,
		double cfrom, double score, const String& fileName = "", double frameInterval = 0.01,
		const String& endMarker = "</s>");

  void writePhoneCTM(const String& conv, const String& channel, const String& spk, const String& utt,
		     double cfrom, double score, const String& fileName = "", double frameInterval = 0.01,
		     const String& endMarker = "</s>") /* const */;

  // write best hypo to disk in HTK format
  void writeHypoHTK(const String& conv, const String& channel, const String& spk, const String& utt,
		    double cfrom, double score, const String& fileName = "", int flag = 0, double frameInterval = 0.01,
		    const String& endMarker = "</s>");

  // write best word sequence annotated with confidence scores
  void writeWordConfs(const String& fileName, const String& uttId, const String& endMarker = "</s>");

  // create a phone lattice
  WFSTSortedInputPtr createPhoneLattice(LatticePtr& lattice, double acScale);

  // prune the lattice
  void prune(double threshold = 100.0);

  // prune the lattice
  void pruneEdges(unsigned edgesN = 0);

  // purge nodes that are not on a successful path
  void purge();

  // get the 1-best hypo
  String bestHypo(bool useInputSymbols = false) /* const */;

  // calculate posterior probabilities of the links
  double gammaProbs(double acScale = 1.0, double lmScale = 12.0, double lmPenalty = 0.0,
		    double silPenalty = 0.0, const String& silSymbol = "SIL-m");

  // calculate posterior probabilities of the links with given distribution set
  double gammaProbsDist(DistribSetBasicPtr& dss, double acScale = 1.0, double lmScale = 12.0, double lmPenalty = 0.0,
			double silPenalty = 0.0, const String& silSymbol = "SIL-m");
};

%rename(Lattice_Node) 	   Lattice::Node;

%ignore Lattice::Node;
class Lattice::Node : public WFSTransducer::Node {
 public:
  Lattice::Node(unsigned idx, double cost = 0.0);
  Lattice::~Node();

  void addEdge(EdgePtr ed);
};

class LatticePtr : public WFSTransducerPtr {
 public:
  %extend {
    LatticePtr(LexiconPtr statelex, LexiconPtr inlex, LexiconPtr outlex) {
      return new LatticePtr(new Lattice(statelex, inlex, outlex));
    }
  }
  Lattice* operator->();
};

void applyConfidences(LatticePtr& A, const ConfidenceListPtr& confidenceList);

void Lattice_reportMemoryUsage();
