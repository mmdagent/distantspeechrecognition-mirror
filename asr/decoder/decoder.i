//                              -*- C++ -*-
//
//                              Millennium
//                  Automatic Speech Recognition System
//                                 (asr)
//
//  Module:  asr.decoder
//  Purpose: Token passing decoder with lattice generation.
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


%module decoder

#ifdef AUTODOC
%section "Decoder"
#endif

/* operator overloading */
%rename(__str__) *::puts();

%{
#include "fsm/fsm.h"
#include "decoder/decoder.h"
%}

%include typedefs.i
%include jexception.i

%import fsm/fsm.i
%import lattice/lattice.i

#ifdef AUTODOC
%subsection "Decoder", before
#endif


// ----- definition for class `WFSTFlyWeight' -----
//
%ignore WFSTFlyWeight;
class WFSTFlyWeight {
 public:
  WFSTFlyWeight(LexiconPtr& statelex, LexiconPtr& inlex, LexiconPtr& outlex, const String& name = "WFSTFlyWeight");
  ~WFSTFlyWeight();

  void read(const String& fileName, bool binary = false);
  void reverseRead(const String& fileName);
  void write(const String& fileName, bool binary = true, bool useSymbols = false);

  void reverse(const WFSTFlyWeightPtr& wfst);

  bool hasFinalState() const;

  LexiconPtr& stateLexicon()  { return _stateLexicon; }
  LexiconPtr& inputLexicon()  { return _inputLexicon; }
  LexiconPtr& outputLexicon() { return _outputLexicon; }
};

class WFSTFlyWeightPtr {
 public:
  %extend {
    WFSTFlyWeightPtr(LexiconPtr& statelex, LexiconPtr& inlex, LexiconPtr& outlex, const String& name = "WFSTFlyWeight") {
      return new WFSTFlyWeightPtr(new WFSTFlyWeight(statelex, inlex, outlex, name));
    }
  }
  WFSTFlyWeight* operator->();
};

void WFSTFlyWeight_reportMemoryUsage();


// ----- definition for class `Decoder' -----
// 
%ignore Decoder;
class Decoder {
 public:
  Decoder(DistribSetPtr& dist,
          double beam = 100.0, double lmScale = 12.0, double lmPenalty = 0.0,
	  double silPenalty = 0.0, const String& silSymbol = "SIL-m", const String& eosSymbol = "</s>",
          unsigned heapSize = 5000, unsigned topN = 0, bool generateLattice = true);
  ~Decoder();

  // specify decoding network
  void set(WFSTransducerPtr& wfst);

  // decode current utterance
  double decode(bool verbose = false);

  // back trace to get 1-best hypo
  String bestHypo(bool useInputSymbols = false);

  // back trace to get best distribPath
  DistribPathPtr bestPath();

  // number of final states for current utterance
  unsigned finalStatesN() const;

  // generate a lattice
  LatticePtr lattice();

  // set upper limit on memory allocated for tokens
  void setTokenMemoryLimit(unsigned limit);

  // write best hypo to disk in CTM format
  void writeCTM(const String& conv, const String& channel, const String& spk, const String& utt,
		double cfrom, double score, const String& fileName = "", double frameInterval = 0.01);

  // write GMM labels for 1-best hypo in CTM format
  void writeGMM(const String& conv, const String& channel, const String& spk, const String& utt,
		double cfrom, double score, const String& fileName = "", double frameInterval = 0.01);

  // token reach the end state
  bool traceBackSucceeded() const;
};

class DecoderPtr {
 public:
  %extend {
    DecoderPtr(DistribSetPtr& dist,
               double beam = 100.0, double lmScale = 12.0, double lmPenalty = 0.0,
	       double silPenalty = 0.0, const String& silSymbol = "SIL-m", const String& eosSymbol = "</s>",
               unsigned heapSize = 5000, unsigned topN = 0, bool generateLattice = true) {
      return new DecoderPtr(new Decoder(dist, beam, lmScale, lmPenalty, silPenalty, silSymbol, eosSymbol, heapSize, topN, generateLattice));
    }
  }

  Decoder* operator->();
};


// ----- definition for class `DecoderFlyWeight' -----
// 
%ignore DecoderFlyWeight;
class DecoderFlyWeight {
 public:
  DecoderFlyWeight(DistribSetPtr& dist,
                   double beam = 100.0, double lmScale = 12.0, double lmPenalty = 0.0,
		   double silPenalty = 0.0, const String& silSymbol = "SIL-m", const String& eosSymbol = "</s>",
                   unsigned heapSize = 5000, unsigned topN = 0, bool generateLattice = true);
  ~DecoderFlyWeight();

  // specify decoding network
  void set(WFSTFlyWeightPtr& wfst);

  // decode current utterance
  double decode(bool verbose = false);

  // back trace to get 1-best hypo
  String bestHypo(bool useInputSymbols = false);

  // back trace to get best distribPath
  DistribPathPtr bestPath();

  // number of final states for current utterance
  unsigned finalStatesN() const;

  // generate a lattice
  LatticePtr lattice();

  // set upper limit on memory allocated for tokens
  void setTokenMemoryLimit(unsigned limit);

  // write best hypo to disk in CTM format
  void writeCTM(const String& conv, const String& channel, const String& spk, const String& utt,
		double cfrom, double score, const String& fileName = "", double frameInterval = 0.01);

  // write GMM labels for 1-best hypo in CTM format
  void writeGMM(const String& conv, const String& channel, const String& spk, const String& utt,
		double cfrom, double score, const String& fileName = "", double frameInterval = 0.01);

  // token reach the end state
  bool traceBackSucceeded() const;

  // specify beam width
  void setBeam(double beam);
};

class DecoderFlyWeightPtr {
 public:
  %extend {
    DecoderFlyWeightPtr(DistribSetPtr& dist,
                        double beam = 100.0, double lmScale = 12.0, double lmPenalty = 0.0,
			double silPenalty = 0.0, const String& silSymbol = "SIL-m", const String& eosSymbol = "</s>",
                        unsigned heapSize = 5000, unsigned topN = 0, bool generateLattice = true) {
      return new DecoderFlyWeightPtr(new DecoderFlyWeight(dist, beam, lmScale, lmPenalty, silPenalty, silSymbol, eosSymbol, heapSize, topN, generateLattice));
    }
  }

  DecoderFlyWeight* operator->();
};


// ----- definition for class `DecoderWordTrace' -----
// 
%ignore DecoderWordTrace;
class DecoderWordTrace {
 public:
  DecoderWordTrace(DistribSetPtr& dist,
		   double beam = 100.0, double lmScale = 12.0, double lmPenalty = 0.0,
		   double silPenalty = 0.0, const String& silSymbol = "SIL-m", const String& eosSymbol = "</s>",
		   unsigned heapSize = 5000, unsigned topN = 0, double epsilon = 0.0, unsigned validEndN = 30,
		   bool generateLattice = true, unsigned propagateN = 5, bool fastHash = false, bool insertSilence = false);
  ~DecoderWordTrace();

  // specify decoding network
  void set(WFSTFlyWeightSortedOutputPtr& wfst);

  // decode current utterance
  double decode(bool verbose = false);

  // back trace to get 1-best hypo
  String bestHypo(bool useInputSymbols = false);

  // back trace to get best distribPath
  DistribPathPtr bestPath();

  // number of final states for current utterance
  unsigned finalStatesN() const;

  // generate a lattice
  LatticePtr lattice();

  // set upper limit on memory allocated for tokens
  void setTokenMemoryLimit(unsigned limit);

  // write best hypo to disk in CTM format
  void writeCTM(const String& conv, const String& channel, const String& spk, const String& utt,
		double cfrom, double score, const String& fileName = "", double frameInterval = 0.01);

  // write GMM labels for 1-best hypo in CTM format
  void writeGMM(const String& conv, const String& channel, const String& spk, const String& utt,
		double cfrom, double score, const String& fileName = "", double frameInterval = 0.01);

  // token reach the end state
  bool traceBackSucceeded() const;
};

class DecoderWordTracePtr {
 public:
  %extend {
    DecoderWordTracePtr(DistribSetPtr& dist,
			double beam = 100.0, double lmScale = 12.0, double lmPenalty = 0.0,
			double silPenalty = 0.0, const String& silSymbol = "SIL-m", const String& eosSymbol = "</s>",
			unsigned heapSize = 5000, unsigned topN = 0, double epsilon = 0.0, unsigned validEndN = 30,
			bool generateLattice = true, unsigned propagateN = 5, bool fastHash = false, bool insertSilence = false) {
      return new DecoderWordTracePtr(new DecoderWordTrace(dist, beam, lmScale, lmPenalty, silPenalty, silSymbol,
							  eosSymbol, heapSize, topN, epsilon, validEndN, generateLattice,
							  propagateN, fastHash, insertSilence));
    }
  }

  DecoderWordTrace* operator->();
};


// ----- definition for class `DecoderFastComposition' -----
// 
%ignore DecoderFastComposition;
class DecoderFastComposition : public DecoderWordTrace {
 public:
  DecoderFastComposition(DistribSetPtr& dist,
			 double beam = 100.0, double lmScale = 12.0, double lmPenalty = 0.0,
			 double silPenalty = 0.0, const String& silSymbol = "SIL-m", const String& eosSymbol = "</s>",
			 unsigned heapSize = 5000, unsigned topN = 0, bool generateLattice = true,
			 unsigned propagateN = 5, bool fastHash = false, bool insertSilence = false,
			 bool pushWeights = false);
  ~DecoderFastComposition();

  // specify decoding networks
  void setAB(WFSTFlyWeightSortedOutputPtr& A, WFSTFlyWeightSortedInputPtr& B);

  // specify fence
  void setFence(FencePtr& fence);

  // specify beam width
  void setBeam(double beam);

  // decode current utterance
  virtual double decode(bool verbose = false);
};

class DecoderFastCompositionPtr : public DecoderWordTracePtr {
 public:
  %extend {
    DecoderFastCompositionPtr(DistribSetPtr& dist,
			      double beam = 100.0, double lmScale = 12.0, double lmPenalty = 0.0,
			      double silPenalty = 0.0, const String& silSymbol = "SIL-m", const String& eosSymbol = "</s>",
			      unsigned heapSize = 5000, unsigned topN = 0, bool generateLattice = true,
			      unsigned propagateN = 5, bool fastHash = false, bool insertSilence = false,
			      bool pushWeights = false) {
      return new DecoderFastCompositionPtr(new DecoderFastComposition(dist, beam, lmScale, lmPenalty, silPenalty, silSymbol,
								      eosSymbol, heapSize, topN, generateLattice,
								      propagateN, fastHash, insertSilence, pushWeights));
    }
  }

  DecoderFastComposition* operator->();
};


// ----- definition for class `WFSTFlyWeightSortedInput' -----
// 
%ignore WFSTFlyWeightSortedInput;
class WFSTFlyWeightSortedInput : public WFSTFlyWeight {
 public:
  WFSTFlyWeightSortedInput(LexiconPtr& statelex, LexiconPtr& inlex, LexiconPtr& outlex,
			   const String& name = "WFSTFlyWeight");

  void hash();
};

class WFSTFlyWeightSortedInputPtr : public WFSTFlyWeightPtr {
 public:
  %extend {
    WFSTFlyWeightSortedInputPtr(LexiconPtr& statelex, LexiconPtr& inlex, LexiconPtr& outlex,
				const String& name = "Fly Weight Sorted Input Transducer") {
      return new WFSTFlyWeightSortedInputPtr(new WFSTFlyWeightSortedInput(statelex, inlex, outlex, name));
    }
  }
  WFSTFlyWeightSortedInput* operator->();
};


// ----- definition for class `WFSTFlyWeightSortedOutput' -----
// 
%ignore WFSTFlyWeightSortedOutput;
class WFSTFlyWeightSortedOutput : public WFSTFlyWeight {
 public:
  WFSTFlyWeightSortedOutput(LexiconPtr& statelex, LexiconPtr& inlex, LexiconPtr& outlex,
			   const String& name = "WFSTFlyWeight");


  // initial state
  // WFSTFlyWeightSortedOutputNodePtr initial();
};

class WFSTFlyWeightSortedOutputPtr : public WFSTFlyWeightPtr {
 public:
  %extend {
    WFSTFlyWeightSortedOutputPtr(LexiconPtr& statelex, LexiconPtr& inlex, LexiconPtr& outlex,
				const String& name = "Fly Weight Sorted Input Transducer") {
      return new WFSTFlyWeightSortedOutputPtr(new WFSTFlyWeightSortedOutput(statelex, inlex, outlex, name));
    }
  }
  WFSTFlyWeightSortedOutput* operator->();
};


// ----- definition for class `Fence' -----
// 
%ignore Fence;
class Fence {
public:
  Fence(unsigned bucketN = 4000001, const String& fileName = "", float rehashFactor = 2.0) { }

  void insert(unsigned indexA, unsigned indexB);
  bool present(unsigned indexA, unsigned indexB);

  void clear();

  unsigned size() const;

  void read(const String& fileName);
  void write(const String& fileName) const;
};

class FencePtr {
 public:
  %extend {
    FencePtr(unsigned bucketN = 4000001, const String& fileName = "") {
      return new FencePtr(new Fence(bucketN, fileName));
    }
  }
  Fence* operator->();
};

FencePtr findFence(const WFSTFlyWeightSortedOutputPtr& wfstA, const WFSTFlyWeightSortedInputPtr& wfstB, unsigned bucketN = 4000001);
