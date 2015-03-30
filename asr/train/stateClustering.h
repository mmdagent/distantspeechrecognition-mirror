//                              -*- C++ -*-
//
//                              Millennium
//                   Distant Speech Recognition System
//                                 (dsr)
//
//  Module:  asr.train
//  Purpose: Speaker adaptation parameter estimation and conventional
//           ML and discriminative HMM training with state clustering.
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

#ifndef _stateClustering_h_
#define _stateClustering_h_

#include "train/codebookTrain.h"
#include "train/distribTrain.h"
#include "dictionary/distribTree.h"
#include "fsm/fsm.h"


// ----- definition for class `ClusterDistribTree' -----
//
class CodebookClusterDistribTree;
class MixtureWeightClusterDistribTree;

class ClusterDistribTree : public DistribTree {
protected:
  class Node : public DistribTree::Node {

    friend class ClusterDistribTree;
    friend class CodebookClusterDistribTree;
    friend class MixtureWeightClusterDistribTree;

  public:
    Node(PhonesSetPtr& ps, DistribTree* dt,
	 const String& nm, const String& ques,
	 const String& left, const String& right, const String& unknown, const String& leaf);

    void distributeContexts(LexiconPtr& lexicon);

    void writeContexts(FILE* fp);

  private:
    void _insertQuestion(const Question& question);
  };

  typedef Inherit<Node, DistribTree::NodePtr>	NodePtr;

  class SplitCandidate {
  public:
    SplitCandidate(double s, NodePtr& n, const Question& q)
      : _score(s), _node(n), _question(q) { }

    double score() const { return _score; }
    NodePtr& node() { return _node; }
    const Question& question() const { return _question; }

  private:
    double					_score;
    NodePtr					_node;
    Question					_question;
  };

  class SplitHeap {

    class LessThan {    // function object for 'SplitCandidate' sorting
    public:
      bool operator()(const SplitCandidate& first, const SplitCandidate& second) {
	return first.score() < second.score();
      }
    };

    typedef vector<SplitCandidate>	_SplitHeap;
    typedef _SplitHeap::iterator	_SplitHeapIterator;
    typedef _SplitHeap::const_iterator	_SplitHeapConstIterator;

  public:
    SplitHeap() : _isHeap(false) { }

    bool empty() { return _splitHeap.empty(); }

    SplitCandidate pop();
    void push(const SplitCandidate& split);
    void push_back(const SplitCandidate& split);
    void heapify();

  private:
    _SplitHeap					_splitHeap;
    bool					_isHeap;
  };

public:
  ClusterDistribTree(const String& nm, PhonesSetPtr& phonesSet, const int contextLength = 1, const String& fileName = "",
		     char comment = ';', const String& wb = "WB", const String& eos = "</s>", bool verbose = false, unsigned minCount = 1000);

  void grow(unsigned leavesN);
  
        NodePtr& node(const String& nm)       { return Cast<NodePtr>(_node(nm)); }
  const NodePtr& node(const String& nm) const { return Cast<const NodePtr>(_node(nm)); }

  void writeContexts(const String& fileName = "");

 protected:
  void virtual _distributeContexts() = 0;
  void _initializeSplitHeap();
  void _initializeQuestionList();
  virtual double _counts(NodePtr& node) const = 0;

  virtual void _allocAccumulators() = 0;

  virtual double _splitScore(const Question& ques, NodePtr& splitNode) = 0;
  unsigned _leavesN();

  void _splitCandidate(NodePtr& splitNode);
  void _split(SplitCandidate& bestSplit);

  SplitHeap					_splitHeap;
  QuestionList					_qlist;
  unsigned					_leafN;
  unsigned					_clusterX;
  unsigned					_minCount;
};

typedef Inherit<ClusterDistribTree, DistribTreePtr> ClusterDistribTreePtr ;


// ----- definition for class `CodebookClusterDistribTree' -----
//
class CodebookClusterDistribTree : public ClusterDistribTree {
public:
  CodebookClusterDistribTree(const String& nm, PhonesSetPtr& phonesSet, CodebookSetTrainPtr& cbs, const int contextLength = 1, const String& fileName = "",
			     char comment = ';', const String& wb = "WB", const String& eos = "</s>", bool verbose = false, unsigned minCount = 1000);
  ~CodebookClusterDistribTree();

  void _allocAccumulators();
  virtual double _splitScore(const Question& ques, NodePtr& splitNode);

protected:
  void virtual _distributeContexts();
  virtual double _counts(NodePtr& node) const;

private:
  CodebookSetTrainPtr				_cbs;

  CodebookTrain::AccuPtr			_parentAccu;
  CodebookTrain::AccuPtr			_leftAccu;
  CodebookTrain::AccuPtr			_rightAccu;

  float*					_count;
  gsl_matrix_float*				_rv;
  gsl_vector_float*				_cv;
};

typedef Inherit<CodebookClusterDistribTree, ClusterDistribTreePtr> CodebookClusterDistribTreePtr;

CodebookClusterDistribTreePtr
clusterCodebooks(PhonesSetPtr& phonesSet, CodebookSetTrainPtr& cbs, unsigned leavesN, const int contextLength = 1, const String& fileName = "",
		 char comment = ';', const String& wb = "WB", const String& eos = "</s>", bool verbose = false, unsigned minCount = 1000);


// ----- definition for class `MixtureWeightClusterDistribTree' -----
//
class MixtureWeightClusterDistribTree : public ClusterDistribTree {
public:
  MixtureWeightClusterDistribTree(const String& nm, PhonesSetPtr& phonesSet, DistribSetTrainPtr& dss, const int contextLength = 1, const String& fileName = "",
				  char comment = ';', const String& wb = "WB", const String& eos = "</s>", bool verbose = false, unsigned minCount = 1000);
  ~MixtureWeightClusterDistribTree();

  void _allocAccumulators();
  virtual double _splitScore(const Question& ques, NodePtr& splitNode);

protected:
  void virtual _distributeContexts();
  virtual double _counts(NodePtr& node) const;

private:
  DistribSetTrainPtr				_dss;
  unsigned					_refN;

  DistribTrain::AccuPtr				_parentAccu;
  DistribTrain::AccuPtr				_leftAccu;
  DistribTrain::AccuPtr				_rightAccu;

  float*					_val;
  float						_count;
};

typedef Inherit<MixtureWeightClusterDistribTree, ClusterDistribTreePtr>	MixtureWeightClusterDistribTreePtr;

MixtureWeightClusterDistribTreePtr
clusterMixtureWeights(PhonesSetPtr& phonesSet, DistribSetTrainPtr& dss, unsigned leavesN, const int contextLength = 1, const String& fileName = "",
		      char comment = ';', const String& wb = "WB", const String& eos = "</s>", bool verbose = false, unsigned minCount = 1000);

#endif
