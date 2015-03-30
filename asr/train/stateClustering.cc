//
//			         Millennium
//                    Automatic Speech Recognition System
//                                  (asr)
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

#include <algorithm>
#include "train/stateClustering.h"


static bool Verbosity = true;

// ----- methods for class `ClusterDistribTree' -----
//
ClusterDistribTree::
ClusterDistribTree(const String& nm, PhonesSetPtr& phonesSet,
		   const int contextLength, const String& fileName,
		   char comment, const String& wb, const String& eos, bool verbose, unsigned minCount)
  : DistribTree(nm, phonesSet, contextLength, /*fileName=*/ "", comment, wb, eos, verbose), _minCount(minCount),
    _leafN(_leavesN())
{
  if (fileName != "") read(fileName);

  _initializeQuestionList();
}

void ClusterDistribTree::grow(unsigned leavesN)
{
  _clusterX = _maxIndex();
  _leafN    = DistribTree::leavesN();
  while (_leafN < leavesN && _splitHeap.empty() == false) {
    SplitCandidate bestSplit(_splitHeap.pop());
    _split(bestSplit);
    _leafN++;
  }
}

void ClusterDistribTree::writeContexts(const String& fileName)
{
  FILE* fp = (fileName == "") ? stdout : fileOpen(fileName, "w");

#if 1
  node("ROOT-b")->writeContexts(fp);
  node("ROOT-m")->writeContexts(fp);
  node("ROOT-e")->writeContexts(fp);
#else
  node("ROOT-0")->writeContexts(fp);
  node("ROOT-1")->writeContexts(fp);
  node("ROOT-2")->writeContexts(fp);
#endif

  if (fp != stdout) fileClose(fileName, fp);
}

void ClusterDistribTree::_split(SplitCandidate& bestSplit)
{
  NodePtr splitNode(bestSplit.node());

  QuestionList qlist;
  qlist.add(bestSplit.question());
  splitNode->_qlist.add(qlist);

  const String splitNodeName(splitNode->name());
  String baseName(splitNodeName);
  String::size_type pos = splitNodeName.find_first_of("(");
  if (pos != String::npos)
    baseName = splitNodeName.substr(0, pos);

  // create children nodes here
  static char nameBuffer[100];
  sprintf(nameBuffer, "%s(%d)", baseName.c_str(), ++_clusterX);
  String leftName(nameBuffer);

#if 1
  _map.insert(_MapType(leftName, _newNode(_phonesSet, this, leftName, /*ques=*/ "", /*left=*/ "-", /*right=*/ "-", /*unknown=*/ "-", /*leaf=*/ leftName)));
#endif

  sprintf(nameBuffer, "%s(%d)", baseName.c_str(), ++_clusterX);
  String rightName(nameBuffer);

#if 1
  _map.insert(_MapType(rightName, _newNode(_phonesSet, this, rightName, /*ques=*/ "", /*left=*/ "-", /*right=*/ "-", /*unknown=*/ "-", /*leaf=*/ rightName)));
#endif

  splitNode->_leaf = "-";  splitNode->_left = leftName; splitNode->_right = rightName;

  splitNode->distributeContexts(splitNode->contexts());

  splitNode->_contexts = NULL;

  if (Verbosity) {
    printf("Splitting %s --> %s %s : Score %g\n",
	   splitNode->name().c_str(), leftName.c_str(), rightName.c_str(), bestSplit.score());
    fflush(stdout);
  }

  NodePtr& leftNode(node(leftName));
  if (_counts(leftNode) > _minCount)
    _splitCandidate(leftNode);

  NodePtr& rightNode(node(rightName));
  if (_counts(rightNode) > _minCount)
    _splitCandidate(rightNode);
}

unsigned ClusterDistribTree::_leavesN()
{
  return 150;
}

void ClusterDistribTree::_initializeSplitHeap()
{
  for (_MapIterator itr = _map.begin(); itr != _map.end(); itr++) {
    NodePtr& splitNode(Cast<NodePtr>((*itr).second));

    if (splitNode->left() == "-" && splitNode->right() == "-" && splitNode->unknown() == "-" && splitNode->contexts()->size() > 1) {

      if (splitNode->leaf() == "-")
	throw jconsistency_error("Leaf is empty.");

      if (_counts(splitNode) > _minCount)
	_splitCandidate(splitNode);
    }
  }

  _splitHeap.heapify();
}

void ClusterDistribTree::Node::_insertQuestion(const Question& question)
{
  if (_qlist.empty() == false) return;

  QuestionList qlist;
  qlist.add(question);
  QuestionListList qllist;
  qllist.add(qlist);
  _qlist = qllist;
}

void ClusterDistribTree::_splitCandidate(NodePtr& splitNode)
{
  double bestScore = 0.0;
  Question bestQuestion;

  for (QuestionList::Iterator itr(_qlist); itr.more(); itr++) {
    double score = _splitScore(itr.question(), splitNode);
    if (score > bestScore) {
      bestScore    = score;
      bestQuestion = itr.question();
    }
  }

  if (bestScore == 0.0) return;

  if (Verbosity) {
    printf("Adding node %s to split heap.\n", splitNode->name().c_str());
    printf("  Best Question %s : Score %f\n", bestQuestion.toString().c_str(), bestScore);  fflush(stdout);
  }

  splitNode->_insertQuestion(bestQuestion);
  _splitHeap.push(SplitCandidate(bestScore, splitNode, bestQuestion));
}

void ClusterDistribTree::_initializeQuestionList()
{
  static const String PhonesString("PHONES");

  if (Verbosity) { printf("Creating question list ... ");  fflush(stdout); }

  for (int context = -_contextLength; context <= _contextLength; context++) {

    // add single phone questions
    PhonesPtr phones(_phonesSet->find(PhonesString));
    for (Phones::Iterator itr(phones); itr.more(); itr++) {
      Question question(context, *itr);

      if (Verbosity) { printf("Added question %s\n", question.toString().c_str());  fflush(stdout); }

      _qlist.add(question);
    }

    // add phonetic class questions
    for (PhonesSet::Iterator itr(_phonesSet); itr.more(); itr++) {
      if ((*itr)->name() == PhonesString) continue;

      Question question(context, *itr);

      if (Verbosity) { printf("Added question %s\n", question.toString().c_str());  fflush(stdout); }

      _qlist.add(question);
    }

    // add word boundary question
    Question question(context, _wb);

    if (Verbosity) { printf("Added question %s\n", question.toString().c_str());  fflush(stdout); }

    _qlist.add(question);
  }

  if (Verbosity) { printf("Done\n");   fflush(stdout); }
}


// ----- methods for class `ClusterDistribTree::Node' -----
//
ClusterDistribTree::Node::
Node(PhonesSetPtr& ps, DistribTree* dt,
     const String& nm, const String& ques,
     const String& left, const String& right, const String& unknown, const String& leaf)
  : DistribTree::Node(ps, dt, nm, ques, left, right, unknown, leaf)
{
  cout << "Created node " << name() << endl;
}

void ClusterDistribTree::Node::distributeContexts(LexiconPtr& lexicon)
{
  if (_left == "-" && _right == "-" && _unknown == "-") {

    if (_leaf == "-")
      throw jconsistency_error("Leaf is empty.");

    if (lexicon->size() == 0)
      cout << "Problem here!" << endl;

    if (Verbosity) {
      printf("Leaf %s has %d contexts\n", _leaf.c_str(), lexicon->size());  fflush(stdout);
    }

    /*
    cout << "Lexicon: " << lexicon->name() << endl;
    cout << "Name:    " << name()          << endl;
    cout << "Left:    " << _left           << endl;
    cout << "Right:   " << _right          << endl;
    */

    _contexts = lexicon;
    return;
  }

  /*
  if (name() == "N-2") {
    Verbose = true;
    cout << "Problem here 2!" << endl;
    cout << "Name:    " << name()          << endl;
    cout << "Left:    " << _left           << endl;
    cout << "Right:   " << _right          << endl;
  }
  */

  bool create = true;
  LexiconPtr left(new Lexicon("left")), right(new Lexicon("right"));
  for (Lexicon::Iterator itr(lexicon); itr.more(); itr++) {
    const String& context(*itr);

    Answer combinedAnswer = _qlist.answer(context);

    if (combinedAnswer == No)
      left->index(context, create);
    else
      right->index(context, create);
  }

  if (_left != "-") {
    NodePtr leftNode(Cast<NodePtr>(_distribTree->node(_left)));
    leftNode->distributeContexts(left);
  }

  if (_right != "-") {
    NodePtr rightNode(Cast<NodePtr>(_distribTree->node(_right)));
    rightNode->distributeContexts(right);
  }
}

void ClusterDistribTree::Node::writeContexts(FILE* fp)
{
  if (_leaf != "-") {
    fprintf(fp, "\nLeaf %s has %d contexts:\n", _leaf.c_str(), _contexts->size());
    for (Lexicon::Iterator itr(_contexts); itr.more(); itr++)
      fprintf(fp, "  %s\n", (*itr).c_str());
  }

  if (_left != "-") {
    NodePtr node(Cast<NodePtr>(_distribTree->node(_left)));
    node->writeContexts(fp);
  }

  if (_right != "-") {
    NodePtr node(Cast<NodePtr>(_distribTree->node(_right)));
    node->writeContexts(fp);
  }
}

// ----- methods for class `ClusterDistribTree::SplitHeap' -----
//
ClusterDistribTree::SplitCandidate ClusterDistribTree::SplitHeap::pop()
{
  if (_isHeap == false)
    throw jconsistency_error("Must heapify before calling 'SplitHeap::pop'.");

  pop_heap(_splitHeap.begin(), _splitHeap.end(), LessThan());
  SplitCandidate popped(_splitHeap.back());
  _splitHeap.pop_back();
  return popped;
}

void ClusterDistribTree::SplitHeap::push(const SplitCandidate& split)
{
  _splitHeap.push_back(split);

  if (_isHeap)
    push_heap(_splitHeap.begin(), _splitHeap.end(), LessThan());
}

void ClusterDistribTree::SplitHeap::heapify()
{
  if (_isHeap)
    throw jconsistency_error("Already heapified!");

  make_heap(_splitHeap.begin(), _splitHeap.end(), LessThan());
  _isHeap = true;
}


// ----- methods for class `CodebookClusterDistribTree' -----
//
CodebookClusterDistribTree::
CodebookClusterDistribTree(const String& nm, PhonesSetPtr& phonesSet, CodebookSetTrainPtr& cbs,
			   const int contextLength, const String& fileName,
			   char comment, const String& wb, const String& eos, bool verbose, unsigned minCount)
  : ClusterDistribTree(nm, phonesSet, contextLength, fileName, comment, wb, eos, verbose, minCount),
    _cbs(cbs)
{
  _distributeContexts();
  _allocAccumulators();
  _initializeSplitHeap();
}

CodebookClusterDistribTree::~CodebookClusterDistribTree()
{
  delete[] _count;  gsl_matrix_float_free(_rv);  gsl_vector_float_free(_cv);
}

double CodebookClusterDistribTree::_counts(NodePtr& node) const
{
  unsigned refX = 0;
  double cnt    = 0.0;
  for (Lexicon::Iterator itr(node->contexts()); itr.more(); itr++) {
    const String& context(*itr);
    CodebookTrainPtr& cb(_cbs->find(context));
    cnt += cb->accu()->numProb(refX);
  }

  return cnt;
}

void CodebookClusterDistribTree::_allocAccumulators()
{
  unsigned sbN = 1;
  unsigned subFeatN = 1;
  unsigned dmN;
  unsigned maxRefN = 0;
  CovType cvTyp = COV_DIAGONAL;
  for (CodebookSetTrain::CodebookIterator itr(_cbs); itr.more(); itr++) {
    CodebookTrainPtr& cbk(itr.cbk());

    dmN = cbk->featLen();

    if (cbk->refN() > maxRefN)
      maxRefN = cbk->refN();
  }

  _parentAccu = new CodebookTrain::Accu(sbN, dmN, subFeatN, maxRefN, cvTyp);
  _leftAccu   = new CodebookTrain::Accu(sbN, dmN, subFeatN, maxRefN, cvTyp);
  _rightAccu  = new CodebookTrain::Accu(sbN, dmN, subFeatN, maxRefN, cvTyp);

  _count      = new float[maxRefN];
  _rv         = gsl_matrix_float_alloc(maxRefN, dmN);
  _cv         = gsl_vector_float_alloc(dmN);
}

void CodebookClusterDistribTree::_distributeContexts()
{
  for (unsigned i = 0; i < 3; i++) {
#if 1
    String ending("-b");
    if (i == 1)
      ending = "-m";
    else if (i == 2)
      ending = "-e";
#else
    String ending("-0");
    if (i == 1)
      ending = "-1";
    else if (i == 2)
      ending = "-2";
#endif

    bool create = true;
    LexiconPtr lexicon(new Lexicon("root"));
    for (CodebookSetTrain::CodebookIterator itr(_cbs); itr.more(); itr++) {
      const String& cbname(itr.cbk()->name());
      String::size_type pos = cbname.find(ending);
      if (pos != String::npos)
	lexicon->index(cbname, create);
    }
    String root("ROOT");  root += ending;

    if (Verbosity) { printf("Distributing contexts to %s\n", root.c_str());  fflush(stdout); }

    NodePtr rootNode(node(root));
    rootNode->distributeContexts(lexicon);
  }
}

double CodebookClusterDistribTree::_splitScore(const Question& ques, NodePtr& splitNode)
{
  _parentAccu->zero();  _leftAccu->zero();  _rightAccu->zero();
  for (Lexicon::Iterator itr(splitNode->_contexts); itr.more(); itr++) {
    const String& context(*itr);

    Answer answer = ques.answer(context);

    CodebookTrainPtr& cb(_cbs->find(context));
    _parentAccu->add(cb->accu());
    if (answer == No)
      _leftAccu->add(cb->accu());
    else
      _rightAccu->add(cb->accu());
  }

  // calculate score for parent
  double parentScore = 0.0;
  _parentAccu->update(_count, _rv, &_cv);
  for (unsigned i = 0; i < _cbs->featLen(); i++)
    parentScore += -log(gsl_vector_float_get(_cv, i));
  parentScore *= _count[0];

  // calculate score for left child
  double leftScore = 0.0;
  _leftAccu->update(_count, _rv, &_cv);
  for (unsigned i = 0; i < _cbs->featLen(); i++)
    leftScore += -log(gsl_vector_float_get(_cv, i));
  leftScore *= _count[0];

  if (_count[0] < _minCount) return -HUGE;

  // calculate score for right child
  double rightScore = 0.0;
  _rightAccu->update(_count, _rv, &_cv);
  for (unsigned i = 0; i < _cbs->featLen(); i++)
    rightScore += -log(gsl_vector_float_get(_cv, i));
  rightScore *= _count[0];

  if (_count[0] < _minCount) return -HUGE;

  // cout << splitNode->name() << " : " << ques.toString() << " : " << leftScore << " : " << rightScore << " : " << parentScore << " : " << ((leftScore + rightScore) - parentScore) << endl;

  return  (leftScore + rightScore) - parentScore;
}

CodebookClusterDistribTreePtr
clusterCodebooks(PhonesSetPtr& phonesSet, CodebookSetTrainPtr& cbs, unsigned leavesN, const int contextLength, const String& fileName,
		 char comment, const String& wb, const String& eos, bool verbose, unsigned minCount)
{
  CodebookClusterDistribTreePtr clusterTree(new CodebookClusterDistribTree("Codebook Tree", phonesSet, cbs, contextLength, fileName,
									   comment, wb, eos, verbose, minCount));
  clusterTree->grow(leavesN);

  return clusterTree;
}


// ----- methods for class `MixtureWeightClusterDistribTree' -----
//
MixtureWeightClusterDistribTree::
MixtureWeightClusterDistribTree(const String& nm, PhonesSetPtr& phonesSet, DistribSetTrainPtr& dss, const int contextLength, const String& fileName,
				char comment, const String& wb, const String& eos, bool verbose, unsigned minCount)
  : ClusterDistribTree(nm, phonesSet, contextLength, fileName, comment, wb, eos, verbose, minCount), _dss(dss)
{
  _distributeContexts();
  _allocAccumulators();
  _initializeSplitHeap();
}

MixtureWeightClusterDistribTree::~MixtureWeightClusterDistribTree()
{
  delete[] _val;
}

double MixtureWeightClusterDistribTree::_counts(NodePtr& node) const
{
  unsigned refX = 0;
  double cnt    = 0.0;
  for (Lexicon::Iterator itr(node->contexts()); itr.more(); itr++) {
    const String& context(*itr);
    DistribTrainPtr& ds(_dss->find(context));
    for (unsigned i = 0; i < ds->valN(); i++)
      cnt += ds->accu()->counts();
  }

  return cnt;
}

void MixtureWeightClusterDistribTree::_distributeContexts()
{
  for (unsigned i = 0; i < 3; i++) {
#if 1
    String ending("-b");
    if (i == 1)
      ending = "-m";
    else if (i == 2)
      ending = "-e";
#else
    String ending("-0");
    if (i == 1)
      ending = "-1";
    else if (i == 2)
      ending = "-2";
#endif

    bool create = true;
    LexiconPtr lexicon(new Lexicon("root"));
    for (DistribSetTrain::Iterator itr(_dss); itr.more(); itr++) {
      const String& cbname(itr.dst()->name());
      String::size_type pos = cbname.find(ending);
      if (pos != String::npos)
	lexicon->index(cbname, create);
    }
    String root("ROOT");  root += ending;
    NodePtr rootNode(node(root));
    rootNode->distributeContexts(lexicon);
  }
}

void MixtureWeightClusterDistribTree::_allocAccumulators()
{
  unsigned valN = 0;
  for (DistribSetBasic::Iterator itr(_dss); itr.more(); itr++) {
    if (itr.dst()->valN() > valN)
      valN = itr.dst()->valN();
  }

  _parentAccu = new DistribTrain::Accu(valN);
  _leftAccu   = new DistribTrain::Accu(valN);
  _rightAccu  = new DistribTrain::Accu(valN);

  _val        = new float[valN];
}

double MixtureWeightClusterDistribTree::_splitScore(const Question& ques, NodePtr& splitNode)
{
  unsigned valN = 0;
  _parentAccu->zero();  _leftAccu->zero();  _rightAccu->zero();
  for (Lexicon::Iterator itr(splitNode->contexts()); itr.more(); itr++) {
    const String& context(*itr);

    Answer answer = ques.answer(context);

    DistribTrainPtr& ds(_dss->find(context));

    if (valN == 0)
      valN = ds->valN();
    else if (valN != ds->valN())
      throw jconsistency_error("valN (%d vs. %d) does not match", valN, ds->valN());

    _parentAccu->add(ds->accu());
    if (answer == No)
      _leftAccu->add(ds->accu());
    else
      _rightAccu->add(ds->accu());
  }

  // calculate score for parent
  double parentScore = 0.0;
  _parentAccu->update(_val, _count);
  for (unsigned i = 0; i < valN; i++)
    parentScore += _val[i];
  parentScore *= _count;

  // calculate score for left child
  double leftScore = 0.0;
  _leftAccu->update(_val, _count);
  for (unsigned i = 0; i < valN; i++)
    leftScore += _val[i];
  leftScore *= _count;

  // calculate score for right child
  double rightScore = 0.0;
  _rightAccu->update(_val, _count);
  for (unsigned i = 0; i < valN; i++)
    rightScore += _val[i];
  rightScore *= _count;

  return  (leftScore + rightScore) - parentScore;
}

MixtureWeightClusterDistribTreePtr
clusterMixtureWeights(PhonesSetPtr& phonesSet, DistribSetTrainPtr& dss, unsigned leavesN, const int contextLength, const String& fileName,
		      char comment, const String& wb, const String& eos, bool verbose, unsigned minCount)
{
  MixtureWeightClusterDistribTreePtr clusterTree(new MixtureWeightClusterDistribTree("Mixture Weight Tree", phonesSet, dss, contextLength, fileName,
										     comment, wb, eos, verbose, minCount));
  clusterTree->grow(leavesN);

  return clusterTree;
}
