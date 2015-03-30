//
//			          Millenium
//                    Automatic Speech Recognition System
//                                  (asr)
//
//  Module:  asr.sat
//  Purpose: Speaker-adapted ML and discriminative HMM training.
//  Author:  John McDonough.
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


#include <set>
#include "common/mach_ind_io.h"
#include "sat/cluster.h"

// ----- global trace variables -----
//
static       int Trace             = 0x0000;
static const int Top               = 0x0001;
static const int NodeScore         = 0x0002;
static const int NodeDistance      = 0x0004;
static const int NodeDistribution  = 0x0010;
static const int NoGaussComponents = 0x0020;
static const int NodeClusters      = 0x0040;
static const int NodeBID           = 0x0100;
static const int NodeDET           = 0x0200;


// ----- methods for class `AccumSum' -----
//
AccumSum::AccumSum(UnShrt len, UnShrt nsub)
  : _len(len), _occ(0.0),
    _sum(_len, nsub),
    _sqr(_len, nsub)
{
  zero();
}

AccumSum::AccumSum(const GaussDensity& mp)
  : _len(mp.featLen()), _occ(1.0),
    _sum(_len, mp.nSubFeat()),
    _sqr(_len, mp.nSubFeat())
{
  const NaturalVector mean(mp.mean()), invVar(mp.invVar());

  for (UnShrt k = 0; k < _len; k++) {
    float mn = mean[k];
    _sum[k] = mn * _occ;
    _sqr[k] = ((1.0 / invVar[k]) + (mn * mn)) * _occ;
  }
}

void AccumSum::zero()
{
  for (UnShrt k = 0; k < _len; k++)
    _sum[k] = _sqr[k] = 0.0;

  _occ = 0.0;
}

void AccumSum::operator+=(const AccumSum& acc)
{
  _occ += acc._occ;
  for (UnShrt k = 0; k < _len; k++) {
    _sum[k] += acc._sum[k];
    _sqr[k] += acc._sqr[k];
  }
}

void AccumSum::calcMean(NaturalVector& mean)
{
  assert(_len == mean.featLen());

  for (UnShrt k = 0; k < _len; k++)
    mean[k] = _sum[k] / _occ;
}

void AccumSum::calcVariance(NaturalVector& var)
{
  assert(_len == var.featLen());

  for (UnShrt k = 0; k < _len; k++)
    var[k] = (_sqr[k] - (_sum[k] * _sum[k] / _occ)) / _occ;
}


// ----- methods for class `AccumSumList' -----
//
// constructor for K-means regression class clustering:
//     allow multiple Gaussian mixtures per state
//
AccumSumList::AccumSumList(CodebookSetFastSATPtr& cb)
  : _cbs(cb)
{
  for (CodebookSetFastSAT::GaussianIterator itr(_cbs); itr.more(); itr++) {
    const GaussDensity pdf(itr.mix());
    _list[pdf.listPtr()] = new AccumSum(pdf);
  }
}

const AccumSum& AccumSumList::operator[](ListPtr mp) const
{
  _AccumSumListIter itr = _list.find(mp);
  if (itr == _list.end())
    throw j_error("Could not find statistics for mixture\n");

  return *((*itr).second);
}

AccumSumList::~AccumSumList()
{
  for (_AccumSumListIter itr = _list.begin(); itr != _list.end(); itr++)
    delete (*itr).second;
  _list.erase(_list.begin(), _list.end());
}


// ----- methods for class `SpkrStat' -----
//
SpkrStat::SpkrStat(const GaussDensity& pdf, const TransformBasePtr& tr)
  : _featLen(pdf.featLen()), _trans(tr),
    _sumO(NaturalVector(pdf.sumO(),    /* deepCopy= */ true)),
    _sumOsq(NaturalVector(pdf.sumOsq(),/* deepCopy= */ true)),
    _occ(pdf.postProb()) { }


// ----- methods for class `SpkrStatList' -----
//
SpkrStatList::SpkrStatList(const SpeakerList& spkrList,
			   const TransformerTreeList& transList,
			   CodebookSetFastSATPtr& cb, const String& accumDir)
  : _maxMix(0), _cbs(cb)
{
  CodebookList codebookList(cb);
  _initSpkrStatList(spkrList, transList, _cbs, codebookList, accumDir);
}

void SpkrStatList::_initSpkrStatList(const SpeakerList& spkrList,
				     const TransformerTreeList& transList,
				     CodebookSetFastSATPtr& cb,
				     CodebookList& codebookList,
				     const String& meanVarStats, bool singleMix)
{
  if (Trace & Top) {
    cout << endl << "Loading SAT statistics ... ";  fflush(stdout);
  }
  for (SpeakerList::Iterator spkr(spkrList); spkr.more(); spkr++) {
    const TransformerTreePtr tree = transList.getTree(spkr);
    String accumFileName(meanVarStats); accumFileName += "."; accumFileName += spkr;

    FILE* fp = fileOpen(accumFileName, "r");
    read_float(fp);  read_int(fp);    // for `totalPr' and `totalT'

    AccMap accMap(fp);
    long int startOfAccs = ftell(fp);

    for (CodebookList::Iterator itr(codebookList); itr.more(); itr++) {
      CodebookFastSATPtr cbk(itr.cbk());
      int nMix  = cbk->refN();

      if (singleMix && nMix != 1)
	throw j_error("Must have a single mixture, not %d.", nMix);
      if (_maxMix == 0) {
	_maxMix = nMix;
      } else if (nMix > _maxMix)
	throw j_error("Need %d mixtures, but maximum is %d\n", nMix, _maxMix);

      String stateName(cbk->name());
      long int current = accMap.state(stateName);

      if (current == AccMap::NotPresent) continue;

      current += startOfAccs;

      if (current != ftell(fp))
	fseek(fp, current, SEEK_SET);

      cbk->loadAccu(fp);
      for (CodebookFastSAT::Iterator gitr(cbk); gitr.more(); gitr++) {
	const CodebookFastSAT::GaussDensity mix(gitr.mix());
	const TransformBasePtr& trans(tree->transformer(mix.regClass()));
	_SpkrStatList& _statList = _statMap[mix.listPtr()];
	_statList.push_back(new SpkrStat(mix, trans));
      }
    }
    fileClose(accumFileName, fp);
  }

  if (Trace & Top)
    cout << "Done" << endl;
}

SpkrStatList::~SpkrStatList()
{
  for (_SpkrStatMapIter mItr=_statMap.begin(); mItr!=_statMap.end(); mItr++) {
    const _SpkrStatList& _statList = (*mItr).second;
    for (_SpkrStatListIter sItr=_statList.begin(); sItr!=_statList.end(); sItr++) {
      delete *sItr;
    }
  }
}

const SpkrStatList::_SpkrStatList SpkrStatList::Iterator::_EmptyList;

const SpkrStatList::_SpkrStatList& SpkrStatList::Iterator::
_initList(SpkrStatList& list, ListPtr pdf)
{
  _SpkrStatMapIter iter = list._statMap.find(pdf);
  if (iter == list._statMap.end())
    return _EmptyList;

  return (*iter).second;
}

SpkrStatList::CodebookList::CodebookList(CodebookSetFastSATPtr& cb)
{
  for (CodebookSetFastSAT::CodebookIterator itr(cb); itr.more(); itr++)
    _slist.push_back(itr.cbk());
}


// ----- methods for classs `BaseRegClassKMeans' -----
//
BaseRegClassKMeans::BaseRegClassKMeans(CodebookSetFastSATPtr& cb, const String& gCovFile, int trace)
  : _featLen(cb->featLen()), _grandVariance(_featLen, cb->nSubFeat()), _invGrandVariance(_featLen, cb->nSubFeat())
{
  Trace = trace;

  if (gCovFile != "")
    _loadGlobalCovariance(_invGrandVariance, gCovFile);
  else
    for (unsigned dimX = 0; dimX < cb->featLen(); dimX++)
      _invGrandVariance[dimX] = 1.0;

  for (int dimX = 0; dimX < cb->featLen(); dimX++)
    _grandVariance[dimX] = 1.0 / _invGrandVariance[dimX];

  if (Trace & Top)
    cout << endl
	 << "Grand Variance:" << endl
	 << _grandVariance    << endl;
}

float BaseRegClassKMeans::_euclidean(const NaturalVector& v1, const NaturalVector& v2)
{
  assert(v1.featLen() == _featLen && v2.featLen() == _featLen);

  float dist = 0.0;
  for (UnShrt k = 0; k < _featLen; k++) {
    float diff = v1[k] - v2[k];
    dist += diff * diff * _invGrandVariance[k];
  }

  return dist;
}

void BaseRegClassKMeans::_loadGlobalCovariance(NaturalVector& invVar, const String& covFloorFile)
{
  CodebookBasic cbk("global", /*rfN=*/ 1, _featLen, COV_DIAGONAL, "");

  FILE* fp = fileOpen(covFloorFile, "r");
  cbk.load(fp, /* readName= */ true);
  fileClose(covFloorFile, fp);

  for (unsigned dimX = 0; dimX < _featLen; dimX++)
    invVar[dimX] = cbk.invCov(/*refX=*/ 0, dimX);
}

float BaseRegClassKMeans::calcNodeScore(RCNode* rNode)
{
  if (Trace & NodeScore)
    cout << "Calculating score for Node "
	 << rNode->index() << ":" << endl;

  float score = 0.0;
  for (GaussListFastSAT::Iterator itr(rNode->list()); itr.more(); itr++)
    score += distance(itr.mix(), rNode);

  if (Trace & NodeScore)
    cout << "Score = " << score << endl;

  return score;
}


// ----- methods for classs `HTKRegClassKMeans' -----
//
HTKRegClassKMeans::HTKRegClassKMeans(CodebookSetFastSATPtr hmm, const String& gCovFile, int trace)
  : BaseRegClassKMeans(hmm, gCovFile, trace), _accumList(hmm),
    _accumSum1(_featLen), _accumSum2(_featLen) { }

void HTKRegClassKMeans::calcClusterDistribution(RCNode* rNode)
{
  _accumSum1.zero();

  unsigned ttlComps = 0;
  for (GaussListFastSAT::Iterator itr(rNode->list()); itr.more(); itr++) {
    GaussDensity& mix(itr.mix());
    _accumSum1 += _accumList[mix.listPtr()];
    ttlComps++;
  }

  _accumSum1.calcMean(rNode->_aveMean);
  _accumSum1.calcVariance(rNode->_aveCovar);

  if (Trace & NodeDistribution)
    cout << "Average Mean:"     << endl
	 << NaturalVector(rNode->_aveMean)  << endl
	 << "Average Variance:" << endl
	 << NaturalVector(rNode->_aveCovar) << endl;

  if (Trace & NoGaussComponents)
    cout << "Node "  << rNode->index() << " has "
	 << ttlComps << " gaussian components." << endl << endl;
}

// calculate the total distance of the current
// split and update the cluster means
double HTKRegClassKMeans::calcDistance(GaussListFastSAT& list, RCNodePtr& ch1, RCNodePtr& ch2)
{
  ch1->_clusterScore = 0.0; ch2->_clusterScore = 0.0;
  ch1->_clustAcc     = 0.0; ch2->_clustAcc     = 0.0;

  _accumSum1.zero();  _accumSum2.zero();
  for (GaussListFastSAT::Iterator itr(list); itr.more(); itr++) {
    GaussDensity& pdf(itr.mix());
    const AccumSum& acc = _accumList[pdf.listPtr()];

    float score1 = distance(pdf, ch1.operator->()), score2 = distance(pdf, ch2.operator->());

    if (score1 < score2) {
      ch1->_clustAcc     += acc.occ();
      ch1->_clusterScore += score1;
      _accumSum1         += acc;
    } else {
      ch2->_clustAcc     += acc.occ();
      ch2->_clusterScore += score2;
      _accumSum2         += acc;
    }
  }
  _accumSum1.calcMean(ch1->_aveMean);
  _accumSum2.calcMean(ch2->_aveMean);

  return ch1->_clusterScore + ch2->_clusterScore;
}

// return count-weighted Euclidean distance
float HTKRegClassKMeans::distance(const GaussDensity& pdf, RCNode* rNode)
{
  const AccumSum& acc = _accumList[pdf.listPtr()];

  return
    _euclidean(pdf.mean(), rNode->_aveMean) * acc.occ();
}


// ----- methods for classs `SATRegClassKMeans' -----
//
SpeakerList*         SATRegClassKMeans::spkrList      = NULL;
TransformerTreeList* SATRegClassKMeans::spkrTransList = NULL;

SATRegClassKMeans::SATRegClassKMeans(CodebookSetFastSATPtr& cbs,
				     const TransformerTreeList& transList,
				     SpeakerList& spkrList, const String& meanVarStats,
				     const String& gCovFile, int trace)
  : BaseRegClassKMeans(cbs, gCovFile, trace),
    _statList(spkrList, transList, cbs, meanVarStats),
    _orgFeatLen(cbs->orgFeatLen()),
    _origMean1(_orgFeatLen, cbs->nSubFeat()),
    _origMean2(_orgFeatLen, cbs->nSubFeat()),
    _transMean(_featLen,    cbs->nSubFeat()),
    _satMean1(_featLen, _orgFeatLen),
    _satMean2(_featLen, _orgFeatLen),
    _satVariance(_featLen) { }

// using the SAT formulae, calculate the ML mean
// and variance of the given node
void SATRegClassKMeans::calcClusterDistribution(RCNode* rNode)
{
  if (Trace & NodeDistribution)
    cout << "Calculating distribution for Node "
	 << rNode->index() << ":" << endl;

  _satMean1.zero();
  for (GaussListFastSAT::Iterator gitr(rNode->list()); gitr.more(); gitr++) {
    GaussDensity& pdf(gitr.mix());
    for (SpkrStatList::Iterator sitr(_statList, pdf.listPtr()); sitr.more(); sitr++) {
      const SpkrStat& stat(sitr());
      _satMean1.accumulate(stat.trans(), stat.occ(), _invGrandVariance, stat.sumO());
    }
  }
  _satMean1.update(rNode->_aveMean);

  _satVariance.zero();
  unsigned ttlComps = 0;
  for (GaussListFastSAT::Iterator gitr(rNode->list()); gitr.more(); gitr++) {
    GaussDensity& pdf(gitr.mix());
    for (SpkrStatList::Iterator sitr(_statList, pdf.listPtr()); sitr.more(); sitr++) {
      const SpkrStat& stat(sitr());
      _satVariance.accumulate(stat.trans(), stat.occ(), rNode->_aveMean,
			      stat.sumO(), stat.sumOsq());
    }
    ttlComps++;
  }
  _satVariance.update(rNode->_aveCovar);

  if (Trace & NodeDistribution)
    cout << "Average Mean:"     << endl
	 << NaturalVector(rNode->_aveMean)  << endl
	 << "Average Variance:" << endl
	 << NaturalVector(rNode->_aveCovar) << endl;

  if (Trace & NoGaussComponents)
    cout << "Node "  << rNode->index() << " has "
	 << ttlComps << " gaussian components." << endl << endl;
}

// calculate the total distance of the current split
// using SAT statistics; update the cluster means
double SATRegClassKMeans::calcDistance(GaussListFastSAT& glist, RCNodePtr& ch1, RCNodePtr& ch2)
{
  ch1->_clusterScore = ch2->_clusterScore = 0.0;
  ch1->_clustAcc     = ch2->_clustAcc     = 0.0;

  if (Trace & NodeDistance)
    cout << "Calculating distance for Nodes "
	 << ch1->index() << " and "
	 << ch2->index() << " ... " << endl;
  
  _satMean1.zero();  _satMean2.zero();
  for (GaussListFastSAT::Iterator itr(glist); itr.more(); itr++) {
    GaussDensity& pdf(itr.mix());
    float score1 = distance(pdf, ch1.operator->()), score2 = distance(pdf, ch2.operator->());

    // should use `min(score1, score2)'
    float       bestScore   = (score1 < score2) ? score1    : score2;
    RCNodePtr&  bestChild   = (score1 < score2) ? ch1       : ch2;
    SATMean&    bestSATMean = (score1 < score2) ? _satMean1 : _satMean2;

    bestChild->_clusterScore += bestScore;
    for (SpkrStatList::Iterator itr(_statList, pdf.listPtr()); itr.more(); itr++) {
      const SpkrStat& stat(itr());

      bestChild->_clustAcc += stat.occ();
      bestSATMean.accumulate(stat.trans(), stat.occ(), _invGrandVariance, stat.sumO());
    }
  }
  _satMean1.update(ch1->_aveMean);  _satMean2.update(ch2->_aveMean);

  if (Trace & NodeDistribution) {
    cout << "Done" << endl;
    cout << "Average Mean for Node " << ch1->index() << ":" << endl;
    cout << NaturalVector(ch1->_aveMean) << endl;
    cout << "Average Mean for Node " << ch2->index() << ":" << endl;
    cout << NaturalVector(ch2->_aveMean) << endl;
  }

  double ttlDistance = ch1->_clusterScore + ch2->_clusterScore;

  if (Trace & NodeDistance) {
    printf("Total distance = %10.4e\n\n", ttlDistance); fflush(stdout);
  }

  return ttlDistance;
}

// return count-weighted Euclidean distance
float SATRegClassKMeans::distance(const GaussDensity& pdf, RCNode* rNode)
{
  static NaturalVector spkMean;
  if (spkMean.featLen() == 0)
    spkMean.resize(pdf.featLen(), pdf.nSubFeat());

  float score = 0.0;
  for (SpkrStatList::Iterator itr(_statList, pdf.listPtr()); itr.more(); itr++) {
    const SpkrStat& stat(itr());

    if (stat.occ() == 0.0) continue;

    stat.trans()->transform(rNode->_aveMean, _transMean);
    spkMean  = stat.sumO();
    spkMean /= stat.occ();
    score += _euclidean(_transMean, spkMean) * stat.occ();
  }

  return score;
}


// ----- methods for classs `SATRegClassLhoodRatio' -----
//
SATRegClassLhoodRatio::
SATRegClassLhoodRatio(CodebookSetFastSATPtr& cbs, const TransformerTreeList& transList,
		      SpeakerList& spkrList, const String& meanVarStats,
		      const String& gCovFile, int trace)
  : SATRegClassKMeans(cbs, transList, spkrList, meanVarStats, gCovFile, trace),
    _spkrVar(_featLen, cbs->nSubFeat())
{
  if (gCovFile == "")
    throw j_error("Must use grand variance with likelihood ratio class splitting.");
}

inline double square(double s) { return s * s; }

static const double CovarianceFloor = 1.0e-19;

bool SATRegClassLhoodRatio::_calcSpeakerVariance(GaussDensity& pdf)
{
  _spkrVar = 0.0;  _ttlCounts = 0.0;
  for (SpkrStatList::Iterator itr(_statList, pdf.listPtr()); itr.more(); itr++) {
    const SpkrStat& stat(itr());

    const NaturalVector& sumO(stat.sumO());
    const NaturalVector& sumOsq(stat.sumOsq());
    float occ = stat.occ();

    if (occ == 0.0) continue;

    _ttlCounts += occ;
    for (UnShrt i = 0; i < _featLen; i++)
      _spkrVar[i] += sumOsq[i] - square(sumO[i]) / occ;
  }

  if (_ttlCounts <= 1.0) return false;

  for (UnShrt i = 0; i < _featLen; i++) {
    _spkrVar[i] /= _ttlCounts;

    if (_spkrVar[i] < CovarianceFloor) {
      printf("Covariance component is less than floor (%g < %g) : Total Counts = %g.\n",
	     _spkrVar[i], CovarianceFloor, _ttlCounts);
      _spkrVar[i] = CovarianceFloor;
    }
  }

  return true;
}

const float SATRegClassLhoodRatio::MinLogLhoodRatioCount = 5.0;

float SATRegClassLhoodRatio::_logLhoodRatio(const NaturalVector& variance)
{
#if 0
  if (_ttlCounts < MinLogLhoodRatioCount) return 0.0;

  float ratio = 0.0;
  for (UnShrt i = 0; i < _featLen; i++)
    ratio -= log(variance[i] * _spkrVar[i]);

  if (isnan(ratio))
    throw j_error("Likelihood ratio is NaN.");

  if (ratio < 0.0)
    throw j_error("Likelihood ratio (%g) < 0.0.", ratio);

  return 0.5 * _ttlCounts * ratio;

#else

  return _ttlCounts;

#endif
}

float SATRegClassLhoodRatio::calcNodeScore(RCNode* rNode)
{
  if (Trace & NodeScore)
    cout << "Calculating score for Node "
	 << rNode->index() << ":" << endl;

  float logLhoodRatio = 0.0;
  for (GaussListFastSAT::Iterator itr(rNode->list()); itr.more(); itr++) {
    GaussDensity& pdf(itr.mix());

    if (_calcSpeakerVariance(pdf))
      logLhoodRatio += _logLhoodRatio(pdf.invVar());
  }

  if (Trace & NodeScore)
    cout << "Score = " << logLhoodRatio << endl;

  return logLhoodRatio;
}


// ----- methods for classs `RegClassTree' -----
//
const double         RegClassTree::SplitFactor   = 1.0;
SpeakerList*         RegClassTree::spkrList      = NULL;
TransformerTreeList* RegClassTree::spkrTransList = NULL;
BaseRegClassKMeans*  RegClassTree::kMeans        = NULL;

RegClassTree::RegClassTree(CodebookSetFastSATPtr& cb, const String& gCovFile,
			   int trace,
			   const String& spkrParamFile,
			   const String& spkrMeanVarStatsFile,
			   const String& spkrListFile)
  : BaseTree(cb)
{
  typedef set<unsigned> RCSet;
  typedef RCSet::const_iterator RCSetIter;

  // determine the set of leaf nodes
  RCSet regClassSet;
  for (CodebookSetFastSAT::GaussianIterator itr(cbs()); itr.more(); itr++) {
    GaussDensity mix(itr.mix());
    regClassSet.insert(mix.regClass());
  }

  // create leaf and internal nodes
  for (RCSetIter itr = regClassSet.begin(); itr != regClassSet.end(); itr++) {
    UnShrt rClass = (*itr);
    NodePtr leaf(new Node(*this, rClass, cbs()));
    _setNode(rClass, leaf);

    if (Trace & NodeClusters)
      printf("Node %4d: No. Gaussian Components %d\n",
	     leaf->index(), leaf->nComps());

    for (UnShrt index = rClass / 2; index > 0; index /= 2) {
      if (_nodePresent(index)) continue;

      NodePtr internalNode(new Node(*this, index, cbs(), Internal));
      _setNode(index, internalNode);
    }
  }

  // load mean and variance statistics
  if (spkrParamFile != "" && spkrMeanVarStatsFile != "" && spkrListFile != "") {

    spkrList      = new SpeakerList(spkrListFile);
    spkrTransList =
      new TransformerTreeList(cbs(), spkrParamFile, /* orgSubFeatLen= */ 0, spkrList);

    kMeans = new SATRegClassLhoodRatio(cbs(), *spkrTransList, *spkrList,
				       spkrMeanVarStatsFile, gCovFile, trace);
  } else {
    kMeans = new HTKRegClassKMeans(cbs(), gCovFile, trace);
  }

  // calculate distributions and cluster scores of leaf nodes
  for (_NodeListIter itr = _nodeList.begin(); itr != _nodeList.end(); itr++) {
    NodePtr& node(Cast<NodePtr>((*itr).second));
    if (node->type() != Leaf) continue;
    node->calcClusterDistribution();
    node->calcNodeScore();
  }
}

RCNodePtr& RegClassTree::operator[](short idx)
{
  return Cast<RCNodePtr>(node(idx));
}

RegClassTree::~RegClassTree()
{
  delete spkrList;	spkrList      = NULL;
  delete spkrTransList;	spkrTransList = NULL;
  delete kMeans;	kMeans        = NULL;
}

ostream& operator<<(ostream& os, RegClassTree& rct)
{  
#if 0
  RegList regList(t, /* onlyLeaves= */ false);
  for (RegList::Iterator itr(regList); itr.more(); itr++) {
    RCNodePtr& n(itr.rNode());
    if (itr.regTree()->left == NULL) {
      fprintf(fp, "<TNODE> %d %d\n",
        n->nodeIndex, n->nComponents );
    } else {
      fprintf(fp, "<NODE> %d ", n->nodeIndex);
      RCNodePtr& left(GetRNode(itr.regTree()->left));
      RCNodePtr& right(GetRNode(itr.regTree()->right));
      fprintf(fp, "%d ",  left->nodeIndex);
      fprintf(fp, "%d\n", right->nodeIndex);
    }
  }
#endif
  return os;
}

void RegClassTree::increment(UnShrt toAdd, bool onlyTopOne)
{
  cout << "Adding " << toAdd << " nodes ..." << endl;
  for (UnShrt cl = 0; cl < toAdd; cl++) {
    NodePtr bestLeaf(bestLeaf2Split());

    if (bestLeaf.isNull())
      throw j_error("No more nodes to split ...");

    UnShrt bestIndex = bestLeaf->index();
    UnShrt index1    = 2 * bestIndex;      NodePtr ch1(new Node(*this, index1, cbs()));
    UnShrt index2    = 2 * bestIndex + 1;  NodePtr ch2(new Node(*this, index2, cbs()));

    bestLeaf->split(ch1, ch2, onlyTopOne);
    _setNode(index1, ch1);  _setNode(index2, ch2);
    replaceIndex(cbs(), bestIndex, index1, index2);
  }
}

void incrementRegClasses(CodebookSetFastSATPtr& cbs, unsigned toAdd, int trace,
			 const String& spkrParamFile,
			 const String& spkrMeanVarStats,
			 const String& spkrListFile,
			 const String& gCovFile,
			 bool onlyTopOne)
{
  RegClassTree tree(cbs, gCovFile, trace, spkrParamFile, spkrMeanVarStats, spkrListFile);
  tree.increment(toAdd, onlyTopOne);
}

RegClassTree::NodePtr
RegClassTree::_bestLeaf(const NodePtr& node, float& score, NodePtr best)
{
  if (node->type() == Leaf) {

    float nodeScore = node->clusterScore();
    if ((score < nodeScore) && (node->nComps() > featLen() * SplitFactor)) {
      score = nodeScore;  best = node;
    }

  } else {

    best = _bestLeaf(leftChild(node),  score, best);
    best = _bestLeaf(rightChild(node), score, best);

  }

  return best;
}

RegClassTree::NodePtr RegClassTree::bestLeaf2Split()
{
  float    score     = 0.0;
  // float    score     = -HUGE;
  UnShrt   rootIndex = 1;
  NodePtr& rootNode(Cast<NodePtr>(node(rootIndex)));
  NodePtr best;

  return _bestLeaf(rootNode, score, best);
}


// ----- methods for classs `RegClassTree::Node' -----
//
double         RegClassTree::Node::RCPerturbation       = 0.1;
const double   RegClassTree::Node::Threshold            = 1.0e-04;
const unsigned RegClassTree::Node::MaxClusterIterations = 10;

RegClassTree::Node::Node(RegClassTree& tree, UnShrt indx, CodebookSetFastSATPtr& cb, NodeType typ)
  : BaseTree::Node(tree, indx, typ),
    _aveMean(cb->orgFeatLen(), cb->nSubFeat()),
    _aveCovar(cb->featLen(),   cb->nSubFeat()),
    _clusterScore(0.0), _clustAcc(0.0),
    _list(cb, indx)
{
  printf("RegClassTree::Node index = %d\n", index());

  printf("Average Mean:\n");
  printf("featLen    = %d\n", _aveMean.featLen());
  printf("subFeatLen = %d\n", _aveMean.subFeatLen());  
  printf("nSubFeat   = %d\n", _aveMean.nSubFeat());  

  printf("Average Covariance:\n");
  printf("featLen    = %d\n", _aveCovar.featLen());
  printf("subFeatLen = %d\n", _aveCovar.subFeatLen());  
  printf("nSubFeat   = %d\n", _aveCovar.nSubFeat());  
  fflush(stdout);
}

void RegClassTree::Node::print() const
{
  printf("For cluster %d there are:\n", index());
  printf("%d components with %f occupation count;\n  Cluster score of %e\n",
	 nComps(), _clustAcc, _clusterScore);

  cout << "MEAN:-"       << endl
       << _aveMean       << endl;

  cout << "COVARIANCE:-" << endl
       << _aveCovar      << endl;
}

void RegClassTree::Node::perturb(float pertDepth)
{  
  UnShrt nSbFt  = _aveCovar.nSubFeat();
  UnShrt sbFtLn = _aveCovar.subFeatLen();

  assert(nSbFt == _aveMean.nSubFeat());  assert(sbFtLn <= _aveMean.subFeatLen());

  for (UnShrt l = 0; l < nSbFt; l++)
    for (UnShrt k = 0; k < sbFtLn; k++)
      _aveMean(l, k) += pertDepth * sqrt(_aveCovar(l, k));
}

void RegClassTree::Node::clusterChildren(NodePtr& ch1, NodePtr& ch2, bool onlyTopOne)
{ 
  unsigned iter = 0;
  double oldDistance = 0.0, newDistance = 0.0;

  do {
    if (++iter == 1) {
      newDistance = calcDistance(ch1, ch2);
      oldDistance = 10.0 * newDistance;
    } else {
      oldDistance = newDistance;
      newDistance = calcDistance(ch1, ch2);
    }
    if (Trace & NodeClusters) {
      if (iter == 1)
	printf("Iteration %d: Distance = %e\n", iter, newDistance);
      else
	printf("Iteration %d: Distance = %e, Delta = %e\n",
	       iter, newDistance, oldDistance - newDistance);
      printf("Total Frames = %e\n", ch1->_clustAcc + ch2->_clustAcc);
      fflush(stdout);
    }
  } while (2.0 * (oldDistance - newDistance) /
	   (oldDistance + newDistance) > Threshold
	   && iter < MaxClusterIterations);

  if (Trace & NodeClusters)
    printf("Cluster 1: Score %e, Occ %e\nCluster 2: Score %e, Occ %e\n",
	   ch1->_clusterScore, ch1->_clustAcc, ch2->_clusterScore, ch2->_clustAcc);

  createChildNodes(ch1, ch2, onlyTopOne);
}

void RegClassTree::Node::split(NodePtr& ch1, NodePtr& ch2, bool onlyTopOne)
{
  if (Trace & NodeBID) {
    printf("Splitting Node %d, score %e\n", _index, _clusterScore);
    fflush(stdout);
  }

  printf("Splitting Node %d, score %e\n", _index, _clusterScore);

  printf("Average Mean:\n");
  cout << _aveMean << endl;

  printf("Average Covariance:\n");
  cout << _aveCovar << endl;
  
  ch1->_aveMean  = ch2->_aveMean  = _aveMean;
  ch1->_aveCovar = ch2->_aveCovar = _aveCovar;

  ch1->perturb( RCPerturbation);
  ch2->perturb(-RCPerturbation);

  printf("Average Mean1:\n");
  cout << ch1->_aveMean << endl;

  printf("Average Covariance1:\n");
  cout << ch1->_aveCovar << endl;

  printf("Average Mean2:\n");
  cout << ch2->_aveMean << endl;

  printf("Average Covariance2:\n");
  cout << ch2->_aveCovar << endl;

  clusterChildren(ch1, ch2, onlyTopOne);

  _setType(Internal);
}

void RegClassTree::Node::createChildNodes(NodePtr& ch1, NodePtr& ch2, bool onlyTopOne)
{
  int index1 = ch1->index();
  int index2 = ch2->index();

  int numLeft = 0, numRight = 0;
  for (GaussListFastSAT::Iterator itr(_list); itr.more(); itr++) {
    GaussDensity& mix(itr.mix());
    float score1 = ch1->distance(mix), score2 = ch2->distance(mix);
    if (score1 < score2) {
      ch1->_list.add(mix);  numLeft++;

      if (onlyTopOne)
	mix.setRegClass(index1);
      else
	mix.replaceRegClass(index1, index2);
    } else {
      ch2->_list.add(mix);  numRight++; 

      if (onlyTopOne)
	mix.setRegClass(index2);
      else
	mix.replaceRegClass(index2, index1);
    }
  }

  if (Trace & NodeClusters) {
    printf("Node %4d: No. Gaussian Components %d\n",
	   ch1->index(), ch1->nComps());
    printf("Node %4d: No. Gaussian Components %d\n",
	   ch2->index(), ch2->nComps());
  }

  ch1->calcClusterDistribution();    // calculate mean and variance
  ch2->calcClusterDistribution();    // of the clusters
  ch1->calcNodeScore();
  ch2->calcNodeScore();
}

void RegClassTree::Node::calcClusterDistribution()
{
  if (kMeans == NULL)
    throw j_error("Regression class K-means object not allocated.");

  kMeans->calcClusterDistribution(this);
}

float RegClassTree::Node::calcNodeScore()
{
  if (kMeans == NULL)
    throw j_error("Regression class K-means object not allocated.");

  _clusterScore = kMeans->calcNodeScore(this);

  if (_clusterScore < 0.0)
    // printf("Warning: Cluster score (%g) < 0.0.\n", _clusterScore);
    throw jconsistency_error("Cluster score (%g) < 0.0.\n", _clusterScore);

  return _clusterScore;
}

double RegClassTree::Node::calcDistance(NodePtr& ch1, NodePtr& ch2)
{
  if (kMeans == NULL)
    throw j_error("Regression class K-means object not allocated.");

  return kMeans->calcDistance(this->_list, ch1, ch2);
}

float RegClassTree::Node::distance(const GaussDensity& pdf)
{
  if (kMeans == NULL)
    throw j_error("Regression class K-means object not allocated.");

  return kMeans->distance(pdf, this);
}
