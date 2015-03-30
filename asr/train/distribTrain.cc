//
//			         Millennium
//                    Automatic Speech Recognition System
//                                  (asr)
//
//  Module:  asr.train
//  Purpose: Speaker adaptation parameter estimation and conventional
//           ML and discriminative HMM training.
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
#include "train/distribTrain.h"
#include "common/mach_ind_io.h"


// ----- methods for class `DistribTrain' -----
//
DistribTrain::DistribTrain(const String& nm, CodebookTrainPtr& cb)
  : DistribBasic(nm, cb), _mixVal(new float[valN()]), _accu(new Accu(valN())) { }

DistribTrain::DistribTrain(const DistribTrain& ds)
  : DistribBasic(ds), _mixVal(new float[valN()]), _accu(new Accu(valN())) { }

DistribTrain::~DistribTrain()
{
  delete[] _mixVal;
}

DistribTrain& DistribTrain::operator=(const DistribTrain& ds)
{
  _mixVal = new float[ds.valN()];
  _accu   = new Accu(ds.valN());

  return *this;
}

void DistribTrain::saveAccu(FILE* fp) const
{
  if (accu().isNull())
    throw jio_error("Distribution accumulator is NULL.");

  accu()->save(fp);
}

void DistribTrain::loadAccu(FILE* fp, float addFactor)
{
  if (accu().isNull())
    throw j_error("Distribution accumulator %s is NULL.", name().c_str());

  // cout << "Loading accumulator for distribution " << name() << endl;
  accu()->load(fp, addFactor);
}

void DistribTrain::zeroAccu()
{
  accu()->zero();
}

float DistribTrain::accumulate(int frameX, float factor)
{
  float score = cbk()->accumulate(factor, val(), frameX, _mixVal);
  accu()->accumulate(factor, _mixVal);

  return score;
}

void DistribTrain::update()
{
  _accu->update(val(), count());
}

void DistribTrain::updateMMI(bool merialdo)
{
  _accu->updateMMI(val(), count(), merialdo);
}

void DistribTrain::split(float minCount, float splitFactor)
{
  unsigned refX = cbk()->split(minCount, splitFactor);

  float* oldVal = _val;
  _val = new float[_valN + 1];
  memcpy(_val, oldVal, _valN * sizeof(float));
  _val[_valN]  = _val[refX] + log(2.0);
  _val[refX]  += log(2.0);
  delete[] oldVal;

  delete[] _mixVal;
  _mixVal = new float[_valN + 1];

  _accu = new Accu(_valN);

  _valN++;
}


// ----- methods for class `DistribTrain::Accu' -----
//
DistribTrain::Accu::Accu(UnShrt valN, UnShrt subN)
  : _subN(subN), _valN(valN), _subX(0)
{
  _count    = new double*[_subN];
  _denCount = new double*[_subN];
  for (UnShrt isub = 0; isub < _subN; isub++) {
    _count[isub]    = new double[_valN];
    _denCount[isub] = new double[_valN];
  }
}

DistribTrain::Accu::~Accu()
{
  for (UnShrt isub = 0; isub < _subN; isub++) {
    delete[] _count[isub];  delete[] _denCount[isub];
  }
  delete[] _count;  delete[] _denCount;
}

void DistribTrain::Accu::accumulate(float factor, float* mixVal)
{
  if (factor > 0.0) {
    for (UnShrt ival = 0; ival < _valN; ival++)
      _count[_subX][ival]    += factor * mixVal[ival];
  } else {
    for (UnShrt ival = 0; ival < _valN; ival++)
      _denCount[_subX][ival] -= factor * mixVal[ival];
  }
}

double PaddingValue = 1.0E-04;

void DistribTrain::Accu::update(float* val, float& count)
{
  double ttlCnt = 0.0;
  for (UnShrt ival = 0; ival < _valN; ival++)
    ttlCnt += (_count[_subX][ival] + PaddingValue);

  for (UnShrt ival = 0; ival < _valN; ival++)
    val[ival] = -log((_count[_subX][ival] + PaddingValue) / ttlCnt);

  count = ttlCnt;
}

void DistribTrain::Accu::updateMMI(float* val, float& count, bool merialdo)
{
  if (merialdo)
    _updateMMIAlternate(val, count);
  else
    _updateMMI(val, count);
}

void DistribTrain::Accu::_updateMMI(float* val, float& count)
{
  double C = PaddingValue;
  for (UnShrt ival = 0; ival < _valN; ival++) {
    double deriv = -(1.0 + PaddingValue) * ((_count[_subX][ival] - _denCount[_subX][ival]) * exp(val[ival]));
    if (deriv > C) C = deriv;
  }

  double ttlCnt = 0.0;
  for (UnShrt ival = 0; ival < _valN; ival++) {
    double cnt = (_count[_subX][ival] - _denCount[_subX][ival]) + C * exp(-val[ival]);
    if (cnt < 0.0)
      throw jconsistency_error("Count (%g) is negative; C = %g.\n", cnt, C);
    ttlCnt += cnt;
  }

  for (UnShrt ival = 0; ival < _valN; ival++) {
    double cnt = (_count[_subX][ival] - _denCount[_subX][ival]) + C * exp(-val[ival]);
    val[ival] = -log(cnt / ttlCnt);
    if (isnan(val[ival]))
      throw jconsistency_error("Weight is NaN : ttlCnt = %g.\n", ttlCnt);
  }

  count = ttlCnt;
}

void DistribTrain::Accu::_updateMMIAlternate(float* val, float& count)
{
  double ttlNumCount = 0.0, ttlDenCount = 0.0;
  for (UnShrt ival = 0; ival < _valN; ival++) {
    ttlNumCount += _count[_subX][ival] + PaddingValue;
    ttlDenCount += _denCount[_subX][ival] + PaddingValue;
  }

  double C = 0.0;
  for (UnShrt ival = 0; ival < _valN; ival++) {
    double deriv = -(1.0 + PaddingValue) * (((_count[_subX][ival] + PaddingValue) / ttlNumCount) - ((_denCount[_subX][ival] + PaddingValue) / ttlDenCount));
    if (deriv > C) C = deriv;
  }

  double ttlCnt = 0.0;
  for (UnShrt ival = 0; ival < _valN; ival++) {
    double cnt = (((_count[_subX][ival] + PaddingValue) / ttlNumCount) - ((_denCount[_subX][ival] + PaddingValue) / ttlDenCount) + C) * exp(-val[ival]) + PaddingValue;
    if (cnt < 0.0)
      throw jconsistency_error("Count (%g) < 0.0 : C = %g.\n", cnt, C);
    ttlCnt += cnt;
  }

  for (UnShrt ival = 0; ival < _valN; ival++) {
    double cnt = (((_count[_subX][ival] + PaddingValue) / ttlNumCount) - ((_denCount[_subX][ival] + PaddingValue) / ttlDenCount) + C) * exp(-val[ival]) + PaddingValue;
    val[ival] = -log(cnt / ttlCnt);
    if (isnan(val[ival]))
      throw jconsistency_error("Weight is NaN : ttlCnt = %g.\n", ttlCnt);
  }

  count = ttlCnt;
}

void DistribTrain::Accu::zero()
{
  for (UnShrt ival = 0; ival < _valN; ival++)
    _count[_subX][ival] = _denCount[_subX][ival] = 0.0;
}

bool DistribTrain::Accu::zeroOccupancy() const
{
  for (UnShrt subX = 0; subX < _subN; subX++)
    for (UnShrt valX = 0; valX < _valN; valX++)
      if (_count[subX][valX] > 0.0 || _denCount[subX][valX] > 0.0) return false;

  return true;
}

void DistribTrain::Accu::save(FILE* fp) const
{
  write_int(fp, _valN);			// size of the distribution
  write_int(fp, _subN);			// number of subaccumulators

  for (UnShrt subX = 0; subX < _subN; subX++) {
    for (UnShrt valX = 0; valX < _valN; valX++) {
      write_float(fp, (float) _count[subX][valX]);
      write_float(fp, (float) _denCount[subX][valX]);
    }
  }

  _dumpMarker(fp);
}

void DistribTrain::Accu::add(AccuPtr& ac, double factor)
{
  for (UnShrt subX = 0; subX < _subN; subX++) {
    for (UnShrt valX = 0; valX < _valN; valX++) {
      _count[subX][valX]    += ac->_count[subX][valX];
      _denCount[subX][valX] += ac->_denCount[subX][valX];
    }
  }
}

const int DistribTrain::Accu::MarkerMagic = 987654321;

// dumpMarker: dump a marker into file fp
void DistribTrain::Accu::_dumpMarker(FILE* fp) const
{
  write_int(fp, MarkerMagic);
}

// checkMarker: check if file fp has a marker next
void DistribTrain::Accu::_checkMarker(FILE* fp) const
{
  if (read_int(fp) != MarkerMagic)
    throw j_error("CheckMarker: Marker expected in dump file.");
}

void DistribTrain::Accu::_dontRead(FILE* fp, int valN, int subN)
{
  for (UnShrt subX = 0; subX < subN; subX++) {
    for (UnShrt valX = 0; valX < valN; valX++) {
      read_float(fp);  read_float(fp);
    }
  }

  _checkMarker(fp);
}

void DistribTrain::Accu::load(FILE* fp, float addFactor)
{
  int valN = read_int(fp);
  int subN = read_int(fp);

  if (valN != _valN) {
    // _dontRead(fp, valN, subN);
    throw jdimension_error("Number of values do not match (%d vs. %d)", valN, _valN);
  }
  if (subN != _subN) {
    // _dontRead(fp, valN, subN);
    throw jdimension_error("Number of sub-accumulators do not match (%d vs. %d)", subN, _subN);
  }

  for (UnShrt subX = 0; subX < _subN; subX++) {
    for (UnShrt valX = 0; valX < _valN; valX++) {
      _count[subX][valX]    += addFactor * read_float(fp);
      _denCount[subX][valX] += addFactor * read_float(fp);
    }
  }

  _checkMarker(fp);
}

unsigned DistribTrain::Accu::size() const
{
  // store the counts as floats and the marker as int
  return 2 * _subN * _valN * sizeof(float) + 3 * sizeof(int);
}

double DistribTrain::Accu::counts() const
{
  double cnt = 0.0;
  for (unsigned i = 0; i < _valN; i++)
    cnt += _count[_subX][i];

  return cnt;
}


// ----- methods for class `AccMap' -----
//
const long int AccMap::NotPresent = -1;

bool AccMap::_readName(FILE* fp, char* name)
{
  read_string(fp, name);

  return (strcmp(name, endOfAccs())) ? true : false;
}

static const int MaxStrLen = 512;

AccMap::AccMap(FILE* fp)
{
  static char name[MaxStrLen];

  while (_readName(fp, name)) {
    if (_stateMap.find(name) != _stateMap.end())
      throw jkey_error("State map name %s is not unique.", name);

    int pos = read_int(fp);

    _stateMap.insert(_ValueType(name, pos));
  }
}

AccMap::AccMap(const CodebookSetTrain& cb, bool countsOnlyFlag)
{
  long int offset = 0;

  // determine locations of state accumulators
  for (CodebookSetBasic::_ConstIterator itr(cb._cblist); itr.more(); itr++) {
    CodebookTrainPtr& cbk(Cast<CodebookTrainPtr>(*itr));

    if (cbk->accu()->zeroOccupancy()) continue;

    if (_stateMap.find(cbk->name()) != _stateMap.end())
      throw jkey_error("Entry for %s is not unique.", cbk->name().c_str());

    _stateMap.insert(_ValueType(cbk->name(), offset));
    offset += cbk->accu()->size(cbk->name(), countsOnlyFlag);
  }
}

AccMap::AccMap(const DistribSetTrain& dss)
{
  long int offset = 0;

  // determine locations of state accumulators
  for (DistribSetBasic::_ConstIterator itr(dss._list); itr.more(); itr++) {
    DistribTrainPtr& ds(Cast<DistribTrainPtr>(*itr));

    if (ds->accu()->zeroOccupancy()) continue;

    if (_stateMap.find(ds->name()) != _stateMap.end())
      throw jkey_error("Entry for %s is not unique.", ds->name().c_str());

    // cout << "Distribution " << ds->name() << " size " << ds->accu()->size() << endl;

    _stateMap.insert(_ValueType(ds->name(), offset));
    offset += ds->accu()->size();
  }
}

void AccMap::_stampEndOfAccs(FILE* fp)
{
  write_string(fp, endOfAccs());
}

void AccMap::write(FILE* f)
{
  // write locations of individual states
  for (_AccumMapIter itr = _stateMap.begin(); itr != _stateMap.end(); itr++) {
    const char* first = (*itr).first;
    int second = (*itr).second;

    write_string(f, first);
    write_int(f, second);
  }
  _stampEndOfAccs(f);
}

long int AccMap::state(const String& name)
{
  _AccumMapIter itr = _stateMap.find(name);
  if (itr == _stateMap.end()) return NotPresent;

  return (*itr).second;
}


// ----- methods for class `DistribSetTrain' -----
//
DistribSetTrain::DistribSetTrain(CodebookSetTrainPtr& cb, const String& descFile,
				 const String& distFile)
  : DistribSetBasic(cb)
{
  if (descFile == "") return;

  freadAdd(descFile, ';', this);

  if (distFile == "")
    _initializeFromCodebooks();
  else
    load(distFile);

}

void DistribSetTrain::saveAccus(const String& fileName) const
{
  FILE* fp = fileOpen(fileName, "w");

  cout << "Saving distribution accumulators to '" << fileName << "'." << endl;

  AccMap accMap(*this);
  accMap.write(fp);

  long int startOfAccs = ftell(fp);

  for (_ConstIterator itr(_list); itr.more(); itr++) {
    DistribTrainPtr& ds(Cast<DistribTrainPtr>(*itr));

    if (ds->accu()->zeroOccupancy()) continue;

    // cout << "Distribution " << ds->name() << " current = " << (ftell(fp) - startOfAccs) << endl;

    ds->saveAccu(fp);
  }

  fileClose(fileName, fp);
}

void DistribSetTrain::loadAccus(const String& fileName, unsigned nParts, unsigned part)
{
  FILE* fp = fileOpen(fileName, "r");

  // cout << "Loading distribution accumulators from '" << fileName << "'." << endl;

  AccMap accMap(fp);
  long int startOfAccs = ftell(fp);

  for (_Iterator itr(_list, nParts, part); itr.more(); itr++) {
    DistribTrainPtr& ds(Cast<DistribTrainPtr>(*itr));
    long int current = accMap.state(ds->name());

    if (current == AccMap::NotPresent) continue;

    // cout << "Distribution " << ds->name() << " current = " << current << endl;

    current += startOfAccs;

    if (current != ftell(fp))
      fseek(fp, current, SEEK_SET);

    ds->loadAccu(fp);
  }

  fileClose(fileName, fp);
}

void DistribSetTrain::zeroAccus(int nParts, int part)
{
  for (_Iterator itr(_list, nParts, part); itr.more(); itr++) {
    DistribTrainPtr& ds(Cast<DistribTrainPtr>(*itr));
    ds->zeroAccu();
  }
}

void DistribSetTrain::update(int nParts, int part)
{
  for (_Iterator itr(_list, nParts, part); itr.more(); itr++) {
    DistribTrainPtr& ds(Cast<DistribTrainPtr>(*itr));
    ds->update();
  }
}

void DistribSetTrain::updateMMI(int nParts, int part, bool merialdo)
{
  for (_Iterator itr(_list, nParts, part); itr.more(); itr++) {
    DistribTrainPtr& ds(Cast<DistribTrainPtr>(*itr));
    ds->updateMMI(merialdo);
  }
}

double DistribSetTrain::accumulate(DistribPathPtr& path, float factor)
{
  cbs()->resetCache();
  unsigned frameX = 0;
  double score = 0.0;
  for (DistribPath::Iterator itr(path); itr.more(); itr++)
    score += find(itr.name())->accumulate(frameX++, factor);

  printf("Finished accumulation for %d frames.\n", frameX);  fflush(stdout);

  return score;
}

void DistribSetTrain::accumulateLattice(LatticePtr& lat, float factor)
{
  cbs()->resetCache();
  unsigned cnt = 0;
  for (Lattice::EdgeIterator itr(lat); itr.more(); itr++) {
    Lattice::EdgePtr& edge(itr.edge());

    unsigned distX = edge->input();

    if (distX == 0) continue;

    LogDouble gamma = edge->data().gamma();

    if (gamma > 10.0) continue;

    double postProb = factor * exp(-gamma);

    DistribTrainPtr& ds(find(distX-1));

    cnt++;
    for (int frameX = edge->data().start(); frameX <= edge->data().end(); frameX++) {
      /*
      if (frameX < 0) {
	printf("Accumulating for distribution %s with posterior prob %g : Frame %d to %d.\n",
	       ds->name().c_str(), postProb, edge->data().start(), edge->data().end());  fflush(stdout);
	continue;
      }
      */

      ds->accumulate(frameX, postProb);
    }
  }
  printf("Finished accumulation for %d links.\n", cnt);  fflush(stdout);
}

void DistribSetTrain::_addDist(const String& name, const String& cbname)
{
  CodebookTrainPtr& cb(cbs()->find(cbname));
  DistribTrain* ptr = new DistribTrain(name, cb);

  _list.add(name, DistribTrainPtr(ptr));
}
