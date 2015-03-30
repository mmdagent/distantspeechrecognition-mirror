//
//                                Millenium
//                   Automatic Speech Recognition System
//                                  (asr)
//
//  Module:  asr.natural
//  Purpose: Common operations.
//  Author:  John McDonough

#include <iostream>
#include <math.h>
#include <set>
#include <common/mach_ind_io.h>
#include "natural/natural.h"


// ----- members for class `NaturalIndex' -----
//
      UnShrt   NaturalIndex::_nSubFeat         = 0;
const UnShrt   NaturalIndex::_MaxNoVectorSizes = 400;
const UnShrt*  NaturalIndex::_Single[_MaxNoVectorSizes];
const UnShrt** NaturalIndex::_Double[_MaxNoVectorSizes];

NaturalIndex::NaturalIndex(UnShrt _size, UnShrt nsub)
{
  if (_size < 1)
    throw jdimension_error("Must specify non-zero size for NaturalIndex.");

  if (nsub != 0 && (_size / nsub) >= _MaxNoVectorSizes)
    throw jdimension_error("Sub-feature size %d is not supported.", _size);

  initialize(nsub);
}

// initialize natural indices for all sub-feature sizes
//
UnShrt NaturalIndex::initialize(UnShrt nsub)
{
  assert(nsub <= 3);

  if (_nSubFeat != 0) {
    if (nsub > _nSubFeat)
      throw jconsistency_error("No. of sub-features is inconsistent.");
    return (nsub == 0) ? _nSubFeat : nsub;
  }
  if (nsub == 0)
    throw jdimension_error("Must specify non-zero number of sub-features.");

  _nSubFeat = nsub;
  for (UnShrt subsize = 1; subsize < _MaxNoVectorSizes; subsize++) {
    UnShrt  size   = _nSubFeat * subsize;
    UnShrt* single = new UnShrt[size];

    for (UnShrt icep = 0; icep < size; icep++) single[icep] = icep;

    _Single[subsize] = (const UnShrt*) single;

    UnShrt** dble = new UnShrt*[_nSubFeat];
    for (UnShrt isub = 0; isub < _nSubFeat; isub++) {
      dble[isub] = new UnShrt[subsize];

      for (UnShrt icep = 0; icep < subsize; icep++)
	dble[isub][icep] = icep + (isub * subsize);
    }
    _Double[subsize] = (const UnShrt**) dble;
  }

  return _nSubFeat;
}

void NaturalIndex::deinitialize()
{
  for (UnShrt subsize = 1; subsize < _MaxNoVectorSizes; subsize++) {
    delete[] _Single[subsize];

    for (UnShrt isub = 0; isub < _nSubFeat; isub++)
      delete[] _Double[subsize][isub];

    delete[] _Double[subsize];
  }
}
